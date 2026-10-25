import json
import logging
import os
import random
import re

import torch

import comfy.model_management
import comfy.sample
import comfy.samplers
import comfy.utils
import folder_paths
import nodes
from comfy.cli_args import args
from comfy_api.latest import InputImpl, Types, io, ui
from comfy_extras.nodes_audio import VAEDecodeAudio
from comfy_extras.nodes_custom_sampler import BasicGuider, BasicScheduler, Noise_RandomNoise
from comfy_extras.nodes_minimax_h3 import MiniMaxH3ReferenceToVideo, align_frame_count
from comfy_extras.nodes_resolution import AspectRatio, ResolutionSelector
from comfy_extras.nodes_video import CreateVideo

from . import MiniMaxH3LatentContinuationGuide, _av_streams, _copy_context_tail
from . import segment_review
from .latent_io import save_h3_latent
from .dynamic_mask import check_dynamic_mask_support, prepare_dynamic_mask
from .drift_correction import correct_video_drift, drift_reference
from .seam_preview import build_seam_preview
from .segment_review import (RECORD_FORMAT, annotated_path, load_record, load_retry_context,
                             record_entry, record_fingerprint, save_record)
from .sampling_preview import H3SamplingPreview


H3Prompts = io.Custom("H3_PROMPTS")


def _sample_h3(noise, guider, sampler, sigmas, latent, preview=None, segment=1):
    latent = latent.copy()
    latent["samples"] = comfy.sample.fix_empty_latent_channels(
        guider.model_patcher, latent["samples"], latent.get("downscale_ratio_spacial"),
        latent.get("downscale_ratio_temporal"))
    progress = comfy.utils.ProgressBar(sigmas.shape[-1] - 1)

    def callback(step, x0, x, total_steps):
        progress.update_absolute(step + 1, total_steps)
        if preview is not None:
            preview.update(segment, step, x0, total_steps, guider.model_patcher)

    samples = guider.sample(noise.generate_noise(latent), latent["samples"], sampler, sigmas,
                            denoise_mask=latent.get("noise_mask"), callback=callback,
                            disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED, seed=noise.seed)
    latent.pop("downscale_ratio_spacial", None)
    latent.pop("downscale_ratio_temporal", None)
    latent["samples"] = samples.to(comfy.model_management.intermediate_device())
    return io.NodeOutput(latent)


def _trim_audio(audio, trim_frames, frame_count):
    """Match the exported video duration, including H3's 40 Hz rounding gaps."""
    sample_rate = audio["sample_rate"]
    start = round(trim_frames / 24 * sample_rate)
    length = round(frame_count / 24 * sample_rate)
    waveform = audio["waveform"][..., start:start + length]
    if waveform.shape[-1] < length:
        padded = waveform.new_zeros((*waveform.shape[:-1], length))
        padded[..., :waveform.shape[-1]] = waveform
        waveform = padded
    return {**audio, "waveform": waveform}


def _reserve_retry_name(run_folder, segment):
    data_folder = os.path.join(run_folder, "_data")
    if not all(folder_paths.is_within_directory(folder_paths.get_output_directory(), path)
               for path in (run_folder, data_folder)):
        raise ValueError("重抽保存目录必须位于 ComfyUI output 内。")
    os.makedirs(data_folder, exist_ok=True)
    pattern = re.compile(rf"(?:segment|seam|context_preview)_{segment:04}_retry_(\d+)\..+", re.IGNORECASE)
    directories = (run_folder, data_folder)
    counter = max((int(match[1]) for directory in directories for name in os.listdir(directory)
                   if (match := pattern.fullmatch(name))), default=0) + 1
    while True:
        suffix = f"{segment:04}_retry_{counter:03}"
        reservation = os.path.join(data_folder, f"segment_{suffix}.reserve")
        try:
            with open(reservation, "x"):
                pass
        except FileExistsError:
            counter += 1
            continue
        collision = any(os.path.join(directory, name) != reservation
                        and (match := pattern.fullmatch(name)) and int(match[1]) == counter
                        for directory in directories for name in os.listdir(directory))
        if not collision:
            return suffix, reservation
        os.remove(reservation)
        counter += 1


class H3VideoPromptPlan(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3VideoPromptPlan",
            display_name="H3 长视频提示词",
            category="model/conditioning/minimax",
            description="共用描述加上各段剧情。独占一行的 --- 分隔各段；可选择从第几段开始，或复用最后一段。",
            inputs=[
                io.String.Input("common", display_name="全片固定描述", multiline=True, default=""),
                io.String.Input("segments", display_name="分段剧情（--- 分隔）", multiline=True, default=""),
                io.Boolean.Input("repeat_last", display_name="不足时复用最后一段", default=False,
                                 tooltip="开启后可只写一段提示词，让所有片段使用它；多段时不足的部分复用最后一段。"),
                io.Int.Input("start_segment", display_name="起始提示词段号", default=1, min=1, max=10000,
                             tooltip="段号从1开始。正常全片选1；重抽第3段选3，生成节点生成段数设1。多段生成从此段依次向后取提示词。"),
            ],
            outputs=[H3Prompts.Output(display_name="提示词")],
        )

    @classmethod
    def execute(cls, common, segments, repeat_last=False, start_segment=1):
        sections = [part.strip() for part in re.split(r"(?m)^[ \t]*---[ \t]*\r?$", segments)]
        if segments.strip() and any(not section for section in sections):
            raise ValueError("分段剧情中有空段，请补写该段或删除多余的 ---。")
        prompts = ["\n\n".join(part for part in (common.strip(), section) if part) for section in sections]
        if start_segment > len(prompts) and repeat_last:
            # Every segment past the written ones reuses the last prompt.
            start_segment = len(prompts)
        if not 1 <= start_segment <= len(prompts):
            raise ValueError(f"第 {start_segment} 段提示词不存在，目前共 {len(prompts)} 段（段号从1开始）。")
        return io.NodeOutput({"prompts": prompts[start_segment - 1:], "repeat_last": repeat_last})


class H3LongVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3LongVideo",
            display_name="H3 长视频：分段生成 + 合并 by神棍",
            category="video/minimax",
            search_aliases=["H3 一体", "H3 long video", "潜空间长视频"],
            description="每段生成完成后立即保存到output，可选保存完整音视频LATENT。合并视频连接外部Save Video时才实际合并、保存；不连接也可正常生成和保存分段。",
            inputs=[
                io.Model.Input("model"),
                io.Clip.Input("clip"),
                io.Vae.Input("vae", display_name="视频 VAE"),
                io.Vae.Input("audio_vae", display_name="音频 VAE"),
                H3Prompts.Input("prompts", display_name="提示词"),
                io.Int.Input("segments", display_name="生成段数", default=4, min=1, max=10000,
                             tooltip="本次新生成的总段数，1生成一段，2生成两段。连接上一段LATENT时，从它的结尾生成；输入段不计入段数，也不加入本次导出。"),
                io.Float.Input("duration", display_name="每段时长", default=5.0, min=0.01, max=150.0, step=0.1,
                               tooltip="沿用官方 24fps 和 17k+5 帧公式，实际时长允许略有出入。"),
                io.Combo.Input("aspect_ratio", display_name="宽高比", options=AspectRatio,
                               default=AspectRatio.WIDESCREEN_H),
                io.Float.Input("megapixels", display_name="百万像素", default=0.4, min=0.1, max=16.0, step=0.1),
                io.Int.Input("seed", display_name="种子", default=0, min=0, max=0xffffffffffffffff,
                             control_after_generate=io.ControlAfterGenerate.randomize),
                io.Int.Input("steps", display_name="采样步数", default=8, min=1, max=10000,
                             tooltip="未连接自定义 sigmas 时使用；连接后，实际步数由 sigmas 序列决定。"),
                io.Combo.Input("sampler_name", display_name="采样器", options=comfy.samplers.SAMPLER_NAMES,
                               default="er_sde"),
                io.Combo.Input("scheduler", display_name="调度器", options=comfy.samplers.SCHEDULER_NAMES,
                               default="simple", tooltip="连接自定义 sigmas 时跳过此调度器。"),
                io.Int.Input("context_frames", display_name="上下文帧数", default=22, min=5, max=3600, step=17,
                             advanced=True, tooltip="直接读取上一段音视频 latent 的尾部。续段导出时自动删去重叠部分。"),
                io.Combo.Input("seed_mode", display_name="各段种子", options=io.ControlAfterGenerate,
                               default="increment", advanced=True,
                               tooltip="首段使用上方种子，后续段按fixed固定、increment递增、decrement递减或randomize随机。randomize且生成多段时，每次排队都会重新随机后续段，即使首段种子固定。生成信息记录实际种子。"),
                io.Float.Input("denoise", display_name="降噪", default=1.0, min=0.01, max=1.0, step=0.01, advanced=True,
                               tooltip="只用于内部调度器；连接自定义 sigmas 时直接使用输入序列。"),
                io.Combo.Input("ref_image_size", display_name="参考图尺寸", options=["match", "max"],
                               default="match", advanced=True),
                io.Boolean.Input("save_latents", display_name="保存每段 LATENT", default=False,
                                 tooltip="每段完成后在批次的_data目录保存完整音视频.h3latent及清单，可用本插件的H3读取音视频LATENT节点恢复续接。增加磁盘占用，不在内存累积各段LATENT。"),
                io.String.Input("filename_prefix", display_name="保存前缀", default="H3",
                                tooltip="正常生成默认在output/H3_00001等编号目录逐段保存正式MP4；辅助文件集中在_data。重抽output中的片段时保存在原批次并忽略此前缀，文件名按原段号和重抽次数递增；合并视频文件名仍由外部Save Video控制。"),
                io.Boolean.Input("dynamic_mask", display_name="动态掩码", default=True,
                                 tooltip="仅续接时生效。按实际sigma动态重绘上下文，保护接缝及音频前缀；用于对照漂移控制，不保证恢复丢失细节。需原生H3音视频掩码支持，不能叠加其他动态掩码。"),
                io.Autogrow.Input("ref_images", display_name="参考图", optional=True,
                    template=io.Autogrow.TemplatePrefix(io.Image.Input("ref_image"), prefix="ref_image_", min=0, max=9)),
                io.Sigmas.Input("sigmas", optional=True,
                                tooltip="连接 Manual Sigmas 等节点后，全部片段直接使用此序列，覆盖内部调度器、步数和降噪设置。"),
                io.Autogrow.Input("ref_videos", display_name="参考视频", optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        io.Image.Input("ref_video", tooltip="与官方 H3 一致，输入24fps的IMAGE帧批次，至少5帧。"),
                        prefix="ref_video_", min=0, max=3)),
                io.Autogrow.Input("ref_video_audios", display_name="参考视频音轨", optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        io.Audio.Input("ref_video_audio", tooltip="按相同编号配对：ref_video_audio_0 对应 ref_video_0。"),
                        prefix="ref_video_audio_", min=0, max=3)),
                io.Autogrow.Input("ref_audios", display_name="独立参考音频", optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        io.Audio.Input("ref_audio", tooltip="独立参考音频，沿用官方 H3 的 <Audio j> 编号。"),
                        prefix="ref_audio_", min=0, max=3)),
                io.Latent.Input("previous_latent", display_name="上一段 LATENT", optional=True,
                                tooltip="连接H3读取音视频LATENT或上一生成节点的最后一段LATENT。必须包含完整音视频且分辨率相同；生成段数设1可单独重抽下一段。"),
                io.Boolean.Input("drift_correction", display_name="防偏色与亮度漂移", default=True,
                                 optional=True, advanced=True,
                                 tooltip="轻量校正续段视频LATENT的颜色与对比度统计，首段不改；音频、头部重叠和尾部续接上下文保持原样，不增加采样。适合同场景连续镜头，主动换光或换场景时建议关闭。"),
                io.String.Input("retry_segment", display_name="重抽片段记录（可留空）", default="",
                                optional=True, advanced=True,
                                tooltip="由下方片段面板自动填写，恢复该段续接上下文。output中的片段重抽保存回原批次，如segment_0002_retry_001.mp4；再次重抽仍沿用原段号并递增次数，不覆盖原版。清空后恢复正常生成。"),
                io.String.Input("retry_prompt", display_name="重抽提示词（留空沿用记录）", default="",
                                multiline=True, optional=True, advanced=True, placeholder="重抽提示词（留空沿用记录）",
                                tooltip="选择片段并恢复参数后可在此修改提示词；仅重抽模式使用，不改原来的全片提示词节点。"),
                io.String.Input("retry_seed", display_name="自定义重抽种子", default="",
                                optional=True, advanced=True,
                                tooltip="仅重抽种子方式选择“自定义种子”时生效，需填写0到18446744073709551615的完整整数。恢复时保留原记录种子，避免浏览器舍入。"),
                io.Boolean.Input("live_preview", display_name="预览采样（增加耗时）", default=False,
                                 optional=True, tooltip="使用本地 taeh3_temporal.safetensors 按时间顺序解码采样画面，在参数面板内覆盖播放。关闭不加载预览模型或生成采样缩略图。"),
                io.Int.Input("preview_interval", display_name="预览更新间隔（步）", default=2, min=1, max=10000,
                             optional=True, advanced=True, tooltip="每隔多少个采样步更新一次；首步和末步仍更新。数值越大，解码次数越少。"),
                io.Int.Input("preview_frames", display_name="每次预览帧数", default=12, min=1, max=48,
                             optional=True, advanced=True, tooltip="旧工作流兼容参数；面板隐藏且不再生效，预览帧数自动对齐本段实际导出帧数。"),
                io.Int.Input("preview_max_resolution", display_name="预览最长边", default=768, min=128, max=2048, step=16,
                             optional=True, advanced=True, tooltip="仅限制采样预览解码尺寸，不改变正式视频分辨率。"),
                io.Int.Input("preview_fps", display_name="预览帧率", default=10, min=1, max=60,
                             optional=True, tooltip="默认10，24与正式视频同速；调低慢放，调高快放。只改变预览播放速度，不改变正式视频的24fps。"),
                io.Combo.Input("retry_seed_mode", display_name="重抽种子方式", options=["randomize", "original", "custom"],
                               default="randomize", optional=True, advanced=True,
                               tooltip="仅重抽时生效。随机种子每次重新随机；原种子使用片段记录的实际种子；自定义种子使用下方填写的完整整数。"),
            ],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo, io.Hidden.unique_id],
            is_output_node=True,
            outputs=[io.Video.Output(display_name="合并视频",
                                     tooltip="外接Save Video时才实际合并并保存完整视频，无需开关。"),
                     io.Latent.Output(display_name="最后一段 latent"),
                     io.String.Output(display_name="生成信息"),
                     io.String.Output(display_name="LATENT 文件路径", is_output_list=True,
                                     tooltip="开启保存每段LATENT后，输出按段号排序的.h3latent路径；关闭时为空列表。")],
        )

    @classmethod
    def fingerprint_inputs(cls, segments=4, seed_mode="increment", retry_segment="", retry_seed_mode="randomize", **kwargs):
        if retry_segment is None:
            return float("nan")
        if retry_segment:
            if retry_seed_mode not in ("original", "custom"):
                return float("nan")
            return record_fingerprint(retry_segment)
        # Linked values are unavailable during fingerprinting; invalidate conservatively.
        may_randomize = seed_mode is None or seed_mode == "randomize"
        may_continue = segments is None or segments > 1
        return float("nan") if may_randomize and may_continue else segment_review.cleanup_revision

    @classmethod
    def _render_segment(cls, model, clip, vae, audio_vae, prompt, width, height, base_frames, previous,
                        context_frames, seed, sampler, sigmas, ref_images, ref_image_size,
                        video_path, metadata, ref_videos=None, ref_video_audios=None, ref_audios=None,
                        dynamic_mask=False, drift_stats=None, sampling_preview=None, segment=1):
        overlap = 0
        length = base_frames
        if previous is not None:
            _, _, previous_frames = _av_streams(previous, "上一段 latent")
            overlap = (min(context_frames, previous_frames) - 5) // 17 * 17 + 5
        if overlap:
            length = max(17, base_frames - 5) + overlap
        positive, latent = MiniMaxH3ReferenceToVideo.execute(
            clip=clip, vae=vae, audio_vae=audio_vae, prompt=prompt, width=width, height=height,
            length=length, ref_image_size=ref_image_size, ref_images=ref_images, ref_videos=ref_videos,
            ref_video_audios=ref_video_audios, ref_audios=ref_audios).result
        if previous is not None:
            positive, overlap = MiniMaxH3LatentContinuationGuide.execute(
                positive, latent, previous, overlap).result
            if dynamic_mask:
                model, latent, positive = prepare_dynamic_mask(model, latent, positive, sigmas)
        guider = BasicGuider.execute(model, positive)[0]
        try:
            if sampling_preview is not None:
                sampling_preview.set_segment(total_frames=length, trim_frames=overlap)
            preview_args = {"preview": sampling_preview, "segment": segment} if sampling_preview is not None else {}
            sampled = _sample_h3(Noise_RandomNoise(seed), guider, sampler, sigmas, latent, **preview_args)[0]
        finally:
            if dynamic_mask and previous is not None:
                model.model_options["denoise_mask_function"].__self__.current_mask = None
        # The dynamic-mask prefix lock belongs to this sampling run only; do not leak it
        # into the output LATENT, where another sampler would treat the prefix as locked.
        sampled.pop("noise_mask", None)
        del guider, positive, latent, previous, model
        comfy.model_management.throw_exception_if_processing_interrupted()
        if drift_stats is not None:
            sampled = correct_video_drift(sampled, drift_stats, overlap, context_frames)
        images = nodes.VAEDecode().decode(vae, sampled)[0]
        audio = VAEDecodeAudio.execute(audio_vae, sampled)[0]
        images = images[overlap:]
        audio = _trim_audio(audio, overlap, images.shape[0])
        video = CreateVideo.execute(images, 24, audio, bit_depth=8, color_space="sRGB")[0]
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        temporary = video_path + ".tmp"
        try:
            video.save_to(temporary, format=Types.VideoContainer("mp4"), codec=Types.VideoCodec("h264"), metadata=metadata)
            os.replace(temporary, video_path)
        finally:
            if os.path.exists(temporary):
                os.remove(temporary)
        return sampled, InputImpl.VideoFromFile(video_path), images.shape[0]

    @classmethod
    def execute(cls, model, clip, vae, audio_vae, prompts, segments, duration, aspect_ratio, megapixels,
                seed, steps, sampler_name, scheduler, context_frames=22, seed_mode="increment",
                denoise=1.0, ref_image_size="match", ref_images=None, ref_videos=None,
                ref_video_audios=None, ref_audios=None, sigmas=None,
                previous_latent=None, save_latents=False, filename_prefix="H3",
                dynamic_mask=True, drift_correction=True, retry_segment="", retry_prompt="", retry_seed="",
                live_preview=False, preview_interval=2, preview_frames=12, preview_max_resolution=768, preview_fps=10,
                retry_seed_mode="randomize"):
        retry = load_record(retry_segment) if retry_segment else None
        if retry is not None:
            previous_latent = load_retry_context(retry)
            segments = 1
            prompts = {"prompts": [retry_prompt if retry_prompt.strip() else retry["prompt"]], "repeat_last": False}
            if retry_seed_mode == "randomize":
                seed = random.randint(0, 0xffffffffffffffff)
            elif retry_seed_mode == "original":
                seed = retry["seed"]
            elif retry_seed_mode == "custom":
                if not isinstance(retry_seed, str) or not retry_seed.strip():
                    raise ValueError("自定义重抽种子需要填写完整整数，不能为空。")
                try:
                    seed = int(retry_seed)
                except ValueError as error:
                    raise ValueError("自定义重抽种子需要填写完整整数，不能为空。") from error
                if not 0 <= seed <= 0xffffffffffffffff:
                    raise ValueError("自定义重抽种子需在0到18446744073709551615之间。")
            else:
                raise ValueError("重抽种子方式需选择randomize、original或custom。")
            if retry["sigmas_source"] == "external":
                dtype = {"float16": torch.float16, "float32": torch.float32,
                         "float64": torch.float64, "bfloat16": torch.bfloat16}.get(retry.get("sigmas_dtype"))
                if dtype is None:
                    raise ValueError("片段记录的 SIGMAS 类型不受支持。")
                sigmas = torch.tensor(retry["sigmas"], dtype=dtype)
            else:
                sigmas = None
        count = segments
        texts = prompts["prompts"]
        if len(texts) < count and not prompts["repeat_last"]:
            raise ValueError(f"生成 {count} 段需要 {count} 段提示词，从所选起始段起只有 {len(texts)} 段；请补齐，或开启提示词节点的复用选项。")
        if dynamic_mask and (count > 1 or previous_latent is not None):
            check_dynamic_mask_support(model)
        width, height = ResolutionSelector.execute(aspect_ratio, megapixels, 32).result
        if previous_latent is not None:
            source_video, _, _ = _av_streams(previous_latent, "上一段 LATENT")
            if source_video.shape[3:] != (height // 16, width // 16):
                raise ValueError(f"上一段LATENT分辨率为{source_video.shape[4] * 16}×{source_video.shape[3] * 16}，本次为{width}×{height}；请使用相同分辨率。")
            del source_video
        base_frames = align_frame_count(max(5, round(duration * 24)))
        sampler = comfy.samplers.sampler_object(sampler_name)
        sigmas_source = "external" if sigmas is not None else "internal"
        if sigmas is None:
            sigmas = BasicScheduler.execute(model, scheduler, steps, denoise)[0]
        output_folder = folder_paths.get_output_directory()
        recipe_path = folder_paths.get_annotated_filepath(retry_segment) if retry is not None else None
        if recipe_path and folder_paths.is_within_directory(output_folder, recipe_path):
            run_folder = os.path.dirname(recipe_path)
            if (os.path.basename(run_folder) == "_data"
                    and folder_paths.is_within_directory(output_folder, os.path.dirname(run_folder))):
                run_folder = os.path.dirname(run_folder)
        else:
            folder, basename, counter, _, _ = folder_paths.get_save_image_path(
                filename_prefix, output_folder, width, height)
            while True:
                run_folder = os.path.join(folder, f"{basename}_{counter:05}")
                try:
                    os.mkdir(run_folder)
                    break
                except FileExistsError:
                    counter += 1
        subfolder = os.path.relpath(run_folder, folder_paths.get_output_directory()).replace(os.sep, "/")
        data_folder = os.path.join(run_folder, "_data")
        data_subfolder = f"{subfolder}/_data"
        metadata = None
        if not args.disable_metadata:
            metadata = dict(cls.hidden.extra_pnginfo or {})
            if cls.hidden.prompt is not None:
                metadata["prompt"] = cls.hidden.prompt
        drift_stats = None
        if retry is not None and drift_correction and retry.get("drift_reference") is not None:
            drift_stats = tuple(torch.tensor(values, dtype=torch.float32) for values in retry["drift_reference"])
        elif drift_correction and previous_latent is not None:
            drift_stats = drift_reference(previous_latent, context_frames)
        previous = _copy_context_tail(previous_latent, context_frames) if previous_latent is not None else None
        previous_video_name = (retry.get("context_video_path") if retry is not None else
                               previous_latent.get("_h3_video_path") if previous_latent is not None else None)
        videos, frame_counts, segment_seeds, continuation_methods = [], [], [], []
        video_paths, latent_results, latent_paths = [], [], []
        segment_entries = []
        context_cache_peak_bytes = 0
        sampling_preview = H3SamplingPreview(cls.hidden.unique_id, interval=preview_interval,
                                            max_resolution=preview_max_resolution, fps=preview_fps) if live_preview else None
        retry_reservation = None
        pending_context = None
        try:
            if retry is not None:
                retry_suffix, retry_reservation = _reserve_retry_name(run_folder, retry["segment"])
            for index in range(count):
                comfy.model_management.throw_exception_if_processing_interrupted()
                logging.info("H3 长视频：第 %s/%s 段，%s×%s", index + 1, count, width, height)
                if seed_mode == "increment":
                    segment_seed = min(0xffffffffffffffff, seed + index)
                elif seed_mode == "decrement":
                    segment_seed = max(0, seed - index)
                elif seed_mode == "randomize" and index > 0:
                    segment_seed = random.randint(0, 0xffffffffffffffff)
                else:
                    segment_seed = seed
                file_suffix = retry_suffix if retry is not None else f"{index + 1:04}"
                file_stem = f"segment_{file_suffix}"
                video_filename = f"{file_stem}.mp4"
                video_path = os.path.join(run_folder, video_filename)
                method = ("dynamic_mask" if dynamic_mask else "guide") if previous is not None else "none"
                context_name = None
                if previous is not None:
                    os.makedirs(data_folder, exist_ok=True)
                    context_path = os.path.join(data_folder, f"{file_stem}.h3context")
                    save_h3_latent(previous, context_path, {"kind": "retry_context"})
                    pending_context = context_path
                    context_name = annotated_path(context_path, "output")
                saved_drift = [stat.tolist() for stat in drift_stats] if drift_stats is not None else None
                previous, video, frames = cls._render_segment(
                    model, clip, vae, audio_vae, texts[min(index, len(texts) - 1)], width, height, base_frames,
                    previous, context_frames, segment_seed, sampler, sigmas, ref_images, ref_image_size,
                    video_path=video_path, metadata=metadata,
                    ref_videos=ref_videos, ref_video_audios=ref_video_audios, ref_audios=ref_audios,
                    dynamic_mask=dynamic_mask, drift_stats=drift_stats,
                    sampling_preview=sampling_preview, segment=retry["segment"] if retry else index + 1)
                if drift_correction and drift_stats is None:
                    drift_stats = drift_reference(previous, context_frames)
                videos.append(video)
                video_paths.append(annotated_path(video_path, "output"))
                frame_counts.append(frames)
                segment_seeds.append(segment_seed)
                continuation_methods.append(method)
                os.makedirs(data_folder, exist_ok=True)
                record = {"format": RECORD_FORMAT, "segment": retry["segment"] if retry else index + 1,
                          "seed": segment_seed, "width": width, "height": height, "fps": 24,
                          "frames": frames, "continuation_method": method,
                          "prompt": texts[min(index, len(texts) - 1)],
                          "settings": {"duration": duration, "aspect_ratio": aspect_ratio, "megapixels": megapixels,
                                       "seed": segment_seed, "steps": steps, "sampler_name": sampler_name,
                                       "scheduler": scheduler, "context_frames": context_frames, "denoise": denoise,
                                       "ref_image_size": ref_image_size, "dynamic_mask": dynamic_mask,
                                       "drift_correction": drift_correction},
                          "sigmas_source": sigmas_source, "sigmas": sigmas.detach().cpu().tolist(),
                          "sigmas_dtype": str(sigmas.dtype).removeprefix("torch."),
                          "context": context_name, "drift_reference": saved_drift,
                          "video_path": video_paths[-1], "seam_path": None, "context_video_path": None,
                          "diagnostics": None}
                if context_name:
                    try:
                        source_path = folder_paths.get_annotated_filepath(previous_video_name) if previous_video_name else None
                        if source_path and os.path.isfile(source_path):
                            seam_path = os.path.join(data_folder, f"seam_{file_suffix}.mp4")
                            context_video_path = os.path.join(data_folder, f"context_preview_{file_suffix}.mp4")
                            record["diagnostics"] = build_seam_preview(source_path, video_path, seam_path,
                                                                      context_path=context_video_path)
                            record["seam_path"] = annotated_path(seam_path, "output")
                            record["context_video_path"] = annotated_path(context_video_path, "output")
                        else:
                            record["diagnostics"] = {"status": "unavailable", "messages": ["前段视频不可用，本段接缝尚未检查；重抽上下文仍已保存。"]}
                    except Exception as error:
                        comfy.model_management.throw_exception_if_processing_interrupted()
                        record["diagnostics"] = {"status": "error", "messages": [f"接缝预览未完成：{error}"]}
                        logging.warning("H3 接缝预览未完成：%s", error)
                recipe_path = os.path.join(data_folder, f"{file_stem}.h3segment.json")
                recipe_name = annotated_path(recipe_path, "output")
                save_record(record, recipe_path)
                pending_context = None
                segment_entries.append(record_entry(record, recipe_name))
                if save_latents:
                    latent_filename = f"{file_stem}.h3latent"
                    save_h3_latent(previous, os.path.join(data_folder, latent_filename), record)
                    latent_results.append(ui.SavedResult(latent_filename, data_subfolder, io.FolderType.output))
                    latent_paths.append(f"{data_subfolder}/{latent_filename} [output]")
                    manifest_record = {"segment": record["segment"], "seed": segment_seed, "width": width,
                                       "height": height, "fps": 24, "frames": frames, "continuation_method": method,
                                       "latent": latent_paths[-1], "recipe": recipe_name}
                    with open(os.path.join(data_folder, "segments.jsonl"), "a", encoding="utf-8") as manifest:
                        manifest.write(json.dumps(manifest_record, ensure_ascii=False) + "\n")
                previous_video_name = video_paths[-1]
                previous["_h3_video_path"] = previous_video_name
                if index < count - 1:
                    previous = _copy_context_tail(previous, context_frames)
                    context_cache_peak_bytes = max(context_cache_peak_bytes,
                        sum(stream.numel() * stream.element_size() for stream in previous["samples"].tensors))
            comfy.model_management.throw_exception_if_processing_interrupted()
            combined = InputImpl.VideoFromList(videos, codec=Types.VideoCodec("h264"))
            info = json.dumps({"width": width, "height": height, "fps": 24, "segments": count,
                               "seed_mode": seed_mode, "seeds": segment_seeds,
                               "retry_seed_mode": retry_seed_mode if retry is not None else None,
                               "continued_from_latent": previous_latent is not None,
                               "save_latents": save_latents,
                               "dynamic_mask": dynamic_mask, "continuation_methods": continuation_methods,
                               "drift_correction": drift_correction,
                               "live_preview": live_preview, "preview_interval": preview_interval,
                               "preview_frames": frame_counts if live_preview else [], "preview_fps": preview_fps,
                               "preview_max_resolution": preview_max_resolution,
                               "video_storage": "output", "output_subfolder": subfolder,
                               "video_paths": video_paths, "latent_paths": latent_paths,
                               "manifest": f"{data_subfolder}/segments.jsonl" if save_latents else None,
                               "frames_per_segment": frame_counts, "total_frames": sum(frame_counts),
                               "segment_records": [item["recipe_path"] for item in segment_entries],
                               "duration_seconds": sum(frame_counts) / 24,
                               "context_cache_device": "cpu", "context_cache_peak_bytes": context_cache_peak_bytes},
                              ensure_ascii=False, indent=2)
            preview = {"latents": latent_results[-1:]} if latent_results else {}
            preview["h3_sgun_segments"] = segment_entries
            preview["h3_sgun_info"] = [info]
            return io.NodeOutput(combined, previous, info, latent_paths, ui=preview)
        finally:
            # A context saved for a segment that never got its record is unreachable.
            if pending_context is not None:
                try:
                    os.remove(pending_context)
                except FileNotFoundError:
                    pass
            if retry_reservation is not None:
                try:
                    os.remove(retry_reservation)
                except FileNotFoundError:
                    pass
            if sampling_preview is not None:
                sampling_preview.finish()
