import json
import logging
import os
import random
import re

import comfy.model_management
import comfy.samplers
import folder_paths
import nodes
from comfy.cli_args import args
from comfy_api.latest import InputImpl, Types, io, ui
from comfy_execution.graph_utils import ExecutionBlocker
from comfy_extras.nodes_audio import VAEDecodeAudio
from comfy_extras.nodes_custom_sampler import BasicGuider, BasicScheduler, Noise_RandomNoise, SamplerCustomAdvanced
from comfy_extras.nodes_minimax_h3 import MiniMaxH3ReferenceToVideo, align_frame_count
from comfy_extras.nodes_resolution import AspectRatio, ResolutionSelector
from comfy_extras.nodes_video import CreateVideo

from . import MiniMaxH3LatentContinuationGuide, _av_streams, _copy_context_tail
from .latent_io import save_h3_latent
from .dynamic_mask import check_dynamic_mask_support, prepare_dynamic_mask


H3Prompts = io.Custom("H3_PROMPTS")


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
            description="逐段生成并即时保存视频，可选保存完整音视频LATENT。片段从磁盘读取，不在内存累积；开启合并后连接外部Save Video保存整片。",
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
                             control_after_generate=True),
                io.Int.Input("steps", display_name="采样步数", default=8, min=1, max=10000,
                             tooltip="未连接自定义 sigmas 时使用；连接后，实际步数由 sigmas 序列决定。"),
                io.Combo.Input("sampler_name", display_name="采样器", options=comfy.samplers.SAMPLER_NAMES,
                               default="res_multistep"),
                io.Combo.Input("scheduler", display_name="调度器", options=comfy.samplers.SCHEDULER_NAMES,
                               default="simple", tooltip="连接自定义 sigmas 时跳过此调度器。"),
                io.Int.Input("context_frames", display_name="上下文帧数", default=22, min=5, max=3600, step=17,
                             advanced=True, tooltip="直接读取上一段音视频 latent 的尾部。续段导出时自动删去重叠部分。"),
                io.Combo.Input("seed_mode", display_name="各段种子", options=io.ControlAfterGenerate,
                               default="increment", advanced=True,
                               tooltip="首段使用上方种子，后续段按fixed固定、increment递增、decrement递减或randomize随机。生成信息记录每段实际种子；生成后控制影响下一次任务的起始种子。"),
                io.Float.Input("denoise", display_name="降噪", default=1.0, min=0.01, max=1.0, step=0.01, advanced=True,
                               tooltip="只用于内部调度器；连接自定义 sigmas 时直接使用输入序列。"),
                io.Combo.Input("ref_image_size", display_name="参考图尺寸", options=["match", "max"],
                               default="match", advanced=True),
                io.Boolean.Input("merge_video", display_name="合并视频", default=False,
                                 tooltip="关闭时只即时保存各片段，并自动跳过合并视频输出下游的Save Video；开启后提供合并VIDEO。两种模式均从磁盘读取片段。"),
                io.Boolean.Input("save_latents", display_name="保存每段 LATENT", default=False,
                                 tooltip="每段完成后保存完整音视频.h3latent，可用本插件的H3读取音视频LATENT节点恢复续接。增加磁盘占用，不在内存累积各段LATENT。"),
                io.String.Input("filename_prefix", display_name="分段保存前缀", default="video/H3-SGUN/H3",
                                tooltip="在ComfyUI output内为每次运行建立独立目录，视频和可选LATENT按段号配对保存。"),
                io.Boolean.Input("dynamic_mask", display_name="动态掩码", default=False,
                                 tooltip="仅续接时生效。按实际sigma动态重绘上下文，保护接缝及音频前缀；用于对照漂移控制，不保证恢复丢失细节。需原生H3音视频掩码支持，不能叠加其他动态掩码。"),
                io.Autogrow.Input("ref_images", display_name="参考图", optional=True,
                    template=io.Autogrow.TemplatePrefix(io.Image.Input("ref_image"), prefix="ref_image_", min=0, max=100)),
                io.Sigmas.Input("sigmas", optional=True,
                                tooltip="连接 Manual Sigmas 等节点后，全部片段直接使用此序列，覆盖内部调度器、步数和降噪设置。"),
                io.Autogrow.Input("ref_videos", display_name="参考视频", optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        io.Image.Input("ref_video", tooltip="与官方 H3 一致，输入24fps的IMAGE帧批次，至少5帧。"),
                        prefix="ref_video_", min=0, max=100)),
                io.Autogrow.Input("ref_video_audios", display_name="参考视频音轨", optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        io.Audio.Input("ref_video_audio", tooltip="按相同编号配对：ref_video_audio_0 对应 ref_video_0。"),
                        prefix="ref_video_audio_", min=0, max=100)),
                io.Autogrow.Input("ref_audios", display_name="独立参考音频", optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        io.Audio.Input("ref_audio", tooltip="独立参考音频，沿用官方 H3 的 <Audio j> 编号。"),
                        prefix="ref_audio_", min=0, max=100)),
                io.Latent.Input("previous_latent", display_name="上一段 LATENT", optional=True,
                                tooltip="连接H3读取音视频LATENT或上一生成节点的最后一段LATENT。必须包含完整音视频且分辨率相同；生成段数设1可单独重抽下一段。"),
            ],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
            is_output_node=True,
            outputs=[io.Video.Output(display_name="合并视频"), io.Latent.Output(display_name="最后一段 latent"),
                     io.String.Output(display_name="生成信息"),
                     io.Video.Output(display_name="分段视频", is_output_list=True,
                                     tooltip="已即时保存的VIDEO文件列表；无需再接Save Video保存，可用于后续编辑。"),
                     io.String.Output(display_name="LATENT 文件路径", is_output_list=True,
                                     tooltip="开启保存每段LATENT后，输出按段号排序的.h3latent路径；关闭时为空列表。")],
        )

    @classmethod
    def _render_segment(cls, model, clip, vae, audio_vae, prompt, width, height, base_frames, previous,
                        context_frames, seed, sampler, sigmas, ref_images, ref_image_size,
                        video_path, metadata, ref_videos=None, ref_video_audios=None, ref_audios=None,
                        dynamic_mask=False):
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
            sampled = SamplerCustomAdvanced.execute(Noise_RandomNoise(seed), guider, sampler, sigmas, latent)[0]
        finally:
            if dynamic_mask and previous is not None:
                model.model_options["denoise_mask_function"].__self__.current_mask = None
        del guider, positive, latent, previous, model
        comfy.model_management.throw_exception_if_processing_interrupted()
        images = nodes.VAEDecode().decode(vae, sampled)[0]
        audio = VAEDecodeAudio.execute(audio_vae, sampled)[0]
        images = images[overlap:]
        sample_rate = audio["sample_rate"]
        start = round(overlap / 24 * sample_rate)
        end = start + round(images.shape[0] / 24 * sample_rate)
        audio = {"waveform": audio["waveform"][..., start:end], "sample_rate": sample_rate}
        video = CreateVideo.execute(images, 24, audio, bit_depth=8, color_space="sRGB")[0]
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
                previous_latent=None, merge_video=False, save_latents=False, filename_prefix="video/H3-SGUN/H3",
                dynamic_mask=False):
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
        if sigmas is None:
            sigmas = BasicScheduler.execute(model, scheduler, steps, denoise)[0]
        folder, basename, counter, _, _ = folder_paths.get_save_image_path(
            filename_prefix, folder_paths.get_output_directory(), width, height)
        while True:
            run_folder = os.path.join(folder, f"{basename}_{counter:05}")
            try:
                os.mkdir(run_folder)
                break
            except FileExistsError:
                counter += 1
        subfolder = os.path.relpath(run_folder, folder_paths.get_output_directory()).replace(os.sep, "/")
        metadata = None
        if not args.disable_metadata:
            metadata = dict(cls.hidden.extra_pnginfo or {})
            if cls.hidden.prompt is not None:
                metadata["prompt"] = cls.hidden.prompt
        previous = _copy_context_tail(previous_latent, context_frames) if previous_latent is not None else None
        videos, frame_counts, segment_seeds, continuation_methods = [], [], [], []
        video_results, latent_results, latent_paths = [], [], []
        context_cache_peak_bytes = 0
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
            video_filename = f"segment_{index + 1:04}.mp4"
            video_path = os.path.join(run_folder, video_filename)
            method = ("dynamic_mask" if dynamic_mask else "guide") if previous is not None else "none"
            previous, video, frames = cls._render_segment(
                model, clip, vae, audio_vae, texts[min(index, len(texts) - 1)], width, height, base_frames,
                previous, context_frames, segment_seed, sampler, sigmas, ref_images, ref_image_size,
                video_path=video_path, metadata=metadata,
                ref_videos=ref_videos, ref_video_audios=ref_video_audios, ref_audios=ref_audios,
                dynamic_mask=dynamic_mask)
            videos.append(video)
            video_results.append(ui.SavedResult(video_filename, subfolder, io.FolderType.output))
            frame_counts.append(frames)
            segment_seeds.append(segment_seed)
            continuation_methods.append(method)
            record = {"segment": index + 1, "seed": segment_seed, "width": width, "height": height,
                      "fps": 24, "frames": frames, "video": f"{subfolder}/{video_filename}",
                      "continuation_method": method}
            if save_latents:
                latent_filename = f"segment_{index + 1:04}.h3latent"
                save_h3_latent(previous, os.path.join(run_folder, latent_filename), record)
                latent_results.append(ui.SavedResult(latent_filename, subfolder, io.FolderType.output))
                latent_paths.append(f"{subfolder}/{latent_filename} [output]")
                record["latent"] = latent_paths[-1]
            with open(os.path.join(run_folder, "segments.jsonl"), "a", encoding="utf-8") as manifest:
                manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
            if index < count - 1:
                previous = _copy_context_tail(previous, context_frames)
                context_cache_peak_bytes = max(context_cache_peak_bytes,
                    sum(stream.numel() * stream.element_size() for stream in previous["samples"].tensors))
        comfy.model_management.throw_exception_if_processing_interrupted()
        combined = ExecutionBlocker(None)
        if merge_video:
            logging.info("H3 长视频：合并 %s 个已保存片段", count)
            combined = InputImpl.VideoFromList(videos, codec=Types.VideoCodec("h264"))
        info = json.dumps({"width": width, "height": height, "fps": 24, "segments": count,
                           "seed_mode": seed_mode, "seeds": segment_seeds,
                           "continued_from_latent": previous_latent is not None,
                           "merge_video": merge_video, "save_latents": save_latents,
                           "dynamic_mask": dynamic_mask, "continuation_methods": continuation_methods,
                           "video_storage": "disk", "output_subfolder": subfolder,
                           "video_paths": [item["subfolder"] + "/" + item["filename"] for item in video_results],
                           "latent_paths": latent_paths, "manifest": f"{subfolder}/segments.jsonl",
                           "frames_per_segment": frame_counts, "total_frames": sum(frame_counts),
                           "duration_seconds": sum(frame_counts) / 24,
                           "context_cache_device": "cpu", "context_cache_peak_bytes": context_cache_peak_bytes},
                          ensure_ascii=False, indent=2)
        preview = ui.PreviewVideo(video_results[-1:]).as_dict()
        if latent_results:
            preview["latents"] = latent_results[-1:]
        return io.NodeOutput(combined, previous, info, videos, latent_paths, ui=preview)
