"""Bounded-window H3 performance generation with one continuous source soundtrack."""

import json
import logging
import math
import os
import shutil
import tempfile

import torch

import comfy.audio
import comfy.model_management
import comfy.samplers
import comfy.utils
import folder_paths
import nodes
from comfy.nested_tensor import NestedTensor
from comfy_api.latest import InputImpl, Types, io
from comfy_extras.nodes_audio import VAEEncodeAudio
from comfy_extras.nodes_custom_sampler import BasicGuider, BasicScheduler, Noise_RandomNoise, SamplerCustomAdvanced
from comfy_extras.nodes_minimax_h3 import EmptyMiniMaxH3LatentAV, MiniMaxH3ReferenceToVideo
from comfy_extras.nodes_resolution import AspectRatio, ResolutionSelector
from comfy_extras.nodes_video import CreateVideo
from .long_video import H3Prompts
from .digital_human_video import inspect_video, normalize_video


CONTEXT_FRAMES = 39
CONTEXT_TOKENS = 12
WINDOW_FRAMES = (141, 192, 243, 294, 345)


def _segment_plan(sample_count, sample_rate, segment_seconds, total_frames=None):
    if sample_count < 1 or sample_rate < 1:
        raise ValueError("数字人需要非空音频及有效采样率。")
    maximum = min(WINDOW_FRAMES, key=lambda frames: abs(frames / 24 - segment_seconds))
    source_frames = (sample_count * 24 + sample_rate - 1) // sample_rate
    total_frames = source_frames if total_frames is None else min(source_frames, total_frames)
    result, start = [], 0
    while start < total_frames:
        context = CONTEXT_FRAMES if start else 0
        keep = min(total_frames - start, maximum - context)
        frames = next(f for f in WINDOW_FRAMES if f >= context + keep)
        result.append({"start_frame": start, "source_start_frame": start - context,
                       "context_frames": context, "sample_frames": frames, "keep_frames": keep})
        start += keep
    return result


def _audio_window(audio, start_frame, frames):
    rate = audio["sample_rate"]
    start = round(start_frame * rate / 24)
    end = round((start_frame + frames) * rate / 24)
    wave = audio["waveform"]
    result = wave.new_zeros((*wave.shape[:-1], end - start))
    available = max(0, min(end, wave.shape[-1]) - start)
    if available:
        result[..., :available] = wave[..., start:start + available]
    return {"waveform": result, "sample_rate": rate}


def _lock_audio(latent, encoded_audio, previous_video=None):
    video, template = latent["samples"].unbind()
    audio = encoded_audio["samples"]
    if audio.ndim != 4 or tuple(audio.shape[:-1]) != tuple(template.shape[:-1]):
        raise ValueError("请使用 H3 音频 VAE；编码后需要 [1,32,2,T] 音频潜变量。")
    audio = audio.to(device=template.device, dtype=template.dtype)
    if audio.shape[-1] > template.shape[-1]:
        audio = audio[..., :template.shape[-1]].clone()
    elif audio.shape[-1] < template.shape[-1]:
        fitted = torch.zeros_like(template)
        fitted[..., :audio.shape[-1]] = audio
        audio = fitted
    video = video.clone()
    video_mask = torch.ones_like(video)
    if previous_video is not None:
        if previous_video.shape[2] != CONTEXT_TOKENS or previous_video.shape[3:] != video.shape[3:]:
            raise ValueError("数字人续接上下文与本段视频尺寸不匹配。")
        video[:, :, :CONTEXT_TOKENS] = previous_video.to(device=video.device, dtype=video.dtype)
        video_mask[:, :, :CONTEXT_TOKENS] = 0
    result = latent.copy()
    result["samples"] = NestedTensor((video, audio))
    result["noise_mask"] = NestedTensor((video_mask, torch.zeros_like(audio)))
    return result


def _video_tail(latent):
    video = latent["samples"].unbind()[0]
    return video[:, :, -CONTEXT_TOKENS:].detach().to(device="cpu", copy=True).contiguous()


def _select_audio(audio, start_time, duration):
    waveform, rate = audio["waveform"], audio["sample_rate"]
    if waveform.ndim != 3 or waveform.shape[0] != 1 or waveform.shape[1] not in (1, 2) or rate < 1:
        raise ValueError("数字人需要单条单声道或立体声音频 [1,C,N]。")
    start = max(0, round(start_time * rate))
    end = min(waveform.shape[-1], start + round(duration * rate)) if duration > 0 else waveform.shape[-1]
    if start >= end:
        raise ValueError("音频选段为空，请检查开始时间和生成时长。")
    return {"waveform": waveform[..., start:end].detach().to(device="cpu", copy=True).contiguous(),
            "sample_rate": rate}


def _prompt_texts(prompt, prompts, count):
    if prompts is None:
        return [prompt] * count
    if not isinstance(prompts, dict) or not isinstance(prompts.get("prompts"), list) or not prompts["prompts"]:
        raise ValueError("分段提示词无效，请连接 H3 长视频提示词节点。")
    texts = prompts["prompts"]
    if any(not isinstance(text, str) for text in texts):
        raise ValueError("每段提示词必须是文本。")
    if len(texts) < count and not prompts.get("repeat_last", False):
        raise ValueError(f"本次音频需要 {count} 段提示词，目前只有 {len(texts)} 段；请补齐或开启复用最后一段。")
    return [texts[min(index, len(texts) - 1)] for index in range(count)]


class H3DigitalHuman(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3DigitalHuman",
            display_name="H3 数字人：原音频驱动长视频 by神棍",
            category="video/minimax",
            search_aliases=["H3 digital human", "H3 数字人", "H3 唱歌", "H3 lip sync"],
            description="按原音频长度和分段提示词自动生成。完整视频与分段视频按下游连线使用，完整视频在外接保存时才合并；保存与预览均在外部，内部仅使用临时缓存。",
            inputs=[
                io.Model.Input("model"),
                io.Clip.Input("clip"),
                io.Vae.Input("vae", display_name="视频 VAE"),
                io.Vae.Input("audio_vae", display_name="音频 VAE"),
                io.Autogrow.Input("ref_images", display_name="参考图", optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input("ref_image", tooltip="按接入顺序引用 <Picture 1>、<Picture 2>…；各段共用参考图。"),
                        prefix="ref_image_", min=0, max=9)),
                io.Audio.Input("audio", display_name="原音频", tooltip="口播或歌曲。最终原音连续合成一次，模型只生成画面。"),
                io.String.Input("prompt", display_name="单一提示词（兼容旧工作流）", multiline=True, default="", advanced=True,
                                tooltip="描述人物、场景与持续表演，整片复用；不需要添加未接入的 <Audio 1> 标签。"),
                io.Float.Input("segment_seconds", display_name="每段时长", default=8.0, min=5.0, max=15.0, step=0.1,
                               tooltip="含续接上下文。自动取最近的 5.875、8、10.125、12.25 或 14.375 秒；降低可减少单段显存。"),
                io.Combo.Input("aspect_ratio", display_name="宽高比", options=AspectRatio,
                               default=AspectRatio.WIDESCREEN_V),
                io.Float.Input("megapixels", display_name="百万像素", default=0.4, min=0.1, max=16.0, step=0.1),
                io.Int.Input("seed", display_name="种子", default=0, min=0, max=0xffffffffffffffff,
                             control_after_generate=True),
                io.Int.Input("steps", display_name="采样步数", default=8, min=1, max=10000),
                io.Combo.Input("sampler_name", display_name="采样器", options=comfy.samplers.SAMPLER_NAMES,
                               default="er_sde", advanced=True),
                io.Combo.Input("scheduler", display_name="调度器", options=comfy.samplers.SCHEDULER_NAMES,
                               default="beta", advanced=True),
                io.Sigmas.Input("sigmas", optional=True, tooltip="可选，直接沿用已有 SIGMAS，覆盖内部步数和调度器。"),
                io.Combo.Input("ref_image_size", display_name="参考图尺寸", options=["match", "max"],
                               default="match", optional=True, advanced=True),
                io.Boolean.Input("tiled_decode", display_name="分块解码", default=True, optional=True, advanced=True,
                                 tooltip="默认使用官方 VAE 分块解码降低峰值显存。"),
                io.Float.Input("start_time", display_name="音频开始秒数", default=0.0, min=0.0, step=0.1,
                               optional=True, advanced=True),
                io.Float.Input("duration", display_name="生成时长（0为剩余全部）", default=0.0, min=0.0, step=0.1,
                               optional=True, advanced=True),
                H3Prompts.Input("prompts", display_name="分段提示词", optional=True,
                                tooltip="连接 H3 长视频提示词。连接后覆盖旧单一提示词，按实际循环段依次使用；不足时提前报错。"),
            ],
            outputs=[io.Video.Output(display_name="完整视频",
                                     tooltip="外接保存视频节点时才实际合并并保存完整原音长视频，无需开关。"),
                     io.String.Output(display_name="生成信息"),
                     io.Video.Output(display_name="分段视频", is_output_list=True,
                                     tooltip="按顺序输出带对应原音的VIDEO列表，接另一个保存视频节点可逐段保存，供后续剪辑。")],
        )

    @classmethod
    def _render_segment(cls, model, vae, audio_vae, positive, width, height, source_audio, plan,
                        previous_video, seed, sampler, sigmas, video_path, tiled_decode, output_audio=None):
        latent = EmptyMiniMaxH3LatentAV.execute(width, height, plan["sample_frames"])[0]
        window = _audio_window(source_audio, plan["source_start_frame"], plan["sample_frames"])
        encoded = VAEEncodeAudio.execute(audio_vae, window)[0]
        latent = _lock_audio(latent, encoded, previous_video)
        del window, encoded, previous_video
        guider = BasicGuider.execute(model, positive)[0]
        sampled = SamplerCustomAdvanced.execute(Noise_RandomNoise(seed), guider, sampler, sigmas, latent)[0]
        del guider, latent
        comfy.model_management.throw_exception_if_processing_interrupted()
        tail = _video_tail(sampled)
        if tiled_decode:
            images = nodes.VAEDecodeTiled().decode(vae, sampled, tile_size=512, overlap=64,
                                                  temporal_size=64, temporal_overlap=8)[0]
        else:
            images = nodes.VAEDecode().decode(vae, sampled)[0]
        del sampled
        prefix = plan["context_frames"]
        images = images[prefix:prefix + plan["keep_frames"]]
        if images.shape[0] != plan["keep_frames"]:
            raise ValueError("视频 VAE 解码帧数不足，不能完成数字人音画对齐。")
        video = CreateVideo.execute(images, 24, audio=output_audio, bit_depth=8, color_space="sRGB")[0]
        temporary = video_path + ".tmp"
        try:
            video.save_to(temporary, format=Types.VideoContainer("mp4"), codec=Types.VideoCodec("h264"))
            os.replace(temporary, video_path)
        finally:
            if os.path.exists(temporary):
                os.remove(temporary)
        return tail

    @classmethod
    def execute(cls, model, clip, vae, audio_vae, audio, prompt="", segment_seconds=8.0,
                aspect_ratio=AspectRatio.WIDESCREEN_V, megapixels=0.4, seed=0, steps=8,
                sampler_name="er_sde", scheduler="beta", sigmas=None, ref_image_size="match",
                tiled_decode=True, start_time=0.0, duration=0.0, ref_images=None, prompts=None):
        if not callable(getattr(model.model, "_denoise_mask_values", None)):
            raise ValueError("数字人需要近期 ComfyUI 原生 H3 音视频遮罩支持，请更新 ComfyUI。")
        if model.model_options.get("denoise_mask_function") is not None:
            raise ValueError("MODEL 含动态去噪遮罩，会改变原音频冻结；请移除 Differential Diffusion 等遮罩补丁。")
        source = _select_audio(audio, start_time, duration)
        target_frames = None
        if duration > 0:
            requested = duration * 24
            target_frames = max(1, round(requested) if abs(requested - round(requested)) < 1e-7 else math.ceil(requested))
        plan = _segment_plan(source["waveform"].shape[-1], source["sample_rate"], segment_seconds, target_frames)
        texts = _prompt_texts(prompt, prompts, len(plan))
        width, height = ResolutionSelector.execute(aspect_ratio, megapixels, 32).result
        if sigmas is None:
            sigmas = BasicScheduler.execute(model, scheduler, steps, 1.0)[0]
        if sigmas.ndim != 1 or sigmas.numel() < 2 or not torch.isfinite(sigmas).all() or float(sigmas[-1]) != 0.0:
            raise ValueError("数字人每段需要完整采样，SIGMAS 至少含起点及结尾 0。")
        sampler = comfy.samplers.sampler_object(sampler_name)
        audio_rate = int(getattr(audio_vae, "audio_sample_rate", 32000))
        driving = {"waveform": comfy.audio.resample(source["waveform"], source["sample_rate"], audio_rate),
                   "sample_rate": audio_rate}
        cache_root = folder_paths.get_temp_directory()
        os.makedirs(cache_root, exist_ok=True)
        cache_folder = tempfile.mkdtemp(prefix="h3_digital_human_", dir=cache_root)
        total_frames = sum(part["keep_frames"] for part in plan)
        previous_video, positive, last_text, pieces = None, None, None, []
        progress = comfy.utils.ProgressBar(len(plan))
        try:
            for index, part in enumerate(plan):
                comfy.model_management.throw_exception_if_processing_interrupted()
                if positive is None or texts[index] != last_text:
                    positive = None
                    positive, unused = MiniMaxH3ReferenceToVideo.execute(
                        clip=clip, vae=vae, prompt=texts[index], width=width, height=height,
                        length=part["sample_frames"], ref_image_size=ref_image_size,
                        ref_images=ref_images).result
                    del unused
                    last_text = texts[index]
                logging.info("H3 数字人：第 %d/%d 段，输出 %.3f–%.3f 秒",
                             index + 1, len(plan), part["start_frame"] / 24,
                             (part["start_frame"] + part["keep_frames"]) / 24)
                path = os.path.join(cache_folder, f"segment_{index + 1:05}.mp4")
                segment_audio = _audio_window(source, part["start_frame"], part["keep_frames"])
                previous_video = cls._render_segment(
                    model, vae, audio_vae, positive, width, height, driving, part, previous_video,
                    (seed + index) % (1 << 64), sampler, sigmas, path, tiled_decode, output_audio=segment_audio)
                try:
                    inspect_video(path, part["keep_frames"], width, height,
                                  segment_audio["waveform"].shape[-1], source["sample_rate"], full_decode=True)
                except ValueError:
                    normalize_video(path, path, part["keep_frames"], width, height, audio=segment_audio)
                    inspect_video(path, part["keep_frames"], width, height,
                                  segment_audio["waveform"].shape[-1], source["sample_rate"], full_decode=True)
                del segment_audio
                pieces.append(InputImpl.VideoFromFile(path))
                progress.update(1)
            previous_video = positive = driving = None
            comfy.model_management.throw_exception_if_processing_interrupted()
            merged = InputImpl.VideoFromList(pieces, complete_audio=source, codec=Types.VideoCodec("h264"))
            info = {"width": width, "height": height, "fps": 24, "segments": len(plan),
                    "total_frames": total_frames, "audio_seconds": source["waveform"].shape[-1] / source["sample_rate"],
                    "sample_frames": [part["sample_frames"] for part in plan], "context_frames": CONTEXT_FRAMES,
                    "tiled_decode": tiled_decode, "audio_mode": "locked_source_single_mux", "latent_saving": False,
                    "video_storage": "temp",
                    "timeline": [{"segment": index + 1, "start_seconds": part["start_frame"] / 24,
                                  "end_seconds": (part["start_frame"] + part["keep_frames"]) / 24}
                                 for index, part in enumerate(plan)]}
            return io.NodeOutput(merged, json.dumps(info, ensure_ascii=False), pieces)
        except BaseException:
            if folder_paths.is_within_directory(cache_root, cache_folder):
                shutil.rmtree(cache_folder)
            raise
