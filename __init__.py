"""Native MiniMax H3 latent continuation and long-video generation."""

import logging

from comfy.nested_tensor import NestedTensor
from comfy.ldm.minimax.model import FRAME_RESCALE
from comfy_api.latest import ComfyExtension, io
from .prompt_schedule import H3LoopPromptSchedule


def _av_streams(latent, name):
    samples = latent.get("samples")
    if not isinstance(samples, NestedTensor) or len(samples.tensors) != 2:
        raise ValueError(f"{name} 必须是 MiniMax H3 的联合音视频 LATENT，请连接 H3 采样器输出。")
    video, audio = samples.tensors
    if (video.ndim != 5 or video.shape[0] != 1 or video.shape[1] != 24
            or audio.ndim != 4 or tuple(audio.shape[:3]) != (1, 32, 2)):
        raise ValueError(f"{name} 需要视频 [1,24,T,H,W]、音频 [1,32,2,T]，当前形状为 {tuple(video.shape)}、{tuple(audio.shape)}。")
    if video.shape[2] < 2 or (video.shape[2] - 2) % 5:
        raise ValueError(f"{name} 的视频时间长度不符合 H3 的 5k+2 latent 网格。")
    frames = (video.shape[2] - 2) // 5 * 17 + 5
    if audio.shape[-1] != round(frames * FRAME_RESCALE):
        raise ValueError(f"{name} 的音视频长度不匹配：{frames} 帧视频需要 {round(frames * FRAME_RESCALE)} 个音频 token，当前为 {audio.shape[-1]}。")
    return video, audio, frames


def _copy_context_tail(latent, context_frames):
    video, audio, previous_frames = _av_streams(latent, "上一段 latent")
    # 39 is the shortest complete AV grid holding at least one second of audio.
    frames = min(previous_frames, max(39, (context_frames - 5) // 17 * 17 + 5))
    video_t = (frames - 5) // 17 * 5 + 2
    audio_t = round(frames * FRAME_RESCALE)
    return {
        "samples": NestedTensor((video[:, :, -video_t:].detach().to(device="cpu", copy=True).contiguous(),
                                  audio[..., -audio_t:].detach().to(device="cpu", copy=True).contiguous())),
        "_h3_audio_overhang": latent.get("_h3_audio_overhang", audio.shape[-1] - previous_frames * FRAME_RESCALE),
    }


class MiniMaxH3LatentContinuationGuide(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3LatentContinuationGuide",
            display_name="MiniMax H3 潜空间续接引导",
            category="model/conditioning/minimax",
            search_aliases=["h3 latent guide", "h3 continuation", "H3 续接", "潜空间续接"],
            description="直接用上一段生成的音视频 latent 尾部引导下一段，无需 VAE 解码和重编码。采样仍用原有节点；拼接时删除新段开头的 trim_frames 帧及对应音频。",
            inputs=[
                io.Conditioning.Input("positive", tooltip="下一段的正向条件，输出接 BasicGuider / CFGGuider 或 KSampler。"),
                io.Latent.Input("latent", display_name="本段 latent", tooltip="下一段要采样的空 H3 音视频 latent，同时直接连接下一段采样器。"),
                io.Latent.Input("previous_latent", display_name="上一段 latent", tooltip="上一段采样器的完整音视频 latent 输出，不是空 latent。两段分辨率必须相同。"),
                io.Int.Input("context_frames", display_name="上下文帧数", default=22, min=5, max=3600, step=17,
                             tooltip="读取上一段最后 5、22、39、56…帧。默认 22；超过可用长度时自动缩短。新段总长度包含这部分重叠。"),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Int.Output(display_name="trim_frames", tooltip="拼接时从新段开头删除的帧数；音频同样删除 trim_frames / 24 秒。"),
            ],
        )

    @classmethod
    def execute(cls, positive, latent, previous_latent, context_frames=22) -> io.NodeOutput:
        target_video, _, target_frames = _av_streams(latent, "本段 latent")
        video, audio, previous_frames = _av_streams(previous_latent, "上一段 latent")
        if video.shape[3:] != target_video.shape[3:]:
            raise ValueError(f"两段分辨率必须相同：上一段 {video.shape[4] * 16}×{video.shape[3] * 16}，本段 {target_video.shape[4] * 16}×{target_video.shape[3] * 16}。")
        if context_frames < 5:
            raise ValueError("上下文至少需要 5 帧，才能直接截取完整的 H3 latent 尾部。")

        frames = min(context_frames, previous_frames, target_frames)
        frames = (frames - 5) // 17 * 17 + 5
        video_t = (frames - 5) // 17 * 5 + 2
        video_guide = {"resolved_frame_index": 0, "latent": video[:, :, -video_t:].clone()}

        # End-align at least one second of audio with the video join on the target's 40 Hz grid.
        audio_t = min(audio.shape[-1], round(max(24, frames) * FRAME_RESCALE))
        overhang = previous_latent.get("_h3_audio_overhang", audio.shape[-1] - previous_frames * FRAME_RESCALE)
        audio_end = round(frames * FRAME_RESCALE + overhang)
        audio_guide = {
            "resolved_frame_index": (audio_end - audio_t) / FRAME_RESCALE,
            "audio_latent": audio[..., -audio_t:].clone(),
        }

        out = []
        removed = set()
        for embedding, metadata in positive:
            extra = metadata.copy()
            kept = []
            for keyframe in metadata.get("minimax_keyframes", ()):
                position = keyframe["resolved_frame_index"]
                if position < frames:
                    removed.add(position)
                else:
                    kept.append(keyframe)
            extra["minimax_keyframes"] = [video_guide, audio_guide] + kept
            out.append([embedding, extra])
        if removed:
            logging.warning("H3 潜空间续接：已替换重叠区内的旧引导（帧 %s）；尾帧和后续引导保留。", sorted(removed))
        return io.NodeOutput(out, frames)


from .long_video import H3LongVideo, H3VideoPromptPlan
from .latent_io import H3LoadLatent


class H3LatentGuideExtension(ComfyExtension):
    async def get_node_list(self):
        return [MiniMaxH3LatentContinuationGuide, H3LoopPromptSchedule, H3LongVideo, H3VideoPromptPlan, H3LoadLatent]


async def comfy_entrypoint():
    return H3LatentGuideExtension()
