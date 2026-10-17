"""Native integrity checks and bounded timeline normalization for H3 segments."""
import os
from fractions import Fraction
from pathlib import Path

import av
import comfy.model_management
from comfy_api.latest import InputImpl, Types


def inspect_video(path, frames, width, height, audio_samples=None, audio_rate=None, full_decode=False):
    try:
        with av.open(str(path)) as container:
            if len(container.streams.video) != 1:
                raise ValueError("视频轨数量不正确。")
            stream = container.streams.video[0]
            if (stream.width, stream.height) != (width, height) or stream.average_rate != 24:
                raise ValueError("视频尺寸或帧率与任务不一致。")
            count, previous, first = 0, None, None
            for frame in container.decode(stream):
                if count % 64 == 0:
                    comfy.model_management.throw_exception_if_processing_interrupted()
                if frame.pts is None:
                    raise ValueError("视频缺少显示时间戳。")
                current = frame.pts * frame.time_base
                if first is None:
                    first = current
                if previous is not None and current - previous != Fraction(1, 24):
                    raise ValueError("视频时间戳有缺口、重复或倒退。")
                previous, count = current, count + 1
                if not full_decode:
                    break
            if first != 0:
                raise ValueError("视频显示起点不是零。")
            actual_frames = count if full_decode else stream.frames
            if actual_frames != frames:
                raise ValueError(f"视频帧数不一致：需要 {frames}，实际 {actual_frames}。")
        audio_count = None
        if audio_samples is not None:
            with av.open(str(path)) as container:
                if len(container.streams.audio) != 1:
                    raise ValueError("完整原音轨缺失或数量不正确。")
                stream = container.streams.audio[0]
                if stream.sample_rate != audio_rate:
                    raise ValueError("音频采样率与原音不一致。")
                audio_count, previous = 0, None
                for frame in container.decode(stream):
                    if audio_count % (1024 * 64) == 0:
                        comfy.model_management.throw_exception_if_processing_interrupted()
                    if frame.pts is None or (previous is not None and frame.pts * frame.time_base < previous):
                        raise ValueError("音频时间戳缺失或倒退。")
                    previous = frame.pts * frame.time_base
                    audio_count += frame.samples
                if not audio_samples <= audio_count < audio_samples + 2048:
                    raise ValueError(f"音轨长度不一致：需要 {audio_samples}，解码 {audio_count}。")
        return {"frames": actual_frames, "width": width, "height": height, "fps": 24,
                "first_pts": 0, "decoded_audio_samples": audio_count, "full_decode": full_decode}
    except (av.error.FFmpegError, OSError) as error:
        raise ValueError(f"视频完整性检查失败：{path}") from error


def normalize_video(path, target_path, frames, width, height, audio=None):
    """Bounded-frame fallback. Preserve the source unless a complete replacement passes."""
    source, destination = Path(path), Path(target_path)
    silent = destination.with_name(destination.name + ".normalize.tmp")
    result = destination.with_name(destination.name + ".repair.tmp")
    try:
        with av.open(str(source)) as incoming, av.open(str(silent), mode="w", format="mp4") as outgoing:
            stream = outgoing.add_stream("libx264", rate=24)
            stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
            stream.options = {"crf": "18", "preset": "fast"}
            count = 0
            for frame in incoming.decode(video=0):
                if count % 64 == 0:
                    comfy.model_management.throw_exception_if_processing_interrupted()
                if count == frames:
                    break
                frame = frame.reformat(width=width, height=height, format="yuv420p")
                frame.pts, frame.time_base = count, Fraction(1, 24)
                outgoing.mux(stream.encode(frame))
                count += 1
            if count != frames:
                raise ValueError("视频帧不足，不能用重复画面补齐。")
            outgoing.mux(stream.encode())
        if audio is not None:
            InputImpl.VideoFromList([InputImpl.VideoFromFile(str(silent))], complete_audio=audio,
                                   codec=Types.VideoCodec("h264")).save_to(
                str(result), format=Types.VideoContainer("mp4"), codec=Types.VideoCodec("h264"))
        else:
            os.replace(silent, result)
        inspect_video(result, frames, width, height,
                      audio["waveform"].shape[-1] if audio else None,
                      audio["sample_rate"] if audio else None, full_decode=True)
        os.replace(result, destination)
    finally:
        silent.unlink(missing_ok=True)
        result.unlink(missing_ok=True)
