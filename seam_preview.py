"""Small, local seam previews and conservative frame-change hints."""
import os
import tempfile
from contextlib import ExitStack
from fractions import Fraction
from pathlib import Path

import av
import numpy as np


FPS = 24
AUDIO_RATE = 48000


def _video_info(path):
    with av.open(str(path)) as container:
        if len(container.streams.video) != 1:
            raise ValueError(f"接缝预览需要一个视频轨：{path}")
        stream = container.streams.video[0]
        if stream.average_rate != FPS or stream.frames < 1:
            raise ValueError(f"接缝预览需要有准确帧数的 24fps 视频：{path}")
        return {"path": path, "frames": stream.frames, "width": stream.width,
                "height": stream.height, "origin": Fraction(stream.start_time or 0) * stream.time_base,
                "audio": bool(container.streams.audio)}


def _frames(info, width, height):
    start, end = info["start"], info["start"] + info["count"]
    with av.open(str(info["path"])) as container:
        stream = container.streams.video[0]
        begin = info["origin"] + Fraction(start, FPS)
        if start:
            container.seek(int(begin / stream.time_base), stream=stream, backward=True)
        count = 0
        for frame in container.decode(stream):
            if frame.pts is None:
                raise ValueError(f"接缝源视频缺少时间戳：{info['path']}")
            timestamp = frame.pts * frame.time_base
            if timestamp < begin:
                continue
            if count == info["count"]:
                break
            if timestamp != begin + Fraction(count, FPS):
                raise ValueError(f"接缝源视频时间戳不连续：{info['path']}")
            count += 1
            yield frame.reformat(width=width, height=height, format="rgb24", dst_colorspace="ITU709")
        if count != end - start:
            raise ValueError(f"接缝源视频帧数不足：{info['path']}")


def _audio_window(info):
    samples = info["count"] * AUDIO_RATE // FPS
    audio = np.zeros((2, samples), dtype=np.float32)
    if not info["audio"]:
        return audio
    covered = np.zeros(samples, dtype=np.bool_)
    begin = info["origin"] + Fraction(info["start"], FPS)
    end = begin + Fraction(info["count"], FPS)
    resampler = av.AudioResampler(format="fltp", layout="stereo", rate=AUDIO_RATE)
    with av.open(str(info["path"])) as container:
        stream = container.streams.audio[0]
        container.seek(int(begin / stream.time_base), stream=stream, backward=True)
        decoded = False
        for frame in container.decode(stream):
            if frame.pts is None:
                raise ValueError(f"接缝源音频缺少时间戳：{info['path']}")
            if frame.pts * frame.time_base >= end:
                break
            decoded = True
            for converted in resampler.resample(frame):
                offset = round((converted.pts * converted.time_base - begin) * AUDIO_RATE)
                left, right = max(0, offset), min(samples, offset + converted.samples)
                if right > left:
                    audio[:, left:right] = converted.to_ndarray()[:, left - offset:right - offset]
                    covered[left:right] = True
        if not decoded:
            raise ValueError(f"接缝窗口没有可解码的音频：{info['path']}")
        for converted in resampler.resample(None):
            offset = round((converted.pts * converted.time_base - begin) * AUDIO_RATE)
            left, right = max(0, offset), min(samples, offset + converted.samples)
            if right > left:
                audio[:, left:right] = converted.to_ndarray()[:, left - offset:right - offset]
                covered[left:right] = True
    if not covered.all():
        raise ValueError(f"接缝源音频未覆盖完整窗口：{info['path']}")
    return audio


def _video_stream(container, width, height):
    stream = container.add_stream("libx264", rate=FPS)
    stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
    stream.options = {"crf": "18", "preset": "fast"}
    stream.codec_context.color_primaries = 1
    stream.codec_context.color_trc = 13  # IEC 61966-2-1 (sRGB)
    stream.codec_context.colorspace = 1
    stream.codec_context.color_range = 1
    return stream


def _write_audio(container, stream, windows):
    offset = 0
    for samples in windows:
        for start in range(0, samples.shape[1], 1024):
            frame = av.AudioFrame.from_ndarray(np.ascontiguousarray(samples[:, start:start + 1024]),
                                               format="fltp", layout="stereo")
            frame.sample_rate, frame.time_base = AUDIO_RATE, Fraction(1, AUDIO_RATE)
            frame.pts = offset + start
            container.mux(stream.encode(frame))
        offset += samples.shape[1]
    container.mux(stream.encode())


def _events(differences, shifts, join):
    events = []
    for index in range(join, len(differences)):
        neighbours = [i for i in range(max(1, index - 4), min(len(differences), index + 5)) if i != index]
        baseline = float(np.median([differences[i] for i in neighbours])) if neighbours else 0.0
        rgb_baseline = float(np.median([shifts[i] for i in neighbours])) if neighbours else 0.0
        kind = None
        if shifts[index] > max(0.08, rgb_baseline * 4):
            kind = "brightness_step"
        elif differences[index] > max(0.16, baseline * 4):
            kind = "pixel_jump"
        if kind:
            events.append({"frame": index, "kind": kind, "difference": round(differences[index], 4),
                           "rgb_shift": round(shifts[index], 4)})
    return events


def build_seam_preview(previous_path, current_path, output_path, *, context_path=None):
    """Write the last second + first two seconds; raise on invalid media.

    `join_frame` is the zero-based first frame from the current segment.
    Hints describe local pixel changes, not a video quality score.
    """
    previous, current = _video_info(previous_path), _video_info(current_path)
    previous_ratio = previous["width"] / previous["height"]
    current_ratio = current["width"] / current["height"]
    ratio_tolerance = 2 * (1 + max(previous_ratio, current_ratio)) / min(previous["height"], current["height"])
    if abs(previous_ratio - current_ratio) > ratio_tolerance:
        raise ValueError("接缝源视频的画面宽高比不一致。")
    previous.update(start=max(0, previous["frames"] - FPS), count=min(FPS, previous["frames"]))
    current.update(start=0, count=min(FPS * 2, current["frames"]))
    scale = min(1, 640 / max(current["width"], current["height"]))
    width, height = (max(2, int(current[key] * scale) // 2 * 2) for key in ("width", "height"))
    destinations = [Path(output_path)] + ([Path(context_path)] if context_path is not None else [])
    resolved = [path.resolve() for path in destinations]
    if len(set(resolved)) != len(resolved) or any(path in {Path(previous_path).resolve(), Path(current_path).resolve()} for path in resolved):
        raise ValueError("接缝预览、上下文和源视频必须使用不同路径。")
    temporary = []
    differences, shifts, last, last_mean = [], [], None, None
    has_audio = previous["audio"] or current["audio"]
    try:
        with ExitStack() as stack:
            outputs = []
            for index, destination in enumerate(destinations):
                destination.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(prefix=f".{destination.stem}-", suffix=".mp4", dir=destination.parent, delete=False) as handle:
                    temporary.append(Path(handle.name))
                container = stack.enter_context(av.open(str(temporary[-1]), "w", format="mp4"))
                video = _video_stream(container, width, height)
                audio = container.add_stream("aac", rate=AUDIO_RATE) if (has_audio if index == 0 else previous["audio"]) else None
                if audio:
                    audio.layout = "stereo"
                outputs.append((container, video, audio))
            position = 0
            for segment_index, info in enumerate((previous, current)):
                for frame in _frames(info, width, height):
                    pixels = frame.reformat(width=96, height=54, format="rgb24").to_ndarray()
                    mean = pixels.mean(axis=(0, 1)) / 255
                    differences.append(float(np.abs(pixels.astype(np.int16) - last).mean() / 255) if last is not None else 0.0)
                    shifts.append(float(np.max(np.abs(mean - last_mean))) if last_mean is not None else 0.0)
                    last, last_mean = pixels.astype(np.int16), mean
                    encoded = frame.reformat(format="yuv420p", src_colorspace="ITU709", dst_colorspace="ITU709")
                    encoded.pts, encoded.time_base = position, Fraction(1, FPS)
                    encoded.color_primaries, encoded.color_trc, encoded.colorspace = 1, 13, 1
                    for container, video, _ in outputs[:1] if segment_index else outputs:
                        container.mux(video.encode(encoded))
                    position += 1
            for container, video, _ in outputs:
                container.mux(video.encode())
            if has_audio:
                windows = [_audio_window(info) for info in (previous, current)]
                _write_audio(outputs[0][0], outputs[0][2], windows)
                if len(outputs) > 1 and outputs[1][2]:
                    _write_audio(outputs[1][0], outputs[1][2], windows[:1])
        for source, destination in zip(temporary, destinations):
            os.replace(source, destination)
    finally:
        for path in temporary:
            path.unlink(missing_ok=True)
    events = _events(differences, shifts, previous["count"])
    messages = []
    for kind, label in (("pixel_jump", "疑似画面跳变"), ("brightness_step", "疑似亮度或颜色台阶")):
        matching = [event for event in events if event["kind"] == kind]
        if matching:
            seconds = (matching[0]["frame"] - previous["count"]) / FPS
            messages.append(f"新段 {seconds:.2f} 秒处{label}。")
    if events:
        messages.append("大动作或换光也可能触发，请结合预览判断。")
    return {"status": "warning" if events else "ok", "messages": messages, "events": events,
            "join_frame": previous["count"], "frame_count": position, "fps": FPS,
            "width": width, "height": height, "has_audio": has_audio,
            "context_frames": previous["count"] if context_path is not None else 0}
