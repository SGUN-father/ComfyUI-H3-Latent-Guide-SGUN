import importlib.util
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
from unittest.mock import patch

import av
import numpy as np


spec = importlib.util.spec_from_file_location("seam_preview", Path(__file__).resolve().parents[1] / "seam_preview.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class TestSeamPreview(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def make_clip(self, name, colors, *, audio=True, frequency=440, size=(64, 48), fps=24):
        path = self.root / name
        with av.open(str(path), "w") as container:
            video = module._video_stream(container, *size)
            video.codec_context.framerate = Fraction(fps)
            sound = container.add_stream("aac", rate=48000) if audio else None
            if sound:
                sound.layout = "stereo"
            for index, color in enumerate(colors):
                pixels = np.empty((size[1], size[0], 3), dtype=np.uint8)
                pixels[:] = color
                frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
                frame = frame.reformat(format="yuv420p", src_colorspace="ITU709", dst_colorspace="ITU709")
                frame.pts, frame.time_base = index, Fraction(1, fps)
                container.mux(video.encode(frame))
            container.mux(video.encode())
            if sound:
                count = len(colors) * 48000 // fps
                sine = np.sin(np.arange(count) * (2 * np.pi * frequency / 48000)).astype(np.float32) * 0.2
                module._write_audio(container, sound, [np.stack((sine, sine))])
        return path

    def read(self, path):
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            info = {"fps": stream.average_rate, "trc": stream.codec_context.color_trc,
                    "primaries": stream.codec_context.color_primaries,
                    "colorspace": stream.codec_context.colorspace}
            frames = [(frame.pts * frame.time_base, frame.to_ndarray(format="rgb24")) for frame in container.decode(stream)]
        with av.open(str(path)) as container:
            audio = np.concatenate([frame.to_ndarray() for frame in container.decode(audio=0)], axis=1) if container.streams.audio else None
        return info, frames, audio

    def test_exact_windows_order_audio_timestamps_and_context(self):
        previous = self.make_clip("previous.mp4", [(255, 0, 0)] * 24 + [(0, 255, 0)] * 24, frequency=440)
        current = self.make_clip("current.mp4", [(0, 0, 255)] * 48 + [(255, 0, 0)] * 24, frequency=880)
        target, context = self.root / "seam.mp4", self.root / "context.mp4"
        result = module.build_seam_preview(previous, current, target, context_path=context)
        self.assertEqual((result["join_frame"], result["frame_count"], result["context_frames"]), (24, 72, 24))
        self.assertTrue(result["has_audio"])
        info, frames, audio = self.read(target)
        self.assertEqual(info, {"fps": 24, "trc": 13, "primaries": 1, "colorspace": 1})
        self.assertEqual(len(frames), 72)
        self.assertEqual([pts for pts, _ in frames], [Fraction(i, 24) for i in range(72)])
        self.assertGreater(frames[0][1][:, :, 1].mean(), 240)
        self.assertGreater(frames[24][1][:, :, 2].mean(), 240)
        self.assertEqual(audio.shape[0], 2)
        self.assertGreaterEqual(audio.shape[1], 144000)
        for start, frequency in ((12000, 440), (72000, 880)):
            spectrum = np.abs(np.fft.rfft(audio[0, start:start + 24000]))
            self.assertAlmostEqual(np.argmax(spectrum) * 2, frequency, delta=2)
        _, context_frames, context_audio = self.read(context)
        self.assertEqual(len(context_frames), 24)
        self.assertGreater(context_frames[-1][1][:, :, 1].mean(), 240)
        self.assertGreaterEqual(context_audio.shape[1], 48000)

    def test_static_and_gradual_change_do_not_warn(self):
        previous = self.make_clip("static.mp4", [(90, 90, 90)] * 32, audio=False)
        current = self.make_clip("gradient.mp4", [(90 + index, 90 + index, 90 + index) for index in range(48)], audio=False)
        result = module.build_seam_preview(previous, current, self.root / "gradient-seam.mp4")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["events"], [])
        self.assertFalse(result["has_audio"])
        _, _, audio = self.read(self.root / "gradient-seam.mp4")
        self.assertIsNone(audio)

    def test_boundary_and_delayed_jumps_are_conservative_hints(self):
        previous = self.make_clip("dark.mp4", [(10, 10, 10)] * 24, audio=False)
        current = self.make_clip("jumps.mp4", [(160, 160, 160)] * 12 + [(30, 30, 30)] * 36, audio=False)
        result = module.build_seam_preview(previous, current, self.root / "jumps-seam.mp4")
        self.assertEqual(result["status"], "warning")
        self.assertEqual([event["frame"] for event in result["events"]], [24, 36])
        self.assertEqual(result["events"][0]["kind"], "brightness_step")
        self.assertIn("疑似", result["messages"][0])
        self.assertIn("大动作或换光", result["messages"][-1])

    def test_pixel_jump_without_mean_brightness_change(self):
        checker = np.zeros((48, 64, 3), dtype=np.uint8)
        checker[:, :32] = 240
        previous = self.make_clip("left.mp4", [checker] * 24, audio=False)
        current = self.make_clip("right.mp4", [240 - checker] * 48, audio=False)
        result = module.build_seam_preview(previous, current, self.root / "spatial.mp4")
        self.assertEqual(result["events"][0]["kind"], "pixel_jump")
        self.assertEqual(result["events"][0]["frame"], 24)

    def test_short_sources_missing_audio_and_bounded_windows(self):
        previous = self.make_clip("short.mp4", [(50, 50, 50)] * 5, audio=False, size=(800, 400))
        current = self.make_clip("long.mp4", [(50, 50, 50)] * 150, size=(800, 400))
        windows = []
        original = module._frames

        def observed(info, width, height):
            windows.append((info["start"], info["count"], width, height))
            yield from original(info, width, height)

        with patch.object(module, "_frames", observed):
            result = module.build_seam_preview(previous, current, self.root / "bounded.mp4")
        self.assertEqual(windows, [(0, 5, 640, 320), (0, 48, 640, 320)])
        self.assertEqual(result["frame_count"], 53)
        self.assertEqual(result["join_frame"], 5)
        _, _, audio = self.read(self.root / "bounded.mp4")
        self.assertLess(float(np.abs(audio[:, :8000]).mean()), 0.001)
        self.assertGreater(float(np.abs(audio[:, 12000:16000]).mean()), 0.05)

    def test_bad_input_raises_without_replacing_existing_destination(self):
        previous = self.make_clip("valid.mp4", [(60, 60, 60)] * 24, audio=False)
        corrupt = self.root / "corrupt.mp4"
        corrupt.write_bytes(b"not an mp4")
        target = self.root / "existing.mp4"
        target.write_bytes(b"original")
        with self.assertRaises(av.error.FFmpegError):
            module.build_seam_preview(previous, corrupt, target)
        self.assertEqual(target.read_bytes(), b"original")
        with self.assertRaises(ValueError):
            module.build_seam_preview(previous, previous, previous)
        with patch.object(module, "_frames", side_effect=ValueError("truncated")):
            with self.assertRaisesRegex(ValueError, "truncated"):
                module.build_seam_preview(previous, previous, target)
        self.assertEqual(target.read_bytes(), b"original")
        self.assertFalse(list(self.root.glob(".*.mp4")))

    def test_saved_small_context_can_preview_an_original_size_retry(self):
        previous = self.make_clip("wide-previous.mp4", [(80, 80, 80)] * 24, size=(1376, 864))
        current = self.make_clip("wide-current.mp4", [(80, 80, 80)] * 48, size=(1376, 864))
        context = self.root / "small-context.mp4"
        module.build_seam_preview(previous, current, self.root / "first-preview.mp4", context_path=context)
        result = module.build_seam_preview(context, current, self.root / "retry-preview.mp4")
        self.assertEqual((result["width"], result["height"]), (640, 400))
        self.assertEqual((result["join_frame"], result["frame_count"]), (24, 72))
        self.assertEqual(result["status"], "ok")
        portrait = self.make_clip("portrait.mp4", [(80, 80, 80)] * 24, size=(48, 64), audio=False)
        with self.assertRaisesRegex(ValueError, "宽高比"):
            module.build_seam_preview(context, portrait, self.root / "bad-ratio.mp4")


if __name__ == "__main__":
    unittest.main(verbosity=2)
