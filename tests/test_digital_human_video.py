import sys
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.argv = [sys.argv[0], "--cpu"]

from test_digital_human import module
from comfy_api.latest import Types
from comfy_extras.nodes_video import CreateVideo
import av
import torch

media = sys.modules[module.__package__ + ".digital_human_video"]


class TestDigitalHumanVideo(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name) / "clips"
        self.root.mkdir()

    def video(self, name="part.mp4", frames=51, offset=0):
        path = self.root / name
        if not offset:
            audio = {"waveform": torch.zeros(1, 1, frames * 2000), "sample_rate": 48000}
            clip = CreateVideo.execute(torch.zeros(frames, 32, 32, 3), 24, audio)[0]
            clip.save_to(str(path), format=Types.VideoContainer("mp4"), codec=Types.VideoCodec("h264"))
        else:
            with av.open(str(path), "w") as container:
                stream = container.add_stream("libx264", rate=24)
                stream.width, stream.height, stream.pix_fmt = 32, 32, "yuv420p"
                for index in range(frames):
                    frame = av.VideoFrame(32, 32, "yuv420p")
                    for plane in frame.planes:
                        plane.update(bytes(plane.buffer_size))
                    frame.pts, frame.time_base = index + offset, Fraction(1, 24)
                    container.mux(stream.encode(frame))
                container.mux(stream.encode())
        return path

    def test_real_media_full_decode_detects_wrong_frames_and_preserves_audio(self):
        path = self.video()
        result = media.inspect_video(path, 51, 32, 32, 51 * 2000, 48000, full_decode=True)
        self.assertEqual(result["frames"], 51)
        with self.assertRaises(ValueError):
            media.inspect_video(path, 50, 32, 32, full_decode=True)
        with self.assertRaises(ValueError):
            media.inspect_video(path, 51, 64, 32)

    def test_normalization_corrects_shift_and_exact_frames_and_muxes_source(self):
        path = self.video(frames=52, offset=1)
        with self.assertRaises(ValueError):
            media.inspect_video(path, 52, 32, 32, full_decode=True)
        audio = {"waveform": torch.zeros(1, 1, 51 * 2000), "sample_rate": 48000}
        media.normalize_video(path, path, 51, 32, 32, audio)
        result = media.inspect_video(path, 51, 32, 32, 51 * 2000, 48000, full_decode=True)
        self.assertEqual(result["first_pts"], 0)
        self.assertEqual(list(self.root.glob("*.tmp")), [])

    def test_failed_normalization_keeps_original_file(self):
        path = self.video(frames=20)
        original = path.read_bytes()
        with self.assertRaises(ValueError):
            media.normalize_video(path, path, 51, 32, 32)
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(list(self.root.glob("*.tmp")), [])

if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]], verbosity=2)
