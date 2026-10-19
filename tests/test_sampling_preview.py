import base64
import importlib.util
import io
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

import torch
from PIL import Image

COMFY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(COMFY_ROOT))
sys.argv = [sys.argv[0], "--cpu"]

from comfy.nested_tensor import NestedTensor

spec = importlib.util.spec_from_file_location("h3_sampling_preview_test", Path(__file__).resolve().parents[1] / "sampling_preview.py")
preview = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preview)


class Decoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inputs = []

    def decode(self, samples):
        self.inputs.append((tuple(samples.shape), samples[0, 0, :, 0, 0].tolist()))
        tokens = samples.shape[2]
        count = 1 if tokens == 1 else ((tokens - 2) // 5) * 17 + 5
        values = torch.linspace(0, 1, count).reshape(1, 1, count, 1, 1)
        return values.expand(1, 3, count, samples.shape[-2] * 16, samples.shape[-1] * 16)


class TestSamplingPreview(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.server = SimpleNamespace(client_id="client-a", send_sync=lambda *args: self.events.append(args))
        self.server_patch = mock.patch.object(preview.PromptServer, "instance", self.server, create=True)
        self.server_patch.start()
        self.addCleanup(self.server_patch.stop)
        self.context_patch = mock.patch.object(preview, "get_executing_context", return_value=SimpleNamespace(prompt_id="prompt-a"))
        self.context_patch.start()
        self.addCleanup(self.context_patch.stop)
        self.active = SimpleNamespace(model=SimpleNamespace(latent_format=SimpleNamespace(process_out=lambda x: x * 2)))

    def fake_loaded(self, instance):
        decoder = Decoder()
        patcher = SimpleNamespace(model=decoder, load_device=torch.device("cpu"), detach=mock.Mock())
        instance.patcher = patcher
        instance.vae = SimpleNamespace(patcher=patcher, first_stage_model=decoder,
                                       vae_dtype=torch.float32, device=torch.device("cpu"),
                                       memory_used_decode=lambda shape, dtype: 1024)
        return patcher

    def test_cadence_first_interval_and_final_steps(self):
        instance = preview.H3SamplingPreview("23", interval=2)
        video = torch.zeros(1, 24, 20, 1, 1)
        with mock.patch.object(instance, "_frames", return_value=(["data:image/jpeg;base64,test"], [0])) as decode:
            for step in range(8):
                instance.update(2, step, video, 8, self.active)
        self.assertEqual(decode.call_count, 5)
        self.assertEqual([event[1]["step"] for event in self.events], [1, 3, 5, 7, 8])
        self.assertTrue(all(event[1]["segment"] == 2 for event in self.events))
        self.assertTrue(all(event[2] == "client-a" for event in self.events))
        self.assertTrue(all(event[1]["fps"] == 10 and event[1]["source_fps"] == 24 for event in self.events))

    def test_nested_streams_unchanged_all_tokens_decoded_once_and_resized_before_decode(self):
        instance = preview.H3SamplingPreview("23", max_resolution=64, fps=12)
        patcher = self.fake_loaded(instance)
        video = torch.arange(72, dtype=torch.float32).reshape(1, 1, 72, 1, 1).expand(1, 24, 72, 8, 16).clone()
        audio = torch.randn(1, 32, 2, 100)
        original_video, original_audio = video.clone(), audio.clone()
        instance.set_segment(total_frames=243)
        with mock.patch.object(preview.mm, "load_models_gpu") as load:
            instance.update(1, 0, NestedTensor((video, audio)), 8, self.active)
        load.assert_called_once_with([patcher, self.active], memory_required=1024)
        self.assertTrue(torch.equal(video, original_video))
        self.assertTrue(torch.equal(audio, original_audio))
        self.assertEqual(patcher.model.inputs, [((1, 24, 72, 2, 4), [float(index * 2) for index in range(72)])])
        payload = self.events[0][1]
        self.assertEqual(payload["fps"], 12)
        self.assertEqual(payload["source_fps"], 24)
        self.assertEqual(payload["frame_count"], 243)
        self.assertEqual(payload["source_frames"], 243)
        self.assertEqual(payload["trim_frames"], 0)
        self.assertEqual(len(payload["frames"]), 243)
        self.assertEqual(len(payload["frame_indices"]), 243)
        self.assertEqual(payload["frame_indices"], list(range(243)))
        self.assertEqual(len(set(payload["frames"])), 243)
        for frame in payload["frames"]:
            image = Image.open(io.BytesIO(base64.b64decode(frame.split(",", 1)[1])))
            self.assertEqual(image.size, (64, 32))
            self.assertEqual(image.format, "JPEG")

    def test_continuation_skips_overlap_without_shortening_remaining_timeline(self):
        instance = preview.H3SamplingPreview("23")
        patcher = self.fake_loaded(instance)
        video = torch.arange(77, dtype=torch.float32).reshape(1, 1, 77, 1, 1).expand(1, 24, 77, 1, 1)
        instance.set_segment(total_frames=260, trim_frames=22)
        with mock.patch.object(preview.mm, "load_models_gpu"):
            instance.update(2, 0, video, 8, self.active)
        payload = self.events[0][1]
        self.assertEqual(payload["frame_count"], 238)
        self.assertEqual(payload["source_frames"], 260)
        self.assertEqual(payload["trim_frames"], 22)
        self.assertEqual(len(payload["frames"]), 238)
        self.assertEqual(len(payload["frame_indices"]), 238)
        self.assertEqual(patcher.model.inputs, [((1, 24, 77, 1, 1), [float(index * 2) for index in range(77)])])
        self.assertEqual(payload["frame_indices"], list(range(238)))
        first = Image.open(io.BytesIO(base64.b64decode(payload["frames"][0].split(",", 1)[1])))
        self.assertEqual(first.getpixel((0, 0)), (21, 21, 21))

    def test_fps_changes_playback_rate_only_and_segment_setup_replaces_previous_grid(self):
        instance = preview.H3SamplingPreview("23", fps=60)
        self.fake_loaded(instance)
        with mock.patch.object(preview.mm, "load_models_gpu"):
            instance.set_segment(total_frames=39, trim_frames=22)
            instance.update(2, 0, torch.zeros(1, 24, 12, 1, 1), 8, self.active)
            instance.set_segment(total_frames=22)
            instance.update(3, 0, torch.zeros(1, 24, 7, 1, 1), 8, self.active)
        first, second = [event[1] for event in self.events]
        self.assertEqual(first["frame_count"], 17)
        self.assertEqual(first["fps"], 60)
        self.assertEqual(len(first["frame_indices"]), 17)
        self.assertEqual(second["frame_count"], 22)
        self.assertEqual(second["trim_frames"], 0)
        self.assertEqual(len(second["frame_indices"]), 22)

    def test_frame_count_mismatch_disables_preview_instead_of_sending_truncated_timeline(self):
        instance = preview.H3SamplingPreview("23")
        self.fake_loaded(instance)
        instance.set_segment(total_frames=243)
        with mock.patch.object(preview.mm, "load_models_gpu"), mock.patch.object(preview.mm, "cleanup_models"):
            instance.update(1, 0, torch.zeros(1, 24, 7, 1, 1), 8, self.active)
        self.assertTrue(instance.failed)
        self.assertEqual(self.events[0][1]["status"], "error")
        self.assertEqual(self.events[0][1]["frames"], [])

    def test_single_frame_does_not_duplicate(self):
        instance = preview.H3SamplingPreview("23")
        patcher = self.fake_loaded(instance)
        with mock.patch.object(preview.mm, "load_models_gpu"):
            instance.update(1, 0, torch.zeros(1, 24, 1, 1, 1), 1, self.active)
        self.assertEqual(len(patcher.model.inputs), 1)
        self.assertEqual(len(self.events[0][1]["frames"]), 1)

    def test_decode_failure_is_reported_once_and_execution_can_continue(self):
        instance = preview.H3SamplingPreview("23")
        patcher = self.fake_loaded(instance)
        with mock.patch.object(instance, "_frames", side_effect=RuntimeError("out of memory")) as decode:
            with mock.patch.object(preview.mm, "cleanup_models"):
                for step in range(8):
                    instance.update(1, step, torch.zeros(1, 24, 1, 1, 1), 8, self.active)
                instance.finish()
        self.assertEqual(decode.call_count, 1)
        self.assertTrue(instance.failed)
        self.assertTrue(instance.closed)
        self.assertIsNone(instance.patcher)
        patcher.detach.assert_called_once()
        self.assertEqual([event[1]["status"] for event in self.events], ["error"])

    def test_interrupt_propagates(self):
        instance = preview.H3SamplingPreview("23")
        with mock.patch.object(preview.mm, "throw_exception_if_processing_interrupted", side_effect=preview.mm.InterruptProcessingException):
            with self.assertRaises(preview.mm.InterruptProcessingException):
                instance.update(1, 0, torch.zeros(1, 24, 1, 1, 1), 8, self.active)
        self.assertEqual(self.events, [])

    def test_corrupt_weights_disable_preview_instead_of_sampling(self):
        instance = preview.H3SamplingPreview("23")
        with mock.patch.object(instance, "_load", side_effect=preview.SafetensorError("invalid header")):
            instance.update(1, 0, torch.zeros(1, 24, 1, 1, 1), 8, self.active)
        self.assertTrue(instance.failed)
        self.assertEqual(self.events[0][1]["status"], "error")
        self.assertIn("invalid header", self.events[0][1]["message"])

    def test_interrupt_after_native_decode_propagates(self):
        instance = preview.H3SamplingPreview("23")
        patcher = self.fake_loaded(instance)
        with mock.patch.object(preview.mm, "load_models_gpu"):
            with mock.patch.object(preview.mm, "throw_exception_if_processing_interrupted", side_effect=[None, None, preview.mm.InterruptProcessingException()]):
                with self.assertRaises(preview.mm.InterruptProcessingException):
                    instance.update(1, 0, torch.zeros(1, 24, 1, 1, 1), 8, self.active)
        self.assertEqual(len(patcher.model.inputs), 1)
        self.assertFalse(instance.failed)
        self.assertEqual(self.events, [])
        with mock.patch.object(preview.mm, "cleanup_models"):
            instance.close()

    def test_finish_releases_once_and_keeps_completed_frames_client_scoped(self):
        instance = preview.H3SamplingPreview("23")
        patcher = self.fake_loaded(instance)
        self.server.client_id = "client-b"
        preview.get_executing_context.return_value = SimpleNamespace(prompt_id="prompt-b")
        with mock.patch.object(preview.mm, "cleanup_models"):
            instance.finish()
            instance.finish()
        patcher.detach.assert_called_once()
        self.assertEqual(len(self.events), 1)
        self.assertEqual(self.events[0][1]["status"], "complete")
        self.assertEqual(self.events[0][1]["prompt_id"], "prompt-a")
        self.assertEqual(self.events[0][2], "client-a")

    def test_missing_client_never_broadcasts(self):
        self.server.client_id = None
        instance = preview.H3SamplingPreview("23")
        instance.finish()
        self.assertEqual(self.events, [])

    def test_missing_execution_context_never_sends_an_unstamped_preview(self):
        preview.get_executing_context.return_value = None
        instance = preview.H3SamplingPreview("23")
        instance.finish()
        self.assertEqual(self.events, [])

    def test_framewise_checkpoint_is_rejected(self):
        instance = preview.H3SamplingPreview("23")
        state = {"1.weight": torch.empty(96, 24, 3, 3)}
        with mock.patch.object(preview.folder_paths, "get_full_path", return_value="framewise.safetensors"), \
                mock.patch.object(preview.comfy.utils, "load_torch_file", return_value=state), \
                mock.patch.object(preview.comfy.sd, "VAE") as load_vae:
            instance.update(1, 0, torch.zeros(1, 24, 1, 1, 1), 1, self.active)
        load_vae.assert_not_called()
        self.assertTrue(instance.failed)
        self.assertIn("TAEHV", self.events[0][1]["message"])

    def test_missing_temporal_checkpoint_does_not_fall_back_to_2d_weights(self):
        instance = preview.H3SamplingPreview("23")
        with mock.patch.object(preview.folder_paths, "get_full_path", return_value=None) as resolve:
            instance.update(1, 0, torch.zeros(1, 24, 1, 1, 1), 1, self.active)
        resolve.assert_called_once_with("vae_approx", "taeh3_temporal.safetensors")
        self.assertTrue(instance.failed)
        self.assertIn("taeh3_temporal.safetensors", self.events[0][1]["message"])

    def test_real_temporal_taeh3_checkpoint_decodes_motion_on_cpu(self):
        path = preview.folder_paths.get_full_path("vae_approx", "taeh3_temporal.safetensors")
        if path is None:
            self.skipTest("No local temporal taeh3 checkpoint")
        instance = preview.H3SamplingPreview("23", max_resolution=32)
        active = preview.comfy.model_patcher.ModelPatcher(torch.nn.Linear(1, 1), torch.device("cpu"), torch.device("cpu"))
        active.model.latent_format = SimpleNamespace(process_out=lambda value: value)
        try:
            torch.manual_seed(7)
            for segment, tokens, source_frames, trim in [(1, 7, 22, 0), (2, 72, 243, 0), (3, 77, 260, 22)]:
                samples = torch.randn(1, 24, tokens, 2, 2)
                instance.set_segment(source_frames, trim)
                instance.update(segment, 0, samples, 1, active)
                self.assertFalse(instance.failed)
                payload = self.events[-1][1]
                count = source_frames - trim
                self.assertEqual(payload["status"], "sampling")
                self.assertEqual(len(payload["frames"]), count)
                self.assertEqual(payload["frame_indices"], list(range(count)))
                self.assertEqual(len(set(payload["frames"])), count)
                image = Image.open(io.BytesIO(base64.b64decode(payload["frames"][0].split(",", 1)[1])))
                self.assertEqual(image.size, (32, 32))
            self.assertIsInstance(instance.patcher, preview.comfy.model_patcher.ModelPatcher)
            self.assertTrue(instance.vae.first_stage_model.is_h3)
            self.assertEqual(instance.vae.first_stage_model.patch_size, 2)
        finally:
            instance.close()
            active.detach()
            preview.mm.cleanup_models()


if __name__ == "__main__":
    sys.argv = [sys.argv[0]]
    unittest.main()
