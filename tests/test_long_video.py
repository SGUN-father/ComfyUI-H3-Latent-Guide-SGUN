import asyncio
import gc
import importlib.util
import json
import sys
import tempfile
import unittest
import weakref
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import av
import torch

COMFY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(COMFY_ROOT))
sys.argv = [sys.argv[0], "--cpu"]

from comfy.nested_tensor import NestedTensor
from comfy_api.latest import io
from comfy_extras.nodes_video import SaveVideo
from comfy_execution.graph_utils import ExecutionBlocker
import execution
import folder_paths
from comfy.model_base import MiniMaxH3
from comfy.model_patcher import ModelPatcher
from comfy.patcher_extension import WrappersMP
import comfy.sampler_helpers
import comfy.utils

spec = importlib.util.spec_from_file_location("h3_long_video_test", Path(__file__).resolve().parents[1] / "__init__.py")
package = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = package
spec.loader.exec_module(package)
module = sys.modules[package.__name__ + ".long_video"]
latent_io = sys.modules[package.__name__ + ".latent_io"]
LongVideo, PromptPlan, LoadLatent = package.H3LongVideo, package.H3VideoPromptPlan, package.H3LoadLatent


def native_latent(frames=141, dtype=torch.float32, value=0.1):
    t = (frames - 5) // 17 * 5 + 2
    return {"samples": NestedTensor((torch.full((1, 24, t, 2, 2), value, dtype=dtype),
                                     torch.full((1, 32, 2, round(frames * 5 / 3)), value, dtype=dtype)))}


class TestLongVideo(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.directory = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.stack.enter_context(patch.object(folder_paths, "get_output_directory", return_value=str(self.directory)))
        self.stack.enter_context(patch.object(module.ResolutionSelector, "execute", return_value=io.NodeOutput(32, 32)))
        self.stack.enter_context(patch.object(LongVideo, "hidden", SimpleNamespace(
            prompt={"test": True}, extra_pnginfo={"workflow": {"nodes": []}})))
        LongVideo.GET_NODE_INFO_V1()

    def run_node(self, count=4, plan=None, **kwargs):
        return LongVideo.execute(kwargs.pop("model", None), None, None, None, plan or {"prompts": ["one"], "repeat_last": True},
                                 count, 5.0, "16:9 (Widescreen)", 0.4, kwargs.pop("seed", 100),
                                 8, "res_multistep", "simple", sigmas=kwargs.pop("sigmas", torch.tensor([1.0, 0.0])),
                                 **kwargs)

    def test_prompt_order_shared_prompt_and_empty_section(self):
        plan = PromptPlan.execute("identity", "first\r\n---\r\nsecond", False)[0]
        self.assertEqual(plan["prompts"], ["identity\n\nfirst", "identity\n\nsecond"])
        self.assertFalse(plan["repeat_last"])
        self.assertEqual(PromptPlan.execute("identity", "", True)[0]["prompts"], ["identity"])
        with self.assertRaisesRegex(ValueError, "空段"):
            PromptPlan.execute("identity", "first\n---\n", False)

    def test_insufficient_prompts_fail_before_sampling_or_creating_files(self):
        with patch.object(LongVideo, "_render_segment") as render:
            with self.assertRaisesRegex(ValueError, "需要 4 段"):
                self.run_node(plan={"prompts": ["one"], "repeat_last": False})
            render.assert_not_called()
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_start_segment_and_total_segment_count(self):
        for start, count, repeat, expected in [(2, 2, False, ["two", "three"]),
                                               (3, 3, True, ["three"] * 3)]:
            with self.subTest(start=start), patch.object(LongVideo, "_render_segment",
                    return_value=(native_latent(), object(), 119)) as render:
                plan = PromptPlan.execute("identity", "one\n---\ntwo\n---\nthree", repeat, start)[0]
                output = self.run_node(count, plan)
                self.assertEqual(json.loads(output[2])["segments"], count)
                self.assertEqual(len(output[3]), count)
                self.assertEqual([call.args[4] for call in render.call_args_list],
                                 ["identity\n\n" + text for text in expected])
        for invalid in (0, 4):
            for repeat in (False, True):
                with self.assertRaisesRegex(ValueError, "不存在"):
                    PromptPlan.execute("identity", "one\n---\ntwo\n---\nthree", repeat, invalid)

    def test_custom_sigmas_skip_scheduler_with_previous_latent(self):
        custom = torch.tensor([1., .9882, .973, .9524, .9231, .878, .8, .6316, .4737, .1579, 0.])
        with patch.object(module.BasicScheduler, "execute", side_effect=AssertionError("scheduler ran")) as scheduler, \
             patch.object(LongVideo, "_render_segment", return_value=(native_latent(), object(), 119)) as render:
            self.run_node(sigmas=custom, denoise=0.5, previous_latent=native_latent())
            scheduler.assert_not_called()
            self.assertEqual(render.call_count, 4)
            for call in render.call_args_list:
                self.assertIs(call.args[12], custom)
                self.assertEqual(package._av_streams(call.args[8], "context")[2], 39)

    def test_schema_native_references_and_safe_defaults(self):
        schema = LongVideo.define_schema()
        inputs = {item.id: item for item in schema.inputs}
        official = {item.id: item for item in module.MiniMaxH3ReferenceToVideo.define_schema().inputs}
        for name in ("ref_images", "ref_videos", "ref_video_audios", "ref_audios"):
            self.assertTrue(inputs[name].optional)
            self.assertEqual(inputs[name].template.input.io_type, official[name].template.input.io_type)
            self.assertEqual(inputs[name].template.prefix, official[name].template.prefix)
        self.assertTrue(inputs["previous_latent"].optional)
        self.assertEqual(inputs["previous_latent"].io_type, "LATENT")
        self.assertNotIn("previous_video", inputs)
        self.assertFalse(inputs["merge_video"].default)
        self.assertFalse(inputs["save_latents"].default)
        self.assertFalse(inputs["dynamic_mask"].default)
        self.assertTrue(schema.is_output_node)
        self.assertEqual([item.is_output_list for item in schema.outputs], [False, False, False, True, True])

    def test_official_seed_modes_and_boundaries(self):
        self.assertEqual(LongVideo.GET_NODE_INFO_V1()["input"]["required"]["seed_mode"][1]["options"],
                         ["fixed", "increment", "decrement", "randomize"])
        maximum = 0xffffffffffffffff
        cases = [("fixed", 100, [100] * 4), ("increment", 100, [100, 101, 102, 103]),
                 ("decrement", 100, [100, 99, 98, 97]),
                 ("increment", maximum - 1, [maximum - 1, maximum, maximum, maximum]),
                 ("decrement", 1, [1, 0, 0, 0]), ("randomize", 100, [100, 0, maximum, 17])]
        for mode, start, expected in cases:
            with self.subTest(mode=mode, start=start), \
                 patch.object(LongVideo, "_render_segment", return_value=(native_latent(), object(), 119)) as render, \
                 patch.object(module.random, "randint", side_effect=[0, maximum, 17]) as randomized:
                output = self.run_node(seed=start, seed_mode=mode)
                self.assertEqual([call.args[10] for call in render.call_args_list], expected)
                self.assertEqual(json.loads(output[2])["seeds"], expected)
                self.assertEqual(randomized.call_count, 3 if mode == "randomize" else 0)

    def mock_pipeline(self, seen, image_refs=None, full_refs=None, refs=None, interrupt_after=None):
        image_refs, full_refs = image_refs if image_refs is not None else [], full_refs if full_refs is not None else []
        refs = refs or {}
        def condition(**kwargs):
            gc.collect()
            self.assertTrue(all(ref() is None for ref in image_refs), "decoded images accumulated")
            self.assertTrue(all(ref() is None for ref in full_refs), "full intermediate LATENT accumulated")
            # Previous files must already exist before the next segment's conditioning starts.
            self.assertEqual(len(list(self.directory.rglob("segment_*.mp4"))), len(seen))
            if interrupt_after is not None and len(seen) == interrupt_after:
                raise RuntimeError("test interrupted")
            seen.append(kwargs)
            for name, value in refs.items():
                self.assertIs(kwargs[name], value)
            latent = native_latent(kwargs["length"], value=0.1 * len(seen))
            full_refs.extend(weakref.ref(stream) for stream in latent["samples"].tensors)
            return io.NodeOutput([[torch.zeros(1, 10, 4), {}]], latent)
        def decode(vae, latent):
            frames = package._av_streams(latent, "sampled")[2]
            images = torch.full((frames, 32, 32, 3), 0.1 * len(seen))
            image_refs.append(weakref.ref(images))
            return (images,)
        def audio(vae, latent):
            frames = package._av_streams(latent, "sampled")[2]
            return io.NodeOutput({"waveform": torch.zeros(1, 2, round(frames / 24 * 48000)), "sample_rate": 48000})
        self.stack.enter_context(patch.object(module.MiniMaxH3ReferenceToVideo, "execute", new=condition))
        self.stack.enter_context(patch.object(module.BasicGuider, "execute",
            new=lambda model, positive: io.NodeOutput(SimpleNamespace(positive=positive))))
        self.stack.enter_context(patch.object(module.SamplerCustomAdvanced, "execute",
            new=lambda noise, guider, sampler, sigmas, latent: io.NodeOutput(latent, latent)))
        self.stack.enter_context(patch.object(module.nodes.VAEDecode, "decode", new=staticmethod(decode)))
        self.stack.enter_context(patch.object(module.VAEDecodeAudio, "execute", new=audio))

    def test_single_segment_without_previous_and_direct_latent_reroll(self):
        seen = []
        self.mock_pipeline(seen)
        for has_previous in (False, True):
            output = self.run_node(1, previous_latent=native_latent() if has_previous else None)
            info = json.loads(output[2])
            self.assertEqual(info["continued_from_latent"], has_previous)
            self.assertEqual(seen[-1]["length"], 141 if has_previous else 124)
            self.assertEqual(info["frames_per_segment"], [119 if has_previous else 124])
            self.assertEqual(len(output[3]), 1)
            self.assertEqual(output[4], [])
            del output

    def test_latent_roundtrip_exact_dtype_values_overhang_and_fingerprint(self):
        path = self.directory / "source.h3latent"
        for dtype in (torch.float16, torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                source = native_latent(124, dtype)
                source["_h3_audio_overhang"] = -1 / 3
                latent_io.save_h3_latent(source, str(path), {"seed": 100})
                loaded = LoadLatent.execute("source.h3latent [output]")[0]
                for original, restored in zip(source["samples"].tensors, loaded["samples"].tensors):
                    self.assertEqual(original.dtype, restored.dtype)
                    self.assertTrue(torch.equal(original, restored))
                self.assertEqual(loaded["_h3_audio_overhang"], source["_h3_audio_overhang"])
                old_hash = LoadLatent.fingerprint_inputs("source.h3latent [output]")
                source["samples"].tensors[0].add_(1)
                latent_io.save_h3_latent(source, str(path), {})
                self.assertNotEqual(old_hash, LoadLatent.fingerprint_inputs("source.h3latent [output]"))

    def test_load_rejects_wrong_format_and_paths_and_generation_rejects_wrong_size(self):
        ordinary = self.directory / "ordinary.latent"
        latent_io.comfy.utils.save_torch_file({"latent_tensor": torch.zeros(1, 4, 2, 2)}, str(ordinary))
        with self.assertRaisesRegex(ValueError, "普通单流"):
            LoadLatent.execute("ordinary.latent [output]")
        for path in ("../escape.h3latent [output]", ""):
            with self.assertRaises(ValueError):
                LoadLatent.execute(path)
        previous = native_latent()
        previous["samples"].tensors[0] = torch.zeros(1, 24, 42, 2, 4)
        with patch.object(LongVideo, "_render_segment") as render:
            with self.assertRaisesRegex(ValueError, "相同分辨率"):
                self.run_node(1, previous_latent=previous)
            render.assert_not_called()
        self.assertEqual(list(self.directory.rglob("segment_*")), [])
        with self.assertRaisesRegex(Exception, "outside the output"):
            self.run_node(1, filename_prefix="../escape/H3")

    def test_streaming_save_retains_only_tail_and_blocked_merge_skips_native_save(self):
        seen, images, full = [], [], []
        self.mock_pipeline(seen, images, full)
        with patch.object(module.InputImpl, "VideoFromList", side_effect=AssertionError("merged while disabled")):
            output = self.run_node()
        self.assertEqual([item["length"] for item in seen], [124, 141, 141, 141])
        info = json.loads(output[2])
        self.assertEqual(info["frames_per_segment"], [124, 119, 119, 119])
        self.assertEqual(info["context_cache_peak_bytes"], (24 * 12 * 2 * 2 + 32 * 2 * 65) * 4)
        self.assertIsInstance(output[0], ExecutionBlocker)
        self.assertEqual(len(list(self.directory.rglob("*.mp4"))), 4)
        self.assertEqual(list(self.directory.rglob("*.h3latent")), [])
        self.assertEqual(len(output.ui["images"]), 1)
        self.assertTrue(all(ref() is None for ref in full[:-2]))
        for ref, stream in zip(full[-2:], output[1]["samples"].tensors):
            self.assertIs(ref(), stream)
        cached, _, _ = execution.get_output_from_returns([output], LongVideo)
        with patch.object(SaveVideo, "execute", side_effect=AssertionError("blocked SaveVideo executed")):
            _, previews, _, _ = asyncio.run(execution.get_output_data("test", "merged", SaveVideo,
                {"video": cached[0], "filename_prefix": ["combined"], "format": ["auto"]}))
        self.assertEqual(previews, {})
        self.assertEqual(len(list(self.directory.rglob("*.mp4"))), 4)

    def test_merge_and_latent_save_export_packets_audio_metadata_and_paths(self):
        seen, images, full = [], [], []
        refs = {"ref_images": {"ref_image_0": object()}, "ref_videos": {"ref_video_0": object()},
                "ref_video_audios": {"ref_video_audio_0": object()}, "ref_audios": {"ref_audio_0": object()}}
        self.mock_pipeline(seen, images, full, refs)
        output = self.run_node(merge_video=True, save_latents=True, **refs)
        info = json.loads(output[2])
        self.assertEqual(info["total_frames"], 481)
        self.assertEqual(len(output[4]), 4)
        self.assertEqual(len(output.ui["latents"]), 1)
        cached, _, _ = execution.get_output_from_returns([output], LongVideo)
        self.assertEqual(len(cached[4]), 4)
        _, previews, has_graph, pending = asyncio.run(execution.get_output_data("test", "merged", SaveVideo,
            {"video": cached[0], "filename_prefix": ["combined/H3"], "format": ["auto"]},
            v3_data={"hidden_inputs": {io.Hidden.prompt: {"test": True},
                                     io.Hidden.extra_pnginfo: {"workflow": {"nodes": []}}}}))
        self.assertFalse(has_graph or pending)
        merged = previews["images"][0]
        paths = [self.directory / path for path in info["video_paths"]]
        paths.append(self.directory / merged["subfolder"] / merged["filename"])
        packets = []
        for path, frames in zip(paths, [124, 119, 119, 119, 481]):
            with av.open(str(path)) as container:
                self.assertEqual(len(container.streams.audio), 1)
                self.assertEqual(container.streams.video[0].frames, frames)
                self.assertIn("workflow", container.metadata)
                packets.append([bytes(packet) for packet in container.demux(container.streams.video[0]) if packet.size])
        self.assertEqual(packets[-1], [packet for part in packets[:-1] for packet in part])
        for index, path in enumerate(output[4]):
            restored = LoadLatent.execute(path)[0]
            self.assertEqual(package._av_streams(restored, "saved")[2], 124 if index == 0 else 141)
            for stream in restored["samples"].tensors:
                self.assertTrue(torch.equal(stream, torch.full_like(stream, .1 * (index + 1))))
        manifest = self.directory / info["manifest"]
        records = [json.loads(line) for line in manifest.read_text("utf-8").splitlines()]
        self.assertEqual([record["seed"] for record in records], [100, 101, 102, 103])
        self.assertEqual(len(list(self.directory.rglob("*.mp4"))), 5)
        self.assertTrue(all(ref() is None for ref in images))

    def test_interrupt_keeps_completed_video_latent_and_seed_record(self):
        self.mock_pipeline([], interrupt_after=1)
        with self.assertRaisesRegex(RuntimeError, "interrupted"):
            self.run_node(save_latents=True)
        self.assertEqual(len(list(self.directory.rglob("*.mp4"))), 1)
        self.assertEqual(len(list(self.directory.rglob("*.h3latent"))), 1)
        records = list(self.directory.rglob("segments.jsonl"))[0].read_text("utf-8").splitlines()
        self.assertEqual(len(records), 1)
        self.assertEqual(json.loads(records[0])["seed"], 100)
        self.assertEqual(list(self.directory.rglob("*.tmp")), [])

    def test_native_main_output_execution_without_external_saver(self):
        self.mock_pipeline([])
        data = {"model": [None], "clip": [None], "vae": [None], "audio_vae": [None],
                "prompts": [{"prompts": ["one"], "repeat_last": False}], "segments": [1],
                "duration": [5.0], "aspect_ratio": ["16:9 (Widescreen)"], "megapixels": [.4],
                "seed": [100], "steps": [8], "sampler_name": ["res_multistep"], "scheduler": ["simple"],
                "sigmas": [torch.tensor([1., 0.])]}
        cached, previews, graph, pending = asyncio.run(execution.get_output_data("test", "main", LongVideo, data,
            v3_data={"hidden_inputs": {io.Hidden.prompt: {"native_execution": True},
                                     io.Hidden.extra_pnginfo: {"workflow": {"nodes": []}}}}))
        self.assertFalse(graph or pending)
        self.assertEqual(len(previews["images"]), 1)
        self.assertIsInstance(cached[0][0], ExecutionBlocker)
        self.assertEqual(len(cached[3]), 1)
        info = json.loads(cached[2][0])
        with av.open(str(self.directory / info["video_paths"][0])) as container:
            self.assertEqual(json.loads(container.metadata["prompt"]), {"native_execution": True})

    def test_dynamic_enabled_single_fresh_segment_uses_normal_path(self):
        self.mock_pipeline([])
        with patch.object(module, "prepare_dynamic_mask", side_effect=AssertionError("no predecessor")), \
             patch.object(module, "check_dynamic_mask_support", side_effect=AssertionError("unneeded patch")):
            output = self.run_node(1, dynamic_mask=True)
        self.assertEqual(json.loads(output[2])["continuation_methods"], ["none"])
        self.assertEqual(json.loads(output[2])["frames_per_segment"], [124])

    def test_dynamic_support_failure_happens_before_sampling_and_file_creation(self):
        with patch.object(LongVideo, "_render_segment") as render:
            with self.assertRaisesRegex(ValueError, "原生H3"):
                self.run_node(4, dynamic_mask=True)
            render.assert_not_called()
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_dynamic_multisegment_path_is_scoped_and_reports_actual_methods(self):
        core = MiniMaxH3.__new__(MiniMaxH3)
        torch.nn.Module.__init__(core)
        core.diffusion_model = torch.nn.Module()
        core.diffusion_model.patch_size = (1, 2, 2)
        patcher = ModelPatcher(core, torch.device("cpu"), torch.device("cpu"), size=1)
        patcher.add_wrapper_with_key(WrappersMP.APPLY_MODEL, "existing_acceleration", lambda executor, *a, **kw: executor(*a, **kw))
        seen, sampled, modes, states = [], [], [], []
        self.mock_pipeline(seen)
        def sample(noise, guider, sampler, sigmas, latent):
            gc.collect()
            self.assertTrue(all(ref() is None for ref in sampled), "dynamic sampling retained full previous LATENT")
            active = guider.model.model_options.get("denoise_mask_function")
            modes.append(active is not None)
            if active is not None:
                states.append(active.__self__)
                self.assertIsNot(guider.model, patcher)
                self.assertEqual(guider.positive[0][1]["minimax_keyframes"], [])
                self.assertIn("existing_acceleration", guider.model.wrappers[WrappersMP.APPLY_MODEL])
                shapes = [tuple(tensor.shape) for tensor in latent["samples"].tensors]
                masks = [comfy.sampler_helpers.prepare_mask(mask, shape, torch.device("cpu"))
                         for mask, shape in zip(latent["noise_mask"].tensors, shapes)]
                packed, _ = comfy.utils.pack_latents(masks)
                for sigma in sigmas[:-1]:
                    live = active(sigma.reshape(1), packed)
                    values = active.__self__.apply_model(lambda **kwargs: kwargs)
                    expected = core._denoise_mask_values(live, shapes)
                    self.assertTrue(all(torch.equal(values[key], tensor) for key, tensor in expected.items()))
            sampled.extend(weakref.ref(tensor) for tensor in latent["samples"].tensors)
            return io.NodeOutput(latent, latent)
        with patch.object(module.BasicGuider, "execute",
                new=lambda model, positive: io.NodeOutput(SimpleNamespace(model=model, positive=positive))), \
             patch.object(module.SamplerCustomAdvanced, "execute", new=sample):
            result = self.run_node(model=patcher, dynamic_mask=True, sigmas=torch.tensor([1., .8, .3, 0.]))
        self.assertEqual(modes, [False, True, True, True])
        self.assertEqual(json.loads(result[2])["continuation_methods"], ["none", "dynamic_mask", "dynamic_mask", "dynamic_mask"])
        self.assertEqual(json.loads(result[2])["frames_per_segment"], [124, 119, 119, 119])
        self.assertNotIn("denoise_mask_function", patcher.model_options)
        self.assertEqual(set(patcher.wrappers[WrappersMP.APPLY_MODEL]), {"existing_acceleration"})
        self.assertEqual(len(list(self.directory.rglob("*.mp4"))), 4)
        self.assertTrue(all(state.current_mask is None for state in states), "loaded model may retain old GPU masks")

    def test_dynamic_sampler_failure_releases_live_mask(self):
        core = MiniMaxH3.__new__(MiniMaxH3)
        torch.nn.Module.__init__(core)
        core.diffusion_model = torch.nn.Module()
        core.diffusion_model.patch_size = (1, 2, 2)
        patcher = ModelPatcher(core, torch.device("cpu"), torch.device("cpu"), size=1)
        self.mock_pipeline([])
        states = []
        def sample(noise, guider, sampler, sigmas, latent):
            active = guider.model.model_options.get("denoise_mask_function")
            if active is None:
                return io.NodeOutput(latent, latent)
            shapes = [tuple(tensor.shape) for tensor in latent["samples"].tensors]
            masks = [comfy.sampler_helpers.prepare_mask(mask, shape, torch.device("cpu"))
                     for mask, shape in zip(latent["noise_mask"].tensors, shapes)]
            packed, _ = comfy.utils.pack_latents(masks)
            active(torch.tensor([1.]), packed)
            states.append(active.__self__)
            raise RuntimeError("dynamic sampler interrupted")
        with patch.object(module.BasicGuider, "execute",
                new=lambda model, positive: io.NodeOutput(SimpleNamespace(model=model, positive=positive))), \
             patch.object(module.SamplerCustomAdvanced, "execute", new=sample):
            with self.assertRaisesRegex(RuntimeError, "dynamic sampler interrupted"):
                self.run_node(model=patcher, dynamic_mask=True)
        self.assertEqual(len(states), 1)
        self.assertIsNone(states[0].current_mask)
        self.assertEqual(len(list(self.directory.rglob("*.mp4"))), 1)

    def test_partial_save_failure_removes_temporary_files(self):
        self.mock_pipeline([])
        def fail_save(path, **kwargs):
            Path(path).write_bytes(b"incomplete")
            raise RuntimeError("encoder failed")
        fake_video = SimpleNamespace(save_to=fail_save)
        with patch.object(module.CreateVideo, "execute", return_value=io.NodeOutput(fake_video)):
            with self.assertRaisesRegex(RuntimeError, "encoder failed"):
                self.run_node(1)
        self.assertEqual(list(self.directory.rglob("*.mp4")), [])
        self.assertEqual(list(self.directory.rglob("*.tmp")), [])
        def fail_latent(tensors, path, **kwargs):
            Path(path).write_bytes(b"incomplete")
            raise RuntimeError("latent save failed")
        with patch.object(latent_io.comfy.utils, "save_torch_file", new=fail_latent):
            with self.assertRaisesRegex(RuntimeError, "latent save failed"):
                latent_io.save_h3_latent(native_latent(), str(self.directory / "broken.h3latent"), {})
        self.assertFalse((self.directory / "broken.h3latent").exists())
        self.assertEqual(list(self.directory.rglob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]], verbosity=2)
