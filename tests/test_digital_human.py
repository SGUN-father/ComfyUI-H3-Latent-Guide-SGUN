import asyncio
import copy
import gc
import importlib.util
import json
import math
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
from comfy_api.latest import InputImpl, Types, io
from comfy_extras.nodes_video import CreateVideo, SaveVideo
import execution
import folder_paths

spec = importlib.util.spec_from_file_location("h3_digital_human_test", Path(__file__).resolve().parents[1] / "__init__.py")
package = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = package
spec.loader.exec_module(package)
module = sys.modules[package.__name__ + ".digital_human"]
DigitalHuman = package.H3DigitalHuman


def av_latent(frames=243, dtype=torch.float32):
    video_t = (frames - 5) // 17 * 5 + 2
    video = torch.arange(24 * video_t * 4, dtype=dtype).reshape(1, 24, video_t, 2, 2)
    audio = torch.full((1, 32, 2, round(frames * 5 / 3)), -9.0, dtype=dtype)
    return {"samples": NestedTensor((video, audio))}


class TestDigitalHumanPlan(unittest.TestCase):
    def check_plan(self, sample_count, sample_rate, seconds):
        plan = module._segment_plan(sample_count, sample_rate, seconds)
        expected_frames = (sample_count * 24 + sample_rate - 1) // sample_rate
        self.assertEqual(sum(part["keep_frames"] for part in plan), expected_frames)
        self.assertTrue(plan)
        position = 0
        for index, part in enumerate(plan):
            with self.subTest(part=index, seconds=seconds, sample_count=sample_count):
                prefix = 0 if index == 0 else 39
                self.assertEqual(part["start_frame"], position)
                self.assertEqual(part["context_frames"], prefix)
                self.assertEqual(part["source_start_frame"], position - prefix)
                self.assertGreaterEqual(part["sample_frames"], 141)
                self.assertLessEqual(part["sample_frames"], 345)
                self.assertEqual((part["sample_frames"] - 39) % 51, 0)
                self.assertGreater(part["keep_frames"], 0)
                self.assertLessEqual(part["keep_frames"] + prefix, part["sample_frames"])
                self.assertEqual(part["source_start_frame"] % 3, 0)
                self.assertEqual(round(part["source_start_frame"] * 5 / 3), part["source_start_frame"] * 5 // 3)
                if index < len(plan) - 1:
                    self.assertEqual(part["keep_frames"] + prefix, part["sample_frames"])
                position += part["keep_frames"]
        return plan

    def test_short_exact_and_long_source_have_complete_bounded_plans(self):
        for rate in (22050, 24000, 32000, 44100, 48000):
            for samples in (1, rate // 10, rate * 21, rate * 121 + 1):
                for seconds in (1.0, 8.0, 10.125, 15.0):
                    self.check_plan(samples, rate, seconds)

    def test_243_frame_window_preserves_absolute_overlap_and_crops_only_final(self):
        plan = self.check_plan(21 * 48000, 48000, 10.125)
        self.assertEqual([part["start_frame"] for part in plan], [0, 243, 447])
        self.assertEqual([part["source_start_frame"] for part in plan], [0, 204, 408])
        self.assertEqual([part["keep_frames"] for part in plan], [243, 204, 57])
        self.assertEqual([part["sample_frames"] for part in plan], [243, 243, 141])

    def test_exact_boundary_creates_no_empty_trailing_segment(self):
        for frames in (243, 447, 651):
            plan = self.check_plan(frames * 2000, 48000, 10.125)
            self.assertEqual(len(plan), 1 + (frames - 243) // 204)
            self.assertEqual(plan[-1]["keep_frames"], 243 if frames == 243 else 204)
        plan = self.check_plan(447 * 2000 + 1, 48000, 10.125)
        self.assertEqual(plan[-1]["keep_frames"], 1)
        self.assertEqual(plan[-1]["sample_frames"], 141)

    def test_different_memory_windows_keep_identical_total_duration(self):
        plans = [self.check_plan(80000 * 37 + 1, 80000, seconds) for seconds in (5.875, 8.0, 14.375)]
        self.assertGreater(len(plans[0]), len(plans[-1]))
        self.assertEqual(len({sum(part["keep_frames"] for part in plan) for plan in plans}), 1)


class TestDigitalHumanAudio(unittest.TestCase):
    def test_absolute_sample_boundaries_preserve_source_without_interior_padding(self):
        for rate in (22050, 24000, 32000, 44100, 48000):
            source = torch.arange(rate * 3, dtype=torch.float64).reshape(1, 1, -1) + 1
            audio = {"sample_rate": rate, "waveform": source}
            for start, frames in ((0, 39), (1, 1), (7, 22), (39, 17)):
                with self.subTest(rate=rate, start=start, frames=frames):
                    result = module._audio_window(audio, start, frames)
                    first, last = round(start * rate / 24), round((start + frames) * rate / 24)
                    self.assertEqual(result["sample_rate"], rate)
                    self.assertEqual(result["waveform"].dtype, source.dtype)
                    self.assertEqual(result["waveform"].device, source.device)
                    self.assertEqual(result["waveform"].shape[-1], last - first)
                    self.assertTrue(torch.equal(result["waveform"], source[..., first:last]))

    def test_window_pads_only_after_real_audio_endpoint(self):
        source = torch.arange(2400, dtype=torch.float32).reshape(1, 2, 1200) + 1
        audio = {"sample_rate": 24000, "waveform": source}
        result = module._audio_window(audio, 1, 2)["waveform"]
        self.assertEqual(result.shape, (1, 2, 2000))
        self.assertTrue(torch.equal(result[..., :200], source[..., 1000:1200]))
        self.assertEqual(torch.count_nonzero(result[..., 200:]).item(), 0)
        after = module._audio_window(audio, 3, 1)["waveform"]
        self.assertEqual(after.shape[-1], 1000)
        self.assertEqual(torch.count_nonzero(after).item(), 0)

    def test_overlap_windows_reproduce_identical_original_samples(self):
        for rate in (22050, 24000, 32000, 44100, 48000):
            source = torch.arange(rate * 21, dtype=torch.float32).reshape(1, 1, -1)
            audio = {"sample_rate": rate, "waveform": source}
            first = module._audio_window(audio, 0, 243)["waveform"]
            second = module._audio_window(audio, 204, 243)["waveform"]
            begin = round(204 * rate / 24)
            end = round(243 * rate / 24)
            self.assertTrue(torch.equal(first[..., begin:end], second[..., :end - begin]))


class TestDigitalHumanLatent(unittest.TestCase):
    def test_audio_is_replaced_and_frozen_including_padding(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            original = av_latent(dtype=dtype)
            input_video, input_audio = original["samples"].tensors
            video_copy, audio_copy = input_video.clone(), input_audio.clone()
            for tokens in (1, input_audio.shape[-1] + 7):
                with self.subTest(dtype=dtype, tokens=tokens):
                    encoded = {"samples": torch.full((1, 32, 2, tokens), 2.0, dtype=dtype)}
                    locked = module._lock_audio(original, encoded)
                    video, audio = locked["samples"].tensors
                    video_mask, audio_mask = locked["noise_mask"].tensors
                    copied = min(tokens, input_audio.shape[-1])
                    self.assertEqual(audio.shape, input_audio.shape)
                    self.assertTrue(torch.all(audio[..., :copied] == 2))
                    self.assertEqual(torch.count_nonzero(audio[..., copied:]).item(), 0)
                    self.assertEqual(torch.count_nonzero(audio_mask).item(), 0)
                    self.assertTrue(torch.all(video_mask == 1))
                    self.assertTrue(torch.equal(video, video_copy))
                    self.assertTrue(torch.equal(input_video, video_copy))
                    self.assertTrue(torch.equal(input_audio, audio_copy))

    def test_continuation_locks_only_copied_video_prefix_and_all_source_audio(self):
        original = av_latent()
        video, audio = original["samples"].tensors
        prefix = torch.full_like(video[:, :, :12], -3)
        encoded = {"samples": torch.full_like(audio, 4)}
        result = module._lock_audio(original, encoded, previous_video=prefix)
        output_video, output_audio = result["samples"].tensors
        video_mask, audio_mask = result["noise_mask"].tensors
        self.assertTrue(torch.equal(output_video[:, :, :12], prefix))
        self.assertTrue(torch.equal(output_video[:, :, 12:], video[:, :, 12:]))
        self.assertEqual(torch.count_nonzero(video_mask[:, :, :12]).item(), 0)
        self.assertTrue(torch.all(video_mask[:, :, 12:] == 1))
        self.assertEqual(torch.count_nonzero(audio_mask).item(), 0)
        self.assertTrue(torch.equal(output_audio, encoded["samples"]))
        output_video[:, :, :12].fill_(123)
        self.assertTrue(torch.all(prefix == -3))
        self.assertFalse(torch.any(video[:, :, :12] == 123))

    def test_tail_is_independent_cpu_storage_and_releases_complete_latent(self):
        original = av_latent()
        video, audio = original["samples"].tensors
        video.requires_grad_(True)
        expected = video[:, :, -12:].detach().clone()
        references = weakref.ref(video), weakref.ref(audio)
        tail = module._video_tail(original)
        self.assertEqual(tail.shape, (1, 24, 12, 2, 2))
        self.assertEqual(tail.device.type, "cpu")
        self.assertFalse(tail.requires_grad)
        self.assertTrue(torch.equal(tail, expected))
        self.assertNotEqual(tail.untyped_storage().data_ptr(), video.untyped_storage().data_ptr())
        self.assertEqual(tail.untyped_storage().nbytes(), tail.numel() * tail.element_size())
        del original, video, audio
        gc.collect()
        self.assertTrue(all(reference() is None for reference in references))
        self.assertTrue(torch.equal(tail, expected))


class TestDigitalHumanNativeVideo(unittest.TestCase):
    def test_node_is_registered_without_changing_original_nodes_or_exposing_latents(self):
        registered = asyncio.run(package.H3LatentGuideExtension().get_node_list())
        for node in (package.MiniMaxH3LatentContinuationGuide, package.H3LoopPromptSchedule,
                     package.H3LongVideo, package.H3VideoPromptPlan, package.H3LoadLatent, DigitalHuman):
            self.assertIn(node, registered)
        schema = DigitalHuman.define_schema()
        inputs = {item.id: item for item in schema.inputs}
        self.assertFalse(schema.hidden)
        self.assertFalse(schema.is_output_node)
        self.assertEqual([output.io_type for output in schema.outputs], ["VIDEO", "STRING", "VIDEO"])
        self.assertTrue(schema.outputs[2].is_output_list)
        for name in ("model", "clip", "vae", "audio_vae", "ref_images", "audio", "prompt", "prompts",
                     "segment_seconds"):
            self.assertIn(name, inputs)
        self.assertTrue(inputs["prompts"].optional)
        self.assertEqual(inputs["prompts"].io_type, "H3_PROMPTS")
        for name in ("image", "previous_latent", "save_latents", "segments", "seed_mode", "filename_prefix", "resume_run", "merge_video"):
            self.assertNotIn(name, inputs)
        official = {item.id: item for item in module.MiniMaxH3ReferenceToVideo.define_schema().inputs}["ref_images"]
        self.assertTrue(inputs["ref_images"].optional)
        self.assertEqual(inputs["ref_images"].template.input.io_type, official.template.input.io_type)
        self.assertEqual(inputs["ref_images"].template.input.id, official.template.input.id)
        for name in ("prefix", "min", "max"):
            self.assertEqual(getattr(inputs["ref_images"].template, name), getattr(official.template, name))

    def test_native_file_concat_uses_single_continuous_soundtrack_without_materializing_video(self):
        rate, counts = 32000, (39, 51, 51)
        samples = round(sum(counts) / 24 * rate)
        time = torch.arange(samples, dtype=torch.float32) / rate
        source = torch.stack((0.2 * torch.sin(2 * math.pi * 375 * time),
                              0.2 * torch.sin(2 * math.pi * 750 * time))).unsqueeze(0)
        with tempfile.TemporaryDirectory() as directory:
            paths, source_packets = [], []
            for index, count in enumerate(counts):
                path = Path(directory) / f"part_{index}.mp4"
                video = CreateVideo.execute(torch.full((count, 32, 32, 3), index * 0.2), 24, audio=None)[0]
                video.save_to(str(path), format=Types.VideoContainer.MP4, codec=Types.VideoCodec.H264)
                with av.open(str(path)) as container:
                    self.assertEqual(len(container.streams.audio), 0)
                    source_packets.extend(bytes(packet) for packet in container.demux(container.streams.video[0]) if packet.size)
                paths.append(InputImpl.VideoFromFile(str(path)))
            soundtrack = {"waveform": source, "sample_rate": rate}
            combined = InputImpl.VideoFromList(paths, complete_audio=soundtrack, codec=Types.VideoCodec.H264)
            output = Path(directory) / "combined.mp4"
            with patch.object(InputImpl.VideoFromFile, "get_components", side_effect=AssertionError("all frames decoded")):
                combined.save_to(str(output), format=Types.VideoContainer.MP4, codec=Types.VideoCodec.AUTO)
            self.assertIs(combined.complete_audio, soundtrack)
            with av.open(str(output)) as container:
                self.assertEqual(container.streams.video[0].frames, sum(counts))
                self.assertEqual(len(container.streams.audio), 1)
                packets = [bytes(packet) for packet in container.demux(container.streams.video[0]) if packet.size]
                self.assertEqual(packets, source_packets)
            with av.open(str(output)) as container:
                stream = container.streams.audio[0]
                self.assertEqual(stream.sample_rate, rate)
                decoded = torch.cat([torch.from_numpy(frame.to_ndarray()) for frame in container.decode(stream)], dim=-1)
            self.assertGreaterEqual(decoded.shape[-1], samples)
            self.assertLess(decoded.shape[-1] - samples, 1024)
            for boundary in (round(counts[0] / 24 * rate), round(sum(counts[:2]) / 24 * rate)):
                actual = decoded[:, boundary - 512:boundary + 512]
                expected = source[0, :, boundary - 512:boundary + 512]
                self.assertGreater(actual.square().mean().item(), 0.015)
                self.assertLess((actual - expected).square().mean().item(), 0.0002)


class TestDigitalHumanExecution(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.directory = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.cache = self.directory / "temp"
        self.cache.mkdir()
        self.output = self.directory / "output"
        self.output.mkdir()
        self.unrelated = self.cache / "other_task.txt"
        self.unrelated.write_text("keep", encoding="utf-8")
        self.stack.enter_context(patch.object(folder_paths, "get_temp_directory", return_value=str(self.cache)))
        self.stack.enter_context(patch.object(folder_paths, "get_output_directory", return_value=str(self.output)))
        self.stack.enter_context(patch.object(module.ResolutionSelector, "execute", return_value=io.NodeOutput(32, 32)))
        self.stack.enter_context(patch.object(SaveVideo, "hidden", SimpleNamespace(prompt=None, extra_pnginfo={})))
        self.model = SimpleNamespace(model=SimpleNamespace(_denoise_mask_values=lambda *args: {}), model_options={})
        self.image = torch.zeros(1, 32, 32, 3)
        self.ref_images = {"ref_image_0": self.image}
        self.source = {"sample_rate": 32000, "waveform": torch.sin(torch.arange(32000 * 23) * 0.1).reshape(1, 1, -1) * 0.2}
        self.sigmas = torch.tensor([1.0, 0.0])
        self.conditions, self.calls, self.image_refs, self.sample_refs, self.encoded_refs = [], [], [], [], []
        self.condition_refs, self.sample_prompts = [], []
        self.locked_prefixes = []

    def run_node(self, **kwargs):
        options = dict(segment_seconds=10.125, seed=100, sigmas=self.sigmas,
                       start_time=1.25, duration=21.0, ref_images=self.ref_images)
        options.update(kwargs)
        output_before = set(self.output.rglob("*"))
        result = DigitalHuman.execute(
            self.model, None, None, SimpleNamespace(audio_sample_rate=32000),
            self.source, "one continuous performance", **options)
        self.assertIsNone(result.ui, "digital-human node must not emit any built-in preview")
        self.assertEqual(len(result.result), 3)
        self.assertEqual(set(self.output.rglob("*")), output_before,
                         "digital-human node created formal output before an external SaveVideo")
        return result

    def cached_parts(self):
        return sorted(self.cache.rglob("segment_*.mp4"))

    def save_external(self, video, prefix="external/final"):
        before = set(self.output.rglob("*.mp4"))
        result = SaveVideo.execute(video, prefix, format="mp4", codec={"codec": "h264"})
        self.assertIs(result[0], video)
        saved = set(self.output.rglob("*.mp4")) - before
        self.assertEqual(len(saved), 1, "official SaveVideo must write exactly one complete video")
        return saved.pop()

    def assert_cached_parts(self, counts):
        paths = self.cached_parts()
        self.assertEqual(len(paths), len(counts))
        parents = {path.parent for path in paths}
        self.assertEqual(len(parents), 1)
        directory = parents.pop()
        self.assertEqual(directory.parent, self.cache)
        self.assertTrue(directory.name.startswith("h3_digital_human_"))
        self.assertEqual(set(directory.iterdir()), set(paths), "temporary cache contains backups or a run manifest")
        for path, frames in zip(paths, counts):
            with self.subTest(path=path.name), av.open(str(path)) as container:
                self.assertEqual(container.streams.video[0].frames, frames)
                self.assertEqual(len(container.streams.audio), 1)
                self.assertEqual(container.streams.audio[0].sample_rate, self.source["sample_rate"])
        for root in (self.output, self.cache):
            for pattern in ("*.latent", "*.safetensors", "*.tmp", "run.json", "complete.mp4"):
                self.assertEqual(list(root.rglob(pattern)), [])
        return paths

    def mock_pipeline(self, fail_at=None, failure=None, native_inputs=False):
        def reference(**kwargs):
            self.conditions.append(kwargs)
            if native_inputs:
                self.assertEqual(list(kwargs["ref_images"]), list(self.ref_images))
                for key, image in self.ref_images.items():
                    self.assertIs(kwargs["ref_images"][key], image)
            else:
                self.assertIs(kwargs["ref_images"], self.ref_images)
            self.assertNotIn("ref_audios", kwargs)
            tensor = torch.zeros(1, 10, 4)
            self.condition_refs.append(weakref.ref(tensor))
            positive = [[tensor, {"test_prompt": kwargs["prompt"]}]]
            return io.NodeOutput(positive, av_latent(kwargs["length"]))
        def encode(vae, audio):
            samples = audio["waveform"].shape[-1]
            encoded = torch.ones(1, 32, 2, round(samples / audio["sample_rate"] * 40))
            self.encoded_refs.append(weakref.ref(encoded))
            return io.NodeOutput({"samples": encoded})
        def sample(noise, guider, sampler, sigmas, latent):
            gc.collect()
            self.assertTrue(all(reference() is None for reference in self.image_refs), "decoded frames survived into next segment")
            self.assertTrue(all(reference() is None for reference in self.sample_refs), "complete sampled LATENT survived into next segment")
            self.assertTrue(all(reference() is None for reference in self.condition_refs[:-1]),
                            "old text conditioning accumulated across prompt changes")
            self.assertIs(sigmas, self.sigmas)
            index = len(self.calls)
            if index == fail_at:
                if failure is not None:
                    raise failure
                raise module.comfy.model_management.InterruptProcessingException()
            video, audio = latent["samples"].tensors
            video_mask, audio_mask = latent["noise_mask"].tensors
            self.assertEqual(torch.count_nonzero(audio_mask).item(), 0)
            has_context = bool(torch.any(video_mask == 0))
            if has_context:
                self.assertEqual(torch.count_nonzero(video_mask[:, :, :12]).item(), 0)
                self.assertTrue(torch.all(video_mask[:, :, 12:] == 1))
            else:
                self.assertTrue(torch.all(video_mask == 1))
            self.calls.append((noise.seed, video.shape[2], audio.shape[-1]))
            self.locked_prefixes.append(has_context)
            self.sample_prompts.append(guider.positive[0][1]["test_prompt"])
            sampled_video, sampled_audio = video.clone(), audio.clone()
            self.sample_refs.extend((weakref.ref(sampled_video), weakref.ref(sampled_audio)))
            sampled = {"samples": NestedTensor((sampled_video, sampled_audio))}
            return io.NodeOutput(sampled, sampled)
        def decode(vae, latent, *args, **kwargs):
            frames = (latent["samples"].tensors[0].shape[2] - 2) // 5 * 17 + 5
            images = torch.full((frames, 32, 32, 3), len(self.calls) * 0.1)
            self.image_refs.append(weakref.ref(images))
            return (images,)
        self.stack.enter_context(patch.object(module.MiniMaxH3ReferenceToVideo, "execute", new=reference))
        self.stack.enter_context(patch.object(module.VAEEncodeAudio, "execute", new=encode))
        self.stack.enter_context(patch.object(module.BasicGuider, "execute", new=lambda model, positive: io.NodeOutput(SimpleNamespace(positive=positive))))
        self.stack.enter_context(patch.object(module.BasicScheduler, "execute", side_effect=AssertionError("custom sigmas ignored")))
        self.stack.enter_context(patch.object(module.SamplerCustomAdvanced, "execute", new=sample))
        self.stack.enter_context(patch.object(module.nodes.VAEDecodeTiled, "decode", new=staticmethod(decode)))
        self.stack.enter_context(patch.object(module.nodes.VAEDecode, "decode", new=staticmethod(decode)))

    def test_execute_reuses_reference_releases_frames_and_only_caches_original_audio_parts_in_temp(self):
        self.mock_pipeline()
        source_refs = []
        concatenate = module.InputImpl.VideoFromList
        def combined(videos, complete_audio, codec):
            if complete_audio["waveform"].shape[-1] == 21 * 32000:
                self.assertTrue(torch.equal(complete_audio["waveform"], self.source["waveform"][..., 40000:712000]))
                source_refs.append(weakref.ref(complete_audio["waveform"]))
            return concatenate(videos, complete_audio=complete_audio, codec=codec)
        with patch.object(module.InputImpl, "VideoFromList", side_effect=combined), \
             patch.object(concatenate, "save_to", side_effect=AssertionError("node saved the complete video internally")), \
             patch.object(concatenate, "get_stream_source", side_effect=AssertionError("node materialized a complete-video buffer")):
            result = self.run_node()
        self.assertEqual(len(self.conditions), 1)
        self.assertEqual(self.calls, [(100, 72, 405), (101, 72, 405), (102, 42, 235)])
        self.assertEqual(self.locked_prefixes, [False, True, True])
        info = json.loads(result[1])
        self.assertEqual(info["total_frames"], 504)
        self.assertEqual(info["segments"], 3)
        self.assertEqual(info["sample_frames"], [243, 243, 141])
        parts = self.assert_cached_parts([243, 204, 57])
        self.assertEqual([Path(video.get_stream_source()) for video in result[2]], parts)
        start_frame = 0
        for part, count in zip(parts, (243, 204, 57)):
            with av.open(str(part)) as container:
                decoded = torch.cat([torch.from_numpy(frame.to_ndarray())
                                     for frame in container.decode(container.streams.audio[0])], dim=-1)
            first = 40000 + round(start_frame * 32000 / 24)
            last = 40000 + round((start_frame + count) * 32000 / 24)
            expected = self.source["waveform"][0, :, first:last]
            self.assertGreaterEqual(decoded.shape[-1], expected.shape[-1])
            self.assertLess(decoded.shape[-1] - expected.shape[-1], 1024)
            self.assertLess((decoded[:, :expected.shape[-1]] - expected).square().mean().item(), 0.0002,
                            "saved segment does not contain its own absolute source-audio interval")
            start_frame += count
        self.assertEqual(set(self.cache.rglob("*.mp4")), set(parts))
        self.assertIsInstance(result[0], concatenate)
        self.assertEqual(result[0].get_frame_count(), 504)
        self.assertEqual([Path(video.get_stream_source()) for video in result[0].videos], parts)
        self.assertEqual(list(self.output.iterdir()), [])
        gc.collect()
        self.assertTrue(all(reference() is None for reference in self.image_refs + self.sample_refs + self.encoded_refs + self.condition_refs))
        self.assertEqual(len(source_refs), 1)
        self.assertIs(source_refs[0](), result[0].complete_audio["waveform"])
        del result
        gc.collect()
        self.assertIsNone(source_refs[0](), "discarding the merged output retained its original soundtrack")
        self.assertEqual(self.unrelated.read_text(encoding="utf-8"), "keep")

    def test_official_save_video_writes_one_full_video_with_zero_based_timeline_and_continuous_original_audio(self):
        self.mock_pipeline()
        result = self.run_node()
        self.assert_cached_parts([243, 204, 57])
        self.assertEqual(list(self.output.iterdir()), [])
        with patch.object(result[0], "save_to", wraps=result[0].save_to) as save:
            path = self.save_external(result[0])
        save.assert_called_once()
        self.assertEqual(list(self.output.rglob("complete.mp4")), [])
        self.assertEqual(set(self.output.rglob("*.mp4")), {path})
        self.assertEqual(list(self.output.rglob("run.json")), [])
        with av.open(str(path)) as container:
            video, audio = container.streams.video[0], container.streams.audio[0]
            self.assertEqual(len(container.streams.video), 1)
            self.assertEqual(len(container.streams.audio), 1)
            self.assertEqual(video.frames, 504)
            self.assertEqual(video.start_time * video.time_base, 0)
            self.assertEqual(audio.start_time * audio.time_base, 0)
            frame_step = 1 / video.average_rate
            times = [frame.pts * frame.time_base for frame in container.decode(video)]
        self.assertEqual(times[0], 0)
        self.assertTrue(all(second - first == frame_step for first, second in zip(times, times[1:])))
        self.assertAlmostEqual(float(times[-1]), 503 / 24)
        with av.open(str(path)) as container:
            decoded = torch.cat([torch.from_numpy(frame.to_ndarray())
                                 for frame in container.decode(container.streams.audio[0])], dim=-1)
        expected = self.source["waveform"][0, :, 40000:712000]
        self.assertGreaterEqual(decoded.shape[-1], expected.shape[-1])
        self.assertLess(decoded.shape[-1] - expected.shape[-1], 1024)
        self.assertLess((decoded[:, :expected.shape[-1]] - expected).square().mean().item(), 0.0002)
        for boundary in (324000, 596000):
            actual = decoded[:, boundary - 512:boundary + 512]
            target = expected[:, boundary - 512:boundary + 512]
            self.assertGreater(actual.square().mean().item(), 0.015)
            self.assertLess((actual - target).square().mean().item(), 0.0002)

    def test_two_references_keep_order_and_original_dictionary_with_one_encode_for_all_segments(self):
        second_image = torch.ones(1, 48, 32, 3)
        self.ref_images = {"ref_image_0": self.image, "ref_image_1": second_image}
        self.mock_pipeline()
        result = self.run_node()
        self.assertEqual(json.loads(result[1])["segments"], 3)
        self.assertEqual(len(self.conditions), 1)
        references = self.conditions[0]["ref_images"]
        self.assertIs(references, self.ref_images)
        self.assertEqual(list(references), ["ref_image_0", "ref_image_1"])
        self.assertIs(list(references.values())[0], self.image)
        self.assertIs(list(references.values())[1], second_image)
        self.assertEqual(len(self.calls), 3)

    def test_no_reference_images_is_passed_to_official_conditioning_unchanged(self):
        self.ref_images = None
        self.mock_pipeline()
        result = self.run_node()
        self.assertEqual(json.loads(result[1])["segments"], 3)
        self.assertEqual(len(self.conditions), 1)
        self.assertIsNone(self.conditions[0]["ref_images"])

    def test_real_interruption_removes_own_completed_cache_and_preserves_unrelated_files(self):
        self.mock_pipeline(fail_at=1)
        with self.assertRaises(module.comfy.model_management.InterruptProcessingException):
            self.run_node()
        self.assertEqual(list(self.cache.iterdir()), [self.unrelated])
        self.assertEqual(list(self.output.iterdir()), [])
        self.assertEqual(len(self.calls), 1)

    def test_generation_failure_removes_own_completed_cache_and_preserves_unrelated_files(self):
        self.mock_pipeline(fail_at=1, failure=OSError("simulated generation failure"))
        with self.assertRaisesRegex(OSError, "simulated generation failure"):
            self.run_node()
        self.assertEqual(list(self.cache.iterdir()), [self.unrelated])
        self.assertEqual(list(self.output.iterdir()), [])
        self.assertEqual(len(self.calls), 1)

    def test_external_save_failure_keeps_parts_and_can_retry_existing_output_without_sampling(self):
        self.mock_pipeline()
        result = self.run_node()
        parts = {path: path.read_bytes() for path in self.assert_cached_parts([243, 204, 57])}
        with patch.object(result[0], "save_to", side_effect=OSError("simulated disk full")):
            with self.assertRaisesRegex(OSError, "simulated disk full"):
                self.save_external(result[0])
        self.assertEqual(set(self.cache.rglob("*.mp4")), set(parts))
        self.assertEqual(list(self.output.rglob("complete.mp4")), [])
        with patch.object(module.MiniMaxH3ReferenceToVideo, "execute", side_effect=AssertionError("save retry encoded conditions")), \
             patch.object(module.SamplerCustomAdvanced, "execute", side_effect=AssertionError("save retry sampled")):
            path = self.save_external(result[0])
        self.assertEqual(len(self.calls), 3)
        for part, content in parts.items():
            self.assertEqual(part.read_bytes(), content)
        with av.open(str(path)) as container:
            self.assertEqual(container.streams.video[0].frames, 504)
            self.assertEqual(len(container.streams.audio), 1)

    def test_segment_prompts_override_legacy_prompt_and_follow_plan_order(self):
        self.mock_pipeline()
        plan = package.H3VideoPromptPlan.execute("same singer", "first action\n---\nsecond action\n---\nfinal action")[0]
        self.run_node(prompts=plan)
        self.assertEqual(self.sample_prompts, ["same singer\n\nfirst action", "same singer\n\nsecond action",
                                              "same singer\n\nfinal action"])
        self.assertEqual([condition["prompt"] for condition in self.conditions], self.sample_prompts)
        self.assert_cached_parts([243, 204, 57])

    def test_repeated_last_prompt_reuses_current_conditioning_without_accumulating_old_conditions(self):
        self.mock_pipeline()
        self.run_node(prompts={"prompts": ["first performance", "second performance"], "repeat_last": True})
        self.assertEqual(self.sample_prompts, ["first performance", "second performance", "second performance"])
        self.assertEqual([condition["prompt"] for condition in self.conditions], ["first performance", "second performance"])
        gc.collect()
        self.assertTrue(all(reference() is None for reference in self.condition_refs))

    def test_insufficient_prompt_plan_is_rejected_before_encoding_sampling_or_creating_cache(self):
        with patch.object(module.MiniMaxH3ReferenceToVideo, "execute", side_effect=AssertionError("prompt encoded before validation")), \
             patch.object(module.SamplerCustomAdvanced, "execute", side_effect=AssertionError("sampled before validation")):
            with self.assertRaisesRegex(ValueError, "提示词|prompt"):
                self.run_node(prompts={"prompts": ["only one of three"], "repeat_last": False})
        self.assertEqual(list(self.output.iterdir()), [])
        self.assertEqual(list(self.cache.iterdir()), [self.unrelated])

    def test_native_executor_reuses_generated_cache_when_full_saver_is_added_after_part_saver(self):
        self.mock_pipeline(native_inputs=True)
        owner = self

        class TestInputs:
            RETURN_TYPES = ("MODEL", "CLIP", "VAE", "VAE", "AUDIO", "SIGMAS", "IMAGE")
            FUNCTION = "load"

            @classmethod
            def INPUT_TYPES(cls):
                return {"required": {}}

            @classmethod
            def load(cls):
                return (owner.model, None, None, SimpleNamespace(audio_sample_rate=32000),
                        owner.source, owner.sigmas, owner.image)

        prompt = {
            "inputs": {"class_type": "H3DigitalHumanTestInputs", "inputs": {}},
            "human": {"class_type": "H3DigitalHuman", "inputs": {
                "model": ["inputs", 0], "clip": ["inputs", 1], "vae": ["inputs", 2],
                "audio_vae": ["inputs", 3], "audio": ["inputs", 4], "sigmas": ["inputs", 5],
                "ref_images.ref_image_0": ["inputs", 6], "prompt": "one continuous performance",
                "segment_seconds": 10.125, "aspect_ratio": "9:16 (Portrait Widescreen)",
                "megapixels": 0.4, "seed": 100, "steps": 8, "sampler_name": "er_sde",
                "scheduler": "beta", "start_time": 1.25, "duration": 21.0}},
            "parts": {"class_type": "SaveVideo", "inputs": {
                "video": ["human", 2], "filename_prefix": "native/part", "format": "mp4", "codec": "h264"}},
        }
        server = SimpleNamespace(client_id=None, last_node_id=None, send_sync=lambda *args, **kwargs: None)
        executor = execution.PromptExecutor(server, cache_args={"ram": 0, "ram_inactive": 0})
        with patch.dict(module.nodes.NODE_CLASS_MAPPINGS, {
                "H3DigitalHumanTestInputs": TestInputs, "H3DigitalHuman": DigitalHuman, "SaveVideo": SaveVideo}):
            executor.execute(copy.deepcopy(prompt), "digital-human-parts", execute_outputs=["parts"])
            self.assertTrue(executor.success, executor.status_messages)
            part_files = set(self.output.rglob("*.mp4"))
            self.assertEqual(len(part_files), 3)
            self.assertEqual(len(self.calls), 3)
            cache_paths = self.assert_cached_parts([243, 204, 57])
            prompt["full"] = {"class_type": "SaveVideo", "inputs": {
                "video": ["human", 0], "filename_prefix": "native/full", "format": "mp4", "codec": "h264"}}
            executor.execute(copy.deepcopy(prompt), "digital-human-full", execute_outputs=["parts", "full"])
        self.assertTrue(executor.success, executor.status_messages)
        self.assertEqual(len(self.calls), 3, "adding a downstream saver sampled the generated video again")
        self.assertEqual(len(self.conditions), 1)
        self.assertEqual(self.cached_parts(), cache_paths)
        saved = set(self.output.rglob("*.mp4"))
        self.assertEqual(len(saved), 4, "cached part saver wrote duplicate files")
        full_files = saved - part_files
        self.assertEqual(len(full_files), 1)
        with av.open(str(full_files.pop())) as container:
            self.assertEqual(container.streams.video[0].frames, 504)
            self.assertEqual(len(container.streams.audio), 1)

    def test_official_save_video_maps_segment_list_to_one_original_audio_file_per_part(self):
        self.mock_pipeline()
        result = self.run_node()
        cached = self.assert_cached_parts([243, 204, 57])
        self.assertEqual(list(self.output.iterdir()), [])
        full_video = self.save_external(result[0])
        # The ComfyUI executor maps a list output to separate calls of the ordinary SaveVideo node.
        saved = [self.save_external(video, "external/part") for video in result[2]]
        self.assertEqual(len(saved), 3)
        self.assertEqual(set(self.output.rglob("*.mp4")), set(saved) | {full_video})
        self.assertEqual(set(self.cache.rglob("*.mp4")), set(cached))
        with av.open(str(full_video)) as container:
            self.assertEqual(container.streams.video[0].frames, 504)
            self.assertEqual(len(container.streams.audio), 1)
        for pattern in ("run.json", "complete.mp4", "*.latent", "*.safetensors", "*.tmp"):
            self.assertEqual(list(self.output.rglob(pattern)), [])
        start_frame = 0
        for path, count in zip(saved, (243, 204, 57)):
            with self.subTest(path=path.name), av.open(str(path)) as container:
                self.assertEqual(container.streams.video[0].frames, count)
                self.assertEqual(len(container.streams.audio), 1)
                self.assertEqual(container.streams.video[0].start_time, 0)
                self.assertEqual(container.streams.audio[0].start_time, 0)
                decoded = torch.cat([torch.from_numpy(frame.to_ndarray())
                                     for frame in container.decode(container.streams.audio[0])], dim=-1)
            first = 40000 + round(start_frame * 32000 / 24)
            last = 40000 + round((start_frame + count) * 32000 / 24)
            expected = self.source["waveform"][0, :, first:last]
            self.assertGreaterEqual(decoded.shape[-1], expected.shape[-1])
            self.assertLess(decoded.shape[-1] - expected.shape[-1], 1024)
            self.assertLess((decoded[:, :expected.shape[-1]] - expected).square().mean().item(), 0.0002)
            start_frame += count

    def test_explicit_frame_aligned_duration_avoids_extra_frame_at_common_sample_rates(self):
        self.mock_pipeline()
        for rate in (24000, 32000, 44100, 48000):
            with self.subTest(sample_rate=rate):
                self.source = {"sample_rate": rate, "waveform": torch.zeros(1, 1, rate * 5)}
                result = self.run_node(start_time=0, duration=104 / 24)
                self.assertEqual(json.loads(result[1])["total_frames"], 104)
                self.assertEqual(result[0].get_frame_count(), 104)

    def test_full_original_audio_uses_real_sample_count_and_never_truncates_fractional_last_frame(self):
        self.mock_pipeline()
        for rate in (24000, 32000, 44100, 48000):
            with self.subTest(sample_rate=rate):
                samples = math.floor(104 * rate / 24) + 1
                self.source = {"sample_rate": rate, "waveform": torch.zeros(1, 1, samples)}
                result = self.run_node(start_time=0, duration=0)
                expected = (samples * 24 + rate - 1) // rate
                self.assertEqual(json.loads(result[1])["total_frames"], expected)
                self.assertEqual(result[0].get_frame_count(), expected)

    def test_existing_dynamic_mask_is_rejected_before_any_work_or_file_creation(self):
        self.model.model_options["denoise_mask_function"] = lambda *args: None
        with patch.object(module.MiniMaxH3ReferenceToVideo, "execute", side_effect=AssertionError("reference encoded before rejection")):
            with self.assertRaisesRegex(ValueError, "遮罩"):
                self.run_node()
        self.assertEqual(list(self.cache.iterdir()), [self.unrelated])


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]], verbosity=2)
