import asyncio
import hashlib
import json
import shutil
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import av
import torch

import test_long_video as legacy


module, package = legacy.module, legacy.package
review = sys.modules[package.__name__ + ".segment_review"]
seam_preview = sys.modules[package.__name__ + ".seam_preview"]


class TestSegmentReview(unittest.TestCase):
    setUp = legacy.TestLongVideo.setUp
    run_node = legacy.TestLongVideo.run_node
    mock_pipeline = legacy.TestLongVideo.mock_pipeline

    def records(self, output):
        return [review.load_record(item["recipe_path"]) for item in output.ui["h3_sgun_segments"]]

    def test_recipes_store_full_prompts_and_actual_random_seeds_without_full_latents(self):
        self.mock_pipeline([])
        prompts = legacy.PromptPlan.execute("same actor", "first action\n---\nsecond action\n---\nthird action")[0]
        with patch.object(module.random, "randint", side_effect=[7, 91]):
            output = self.run_node(3, prompts, seed=42, seed_mode="randomize")
        records = self.records(output)
        self.assertEqual([record["seed"] for record in records], [42, 7, 91])
        self.assertEqual([record["prompt"] for record in records], prompts["prompts"])
        self.assertEqual(list(self.directory.rglob("*.h3latent")), [])
        self.assertEqual(list(self.cache_directory.rglob("*.h3latent")), [])
        self.assertIsNone(records[0]["context"])
        self.assertIsNone(records[0]["diagnostics"])
        self.assertEqual(len(list(self.directory.rglob("*.h3context"))), 2)
        self.assertEqual(list(self.cache_directory.iterdir()), [])
        for record in records[1:]:
            context = review.load_retry_context(record)
            self.assertEqual(package._av_streams(context, "saved retry context")[2], 39)
            self.assertTrue(all(stream.device.type == "cpu" for stream in context["samples"].tensors))
            self.assertLess(sum(stream.numel() for stream in context["samples"].tensors),
                            sum(stream.numel() for stream in output[1]["samples"].tensors))

    def test_retry_loads_the_selected_predecessor_and_preserves_custom_sigma_dtype(self):
        self.mock_pipeline([])
        sigmas = torch.tensor([1., .75, .5, 0.], dtype=torch.float64)
        prompts = {"prompts": ["first", "second", "third"], "repeat_last": False}
        output = self.run_node(3, prompts, sigmas=sigmas, seed=25)
        entry = output.ui["h3_sgun_segments"][2]
        record = review.load_record(entry["recipe_path"])
        context = review.load_retry_context(record)
        self.assertEqual(record["sigmas_source"], "external")
        self.assertEqual(record["sigmas_dtype"], "float64")
        self.assertEqual(record["sigmas"], sigmas.tolist())
        self.assertEqual(record["settings"]["context_frames"], 22)
        with patch.object(legacy.LongVideo, "_render_segment", return_value=(
                legacy.native_latent(), module.InputImpl.VideoFromFile(
                    legacy.folder_paths.get_annotated_filepath(record["video_path"])), 119)) as render, \
             patch.object(module.BasicScheduler, "execute", side_effect=AssertionError("external sigmas recalculated")):
            retry = self.run_node(7, seed=999, retry_segment=entry["recipe_path"], retry_prompt="replacement action",
                                  retry_seed_mode="custom", retry_seed="999")
        self.assertEqual(render.call_count, 1)
        call = render.call_args
        self.assertEqual(call.args[4], "replacement action")
        self.assertEqual(call.args[10], 999)
        self.assertEqual(call.args[12].dtype, sigmas.dtype)
        self.assertTrue(torch.equal(call.args[12], sigmas))
        for actual, expected in zip(call.args[8]["samples"].tensors, context["samples"].tensors):
            self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(json.loads(retry[2])["segments"], 1)
        self.assertEqual(len(json.loads(retry[2])["video_paths"]), 1)

    def test_data_folder_records_context_and_real_seam_survive_retry(self):
        self.mock_pipeline([])
        with patch.object(module, "build_seam_preview", wraps=seam_preview.build_seam_preview):
            output = self.run_node(2, duration=1.0)
        info = json.loads(output[2])
        run_folder = self.directory / info["output_subfolder"]
        data_folder = run_folder / "_data"
        self.assertEqual({path.name for path in run_folder.iterdir()},
                         {"segment_0001.mp4", "segment_0002.mp4", "_data"})
        self.assertEqual({path.name for path in data_folder.iterdir()}, {
            "segment_0001.h3segment.json", "segment_0002.h3segment.json", "segment_0002.h3context",
            "seam_0002.mp4", "context_preview_0002.mp4"})
        entry = output.ui["h3_sgun_segments"][1]
        record = review.load_record(entry["recipe_path"])
        for key in ("context", "seam_path", "context_video_path"):
            path = Path(legacy.folder_paths.get_annotated_filepath(record[key]))
            self.assertEqual(path.parent, data_folder)
            self.assertTrue(path.is_file())
        self.assertTrue(entry["context_available"])
        self.assertEqual(entry["video"]["subfolder"], info["output_subfolder"])
        self.assertEqual(entry["seam"]["subfolder"], f"{info['output_subfolder']}/_data")
        for key, frames in (("seam_path", 58), ("context_video_path", 24)):
            with av.open(legacy.folder_paths.get_annotated_filepath(record[key])) as container:
                self.assertEqual(container.streams.video[0].frames, frames)
                self.assertEqual(len(container.streams.audio), 1)
        context = review.load_retry_context(record)
        self.assertEqual(package._av_streams(context, "retry context")[2], 39)
        self.assertIn(entry["recipe_path"], [item["recipe_path"] for item in review.list_segment_records()])
        self.mock_pipeline([])
        with patch.object(module, "build_seam_preview", wraps=seam_preview.build_seam_preview) as build:
            retry = self.run_node(1, duration=1.0, retry_segment=entry["recipe_path"], retry_seed_mode="original")
        self.assertEqual(build.call_args.args[0], legacy.folder_paths.get_annotated_filepath(record["context_video_path"]))
        retried = self.records(retry)[0]
        self.assertEqual(retried["segment"], 2)
        self.assertNotEqual(retried["diagnostics"]["status"], "error")
        self.assertTrue(Path(legacy.folder_paths.get_annotated_filepath(retried["seam_path"])).is_file())
        self.assertEqual(Path(legacy.folder_paths.get_annotated_filepath(retried["context"])).parent.name, "_data")
        self.assertEqual(json.loads(retry[2])["output_subfolder"], info["output_subfolder"])
        self.assertEqual(entry["video"]["subfolder"], retry.ui["h3_sgun_segments"][0]["video"]["subfolder"])

    def test_existing_flat_records_remain_discoverable_and_retry_into_data_folder(self):
        self.mock_pipeline([])
        with patch.object(module, "build_seam_preview", wraps=seam_preview.build_seam_preview):
            output = self.run_node(2, duration=1.0)
        entry = output.ui["h3_sgun_segments"][1]
        record = review.load_record(entry["recipe_path"])
        old_folder = self.directory / "old_flat"
        old_folder.mkdir()
        for key in ("video_path", "context", "seam_path", "context_video_path"):
            source = Path(legacy.folder_paths.get_annotated_filepath(record[key]))
            destination = old_folder / source.name
            shutil.copy2(source, destination)
            record[key] = review.annotated_path(str(destination), "output")
        recipe = old_folder / "segment_0002.h3segment.json"
        review.save_record(record, str(recipe))
        recipe_name = review.annotated_path(str(recipe), "output")
        loaded = review.load_record(recipe_name)
        self.assertEqual(loaded, record)
        self.assertEqual(package._av_streams(review.load_retry_context(loaded), "legacy context")[2], 39)
        listed = next(item for item in review.list_segment_records() if item["recipe_path"] == recipe_name)
        self.assertTrue(listed["context_available"])
        self.assertEqual(listed["video"]["subfolder"], "old_flat")
        self.assertEqual(listed["seam"]["subfolder"], "old_flat")
        self.mock_pipeline([])
        with patch.object(module, "build_seam_preview", wraps=seam_preview.build_seam_preview) as build:
            retry = self.run_node(1, duration=1.0, retry_segment=recipe_name, retry_seed_mode="original")
        self.assertEqual(build.call_args.args[0], str(old_folder / "context_preview_0002.mp4"))
        retried_entry = retry.ui["h3_sgun_segments"][0]
        self.assertEqual(json.loads(retry[2])["output_subfolder"], "old_flat")
        self.assertEqual(Path(legacy.folder_paths.get_annotated_filepath(retried_entry["recipe_path"])).parent.name, "_data")
        self.assertNotEqual(self.records(retry)[0]["diagnostics"]["status"], "error")
        self.assertTrue(recipe.is_file())

    def test_retries_keep_original_batch_and_number_versions_by_logical_segment(self):
        self.mock_pipeline([])
        with patch.object(module, "build_seam_preview", wraps=seam_preview.build_seam_preview):
            original = self.run_node(3, duration=1.0, save_latents=True)
        original_info = json.loads(original[2])
        run_folder = self.directory / original_info["output_subfolder"]
        data_folder = run_folder / "_data"
        original_hashes = {path: hashlib.sha256(path.read_bytes()).hexdigest()
                           for path in run_folder.rglob("*") if path.is_file() and path.name != "segments.jsonl"}
        manifest_path = self.directory / original_info["manifest"]
        original_manifest = manifest_path.read_bytes()
        segment_two = original.ui["h3_sgun_segments"][1]["recipe_path"]
        segment_three = original.ui["h3_sgun_segments"][2]["recipe_path"]
        sources = [segment_two, segment_two, None, segment_three]
        expected_stems = ["segment_0002_retry_001", "segment_0002_retry_002",
                          "segment_0002_retry_003", "segment_0003_retry_001"]
        recipes = list(original_info["segment_records"])
        hashes = dict(original_hashes)
        retry_outputs = []
        for index, stem in enumerate(expected_stems):
            source = sources[index] if sources[index] is not None else retry_outputs[0].ui["h3_sgun_segments"][0]["recipe_path"]
            source_record = review.load_record(source)
            expected_context = review.load_retry_context(source_record)
            self.mock_pipeline([])
            with patch.object(module, "build_seam_preview", wraps=seam_preview.build_seam_preview) as build:
                result = self.run_node(1, duration=1.0, retry_segment=source, save_latents=True,
                                       retry_seed_mode="custom", retry_seed=str(800 + index),
                                       filename_prefix="different/unwanted/new_batch")
            retry_outputs.append(result)
            info = json.loads(result[2])
            self.assertEqual(info["output_subfolder"], original_info["output_subfolder"])
            video_path = Path(legacy.folder_paths.get_annotated_filepath(info["video_paths"][0]))
            self.assertEqual(video_path, run_folder / f"{stem}.mp4")
            with av.open(str(video_path)) as container:
                self.assertEqual(container.streams.video[0].frames, 34)
                self.assertEqual(len(container.streams.audio), 1)
            entry = result.ui["h3_sgun_segments"][0]
            record = review.load_record(entry["recipe_path"])
            self.assertEqual(record["segment"], source_record["segment"])
            self.assertEqual(record["seed"], 800 + index)
            self.assertEqual(Path(legacy.folder_paths.get_annotated_filepath(entry["recipe_path"])),
                             data_folder / f"{stem}.h3segment.json")
            self.assertEqual(Path(legacy.folder_paths.get_annotated_filepath(record["context"])),
                             data_folder / f"{stem}.h3context")
            for actual, expected in zip(review.load_retry_context(record)["samples"].tensors,
                                        expected_context["samples"].tensors):
                self.assertTrue(torch.equal(actual, expected))
            suffix = stem.removeprefix("segment_")
            self.assertEqual(Path(legacy.folder_paths.get_annotated_filepath(record["seam_path"])),
                             data_folder / f"seam_{suffix}.mp4")
            self.assertEqual(Path(legacy.folder_paths.get_annotated_filepath(record["context_video_path"])),
                             data_folder / f"context_preview_{suffix}.mp4")
            self.assertEqual(build.call_args.args[0], legacy.folder_paths.get_annotated_filepath(source_record["context_video_path"]))
            self.assertEqual(entry["video"]["subfolder"], original_info["output_subfolder"])
            self.assertEqual(entry["seam"]["subfolder"], f"{original_info['output_subfolder']}/_data")
            for key, frames in (("seam_path", 58), ("context_video_path", 24)):
                with av.open(legacy.folder_paths.get_annotated_filepath(record[key])) as container:
                    self.assertEqual(container.streams.video[0].frames, frames)
                    self.assertEqual(len(container.streams.audio), 1)
            latent_path = Path(legacy.folder_paths.get_annotated_filepath(result[3][0]))
            self.assertEqual(latent_path, data_folder / f"{stem}.h3latent")
            restored = legacy.LoadLatent.execute(result[3][0])[0]
            for actual, expected in zip(restored["samples"].tensors, result[1]["samples"].tensors):
                self.assertTrue(torch.equal(actual, expected))
            self.assertEqual(result.ui["latents"][0]["subfolder"], f"{original_info['output_subfolder']}/_data")
            self.assertEqual(result.ui["latents"][0]["filename"], f"{stem}.h3latent")
            self.assertEqual(info["manifest"], original_info["manifest"])
            self.assertTrue(manifest_path.read_bytes().startswith(original_manifest))
            manifest = [json.loads(line) for line in manifest_path.read_text("utf-8").splitlines()]
            self.assertEqual(len(manifest), 4 + index)
            self.assertEqual(manifest[-1]["recipe"], entry["recipe_path"])
            self.assertEqual(manifest[-1]["latent"], result[3][0])
            for path, digest in hashes.items():
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest, path.name)
            hashes = {path: hashlib.sha256(path.read_bytes()).hexdigest()
                      for path in run_folder.rglob("*") if path.is_file() and path.name != "segments.jsonl"}
            recipes.append(entry["recipe_path"])
        self.assertEqual({entry["recipe_path"] for entry in review.list_segment_records()}, set(recipes))
        self.assertEqual({path.name for path in run_folder.iterdir()},
                         {"_data", *(f"segment_{index:04}.mp4" for index in range(1, 4)),
                          *(f"{stem}.mp4" for stem in expected_stems)})
        self.assertFalse((self.directory / "different").exists())
        self.assertEqual({path for path in self.directory.iterdir()}, {run_folder})
        self.mock_pipeline([])
        following = self.run_node(1, duration=0.2)
        self.assertNotEqual(json.loads(following[2])["output_subfolder"], original_info["output_subfolder"])
        self.assertEqual(len(list(self.directory.glob("H3_*"))), 2)

    def test_retry_sampling_failure_cleans_reservation_without_changing_original_files(self):
        self.mock_pipeline([])
        original = self.run_node(1, duration=0.2)
        recipe = original.ui["h3_sgun_segments"][0]["recipe_path"]
        before = {path: path.read_bytes() for path in self.directory.rglob("*") if path.is_file()}
        with patch.object(legacy.LongVideo, "_render_segment", side_effect=RuntimeError("sampling failed")):
            with self.assertRaisesRegex(RuntimeError, "sampling failed"):
                self.run_node(1, duration=0.2, retry_segment=recipe)
        self.assertEqual({path: path.read_bytes() for path in self.directory.rglob("*") if path.is_file()}, before)
        self.assertEqual(list(self.directory.rglob("*.reserve")), [])

    def test_temp_recipe_retries_into_output_without_writing_to_temp_or_original_batch(self):
        self.mock_pipeline([])
        original = self.run_node(1, duration=0.2)
        record = self.records(original)[0]
        temp_recipe = self.cache_directory / "segment_0001.h3segment.json"
        review.save_record(record, str(temp_recipe))
        temp_name = review.annotated_path(str(temp_recipe), "temp")
        before_temp = {path: path.read_bytes() for path in self.cache_directory.rglob("*") if path.is_file()}
        before_output = {path: path.read_bytes() for path in self.directory.rglob("*") if path.is_file()}
        self.mock_pipeline([])
        retry = self.run_node(1, duration=0.2, retry_segment=temp_name, filename_prefix="Imported")
        info = json.loads(retry[2])
        self.assertTrue(info["output_subfolder"].startswith("Imported_"))
        video_path = Path(legacy.folder_paths.get_annotated_filepath(info["video_paths"][0]))
        self.assertEqual(video_path.parent, self.directory / info["output_subfolder"])
        self.assertNotEqual(info["output_subfolder"], json.loads(original[2])["output_subfolder"])
        self.assertEqual({path: path.read_bytes() for path in self.cache_directory.rglob("*") if path.is_file()}, before_temp)
        for path, contents in before_output.items():
            self.assertEqual(path.read_bytes(), contents)

    def test_retry_reuses_prompt_and_uses_current_internal_sampling_controls(self):
        self.mock_pipeline([])
        original_sigmas = torch.tensor([1., .8, .2, 0.])
        with patch.object(module.BasicScheduler, "execute", return_value=legacy.io.NodeOutput(original_sigmas)):
            output = self.run_node(2, {"prompts": ["identity\n\nfirst", "identity\n\nsecond"], "repeat_last": False},
                                   sigmas=None, context_frames=39, denoise=.7)
        entry = output.ui["h3_sgun_segments"][1]
        self.assertEqual(review.load_record(entry["recipe_path"])["sigmas_source"], "internal")
        with patch.object(legacy.LongVideo, "_render_segment", return_value=(
                legacy.native_latent(), module.InputImpl.VideoFromFile(
                    legacy.folder_paths.get_annotated_filepath(self.records(output)[1]["video_path"])), 119)) as render, \
             patch.object(module.BasicScheduler, "execute", return_value=legacy.io.NodeOutput(original_sigmas)) as schedule:
            self.run_node(1, retry_segment=entry["recipe_path"], context_frames=5, denoise=1., sigmas=None,
                          retry_seed_mode="original")
        self.assertEqual(render.call_args.args[4], "identity\n\nsecond")
        self.assertEqual(render.call_args.args[9], 5)
        self.assertEqual(schedule.call_args.args[1:], ("simple", 8, 1.))

    def test_missing_or_outside_context_fails_before_sampling(self):
        self.mock_pipeline([])
        output = self.run_node(2)
        entry = output.ui["h3_sgun_segments"][1]
        record = review.load_record(entry["recipe_path"])
        recipe_path = Path(legacy.folder_paths.get_annotated_filepath(entry["recipe_path"]))
        for context in ("does-not-exist.h3context [output]", "../outside.h3context [output]"):
            with self.subTest(context=context):
                record["context"] = context
                review.save_record(record, str(recipe_path))
                with patch.object(legacy.LongVideo, "_render_segment") as render:
                    with self.assertRaises((ValueError, FileNotFoundError)):
                        self.run_node(1, retry_segment=entry["recipe_path"])
                    render.assert_not_called()

    def test_output_recipe_context_and_drift_stats_survive_temp_cleanup(self):
        self.mock_pipeline([])
        output = self.run_node(2, drift_correction=True)
        entry = output.ui["h3_sgun_segments"][1]
        record = review.load_record(entry["recipe_path"])
        self.assertTrue(entry["recipe_path"].endswith(" [output]"))
        self.assertTrue(record["context"].endswith(" [output]"))
        self.assertTrue(record["video_path"].endswith(" [output]"))
        self.assertEqual(list(self.directory.rglob("*.h3latent")), [])
        self.assertEqual([len(values) for values in record["drift_reference"]], [24, 24])
        for path in self.cache_directory.iterdir():
            shutil.rmtree(path) if path.is_dir() else path.unlink()
        self.mock_pipeline([])
        with patch.object(module, "drift_reference", side_effect=AssertionError("lost original drift reference")), \
             patch.object(module, "correct_video_drift", wraps=module.correct_video_drift) as correct:
            retry = self.run_node(1, retry_segment=entry["recipe_path"], seed=101, drift_correction=True,
                                  retry_seed_mode="original")
        self.assertEqual(correct.call_count, 1)
        for actual, expected in zip(correct.call_args.args[1], record["drift_reference"]):
            self.assertTrue(torch.equal(actual.cpu(), torch.tensor(expected)))
        self.assertEqual(json.loads(retry[2])["frames_per_segment"], [119])

    def test_external_latent_without_source_video_reports_unavailable_preview(self):
        self.mock_pipeline([])
        output = self.run_node(1, previous_latent=legacy.native_latent())
        record = self.records(output)[0]
        self.assertEqual(record["diagnostics"]["status"], "unavailable")
        self.assertIsNone(record["seam_path"])
        module.build_seam_preview.assert_not_called()

    def test_preview_failure_keeps_generated_segments_and_records(self):
        self.mock_pipeline([])
        module.build_seam_preview.side_effect = RuntimeError("preview failed")
        output = self.run_node(2)
        self.assertEqual(len(json.loads(output[2])["video_paths"]), 2)
        self.assertEqual(len(list(self.directory.rglob("segment_*.mp4"))), 2)
        self.assertEqual(self.records(output)[1]["diagnostics"]["status"], "error")

    def test_record_index_and_fingerprint_reflect_prompt_edits(self):
        self.mock_pipeline([])
        output = self.run_node(1)
        entry = output.ui["h3_sgun_segments"][0]
        records = review.list_segment_records()
        self.assertIn(entry["recipe_path"], [item["recipe_path"] for item in records])
        before = review.record_fingerprint(entry["recipe_path"])
        record = review.load_record(entry["recipe_path"])
        record["prompt"] = "edited action"
        review.save_record(record, legacy.folder_paths.get_annotated_filepath(entry["recipe_path"]))
        self.assertNotEqual(review.record_fingerprint(entry["recipe_path"]), before)
        self.assertNotIn(".h3context", "\n".join(legacy.latent_io.list_h3_latents()))

    def test_review_routes_are_registered_and_reject_invalid_paths(self):
        routes = review.web.RouteTableDef()
        with patch.object(package.PromptServer, "instance", SimpleNamespace(routes=routes), create=True):
            asyncio.run(package.H3LatentGuideExtension().on_load())
        self.assertTrue({("GET", "/h3-sgun/segments"), ("GET", "/h3-sgun/segment")}.issubset(
            {(route.method, route.path) for route in routes}))
        self.mock_pipeline([])
        output = self.run_node(1)
        path = output.ui["h3_sgun_segments"][0]["recipe_path"]
        response = asyncio.run(review.get_segment(SimpleNamespace(query={"path": path})))
        self.assertEqual(response.status, 200)
        self.assertEqual(json.loads(response.text)["prompt"], "one")
        for invalid in ("missing.h3segment.json [output]", "../outside.h3segment.json [output]"):
            with self.subTest(path=invalid):
                response = asyncio.run(review.get_segment(SimpleNamespace(query={"path": invalid})))
                self.assertEqual(response.status, 400)

    def test_retry_seed_text_preserves_uint64_precision_and_rejects_invalid_values(self):
        self.mock_pipeline([])
        big_seed = 0xfedcba9876543210
        output = self.run_node(1, seed=big_seed)
        entry = output.ui["h3_sgun_segments"][0]
        self.assertEqual(entry["seed_text"], str(big_seed))
        with patch.object(legacy.LongVideo, "_render_segment", return_value=(
                legacy.native_latent(), module.InputImpl.VideoFromFile(
                    legacy.folder_paths.get_annotated_filepath(self.records(output)[0]["video_path"])), 119)) as render:
            retry = self.run_node(1, seed=0, retry_segment=entry["recipe_path"], retry_seed=str(big_seed + 17),
                                  retry_seed_mode="custom")
        self.assertEqual(render.call_args.args[10], big_seed + 17)
        self.assertEqual(self.records(retry)[0]["seed"], big_seed + 17)
        for invalid in ("", "   ", "not-a-seed", "1.5", "-1", str(2 ** 64), None, 1.5):
            with self.subTest(seed=invalid), patch.object(legacy.LongVideo, "_render_segment") as render:
                with self.assertRaises(ValueError):
                    self.run_node(1, retry_segment=entry["recipe_path"], retry_seed=invalid, retry_seed_mode="custom")
                render.assert_not_called()

    def test_retry_defaults_to_fresh_random_seed_and_records_actual_uint64(self):
        self.mock_pipeline([])
        output = self.run_node(1, seed=100)
        entry = output.ui["h3_sgun_segments"][0]
        maximum = 0xffffffffffffffff
        with patch.object(legacy.LongVideo, "_render_segment", return_value=(
                legacy.native_latent(), module.InputImpl.VideoFromFile(
                    legacy.folder_paths.get_annotated_filepath(self.records(output)[0]["video_path"])), 119)) as render, \
             patch.object(module.random, "randint", side_effect=[maximum, 0]) as randomized:
            first = self.run_node(1, seed=100, retry_segment=entry["recipe_path"], retry_seed="invalid")
            second = self.run_node(1, seed=100, retry_segment=entry["recipe_path"], retry_seed="invalid")
        self.assertEqual([call.args[10] for call in render.call_args_list], [maximum, 0])
        self.assertEqual(randomized.call_args_list, [legacy.call(0, maximum), legacy.call(0, maximum)])
        for result, actual in ((first, maximum), (second, 0)):
            self.assertEqual(json.loads(result[2])["seeds"], [actual])
            self.assertEqual(json.loads(result[2])["retry_seed_mode"], "randomize")
            self.assertEqual(self.records(result)[0]["seed"], actual)
            self.assertEqual(self.records(result)[0]["settings"]["seed"], actual)
            self.assertEqual(result.ui["h3_sgun_segments"][0]["seed_text"], str(actual))

    def test_original_retry_seed_uses_actual_record_not_current_seed_or_custom_text(self):
        self.mock_pipeline([])
        big_seed = 0xfedcba9876543210
        output = self.run_node(1, seed=big_seed)
        entry = output.ui["h3_sgun_segments"][0]
        with patch.object(legacy.LongVideo, "_render_segment", return_value=(
                legacy.native_latent(), module.InputImpl.VideoFromFile(
                    legacy.folder_paths.get_annotated_filepath(self.records(output)[0]["video_path"])), 119)) as render, \
             patch.object(module.random, "randint", side_effect=AssertionError("original randomized")):
            retry = self.run_node(1, seed=25, retry_segment=entry["recipe_path"], retry_seed="invalid",
                                  retry_seed_mode="original")
        self.assertEqual(render.call_args.args[10], big_seed)
        self.assertEqual(self.records(retry)[0]["seed"], big_seed)
        self.assertEqual(json.loads(retry[2])["seeds"], [big_seed])

    def test_direct_latent_chaining_carries_source_video_for_preview(self):
        self.mock_pipeline([])
        first = self.run_node(1)
        source_path = legacy.folder_paths.get_annotated_filepath(first[1]["_h3_video_path"])
        self.assertTrue(Path(source_path).is_file())
        # This graph deliberately retains the first node's full LATENT output.
        self.mock_pipeline([])
        second = self.run_node(1, previous_latent=first[1])
        module.build_seam_preview.assert_called_once()
        self.assertEqual(module.build_seam_preview.call_args.args[0], source_path)
        self.assertEqual(self.records(second)[0]["diagnostics"]["status"], "ok")

    def test_invalid_source_video_path_only_disables_preview(self):
        self.mock_pipeline([])
        previous = legacy.native_latent()
        previous["_h3_video_path"] = "../outside.mp4 [output]"
        output = self.run_node(1, previous_latent=previous)
        self.assertEqual(len(json.loads(output[2])["video_paths"]), 1)
        self.assertEqual(self.records(output)[0]["diagnostics"]["status"], "error")


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]], verbosity=2)
