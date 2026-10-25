import asyncio
from contextlib import ExitStack
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import test_long_video as legacy


review = sys.modules[legacy.package.__name__ + ".segment_review"]
KEPT = ("video_path", "latent")


class TestSegmentCleanup(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.directory = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.roots = {kind: self.directory / kind for kind in ("output", "input", "temp")}
        self.trash = self.directory / "recycle-bin"
        for path in (*self.roots.values(), self.trash):
            path.mkdir()
        for kind, path in self.roots.items():
            self.stack.enter_context(patch.object(legacy.folder_paths, f"get_{kind}_directory",
                                                  return_value=str(path)))
        self.stack.enter_context(patch.object(legacy.folder_paths, "get_directory_by_type",
                                              side_effect=lambda kind: str(self.roots[kind])))
        self.moved = []
        self.stack.enter_context(patch.object(review, "send2trash", self.fake_trash, create=True))
        self.stack.enter_context(patch.object(review, "cleanup_revision", 0))

    def fake_trash(self, paths):
        if isinstance(paths, (str, Path)):
            paths = [paths]
        for name in paths:
            path = Path(name)
            self.assertTrue(path.is_relative_to(self.directory))
            if path.is_dir():
                self.assertEqual(list(path.iterdir()), [], "Only confirmed empty directories may be recycled")
                self.assertNotIn(path, self.roots.values(), "ComfyUI root directories must be preserved")
            shutil.move(str(path), str(self.trash / f"{len(self.moved):04}_{path.name}"))
            self.moved.append(path)

    def annotated(self, path, kind="output"):
        return f"{Path(path).relative_to(self.roots[kind]).as_posix()} [{kind}]"

    def create_record(self, segment=2, *, retry=None, kind="output", batch="H3_00001", flat=False):
        run_folder = self.roots[kind] / batch
        data_folder = run_folder if flat else run_folder / "_data"
        data_folder.mkdir(parents=True, exist_ok=True)
        suffix = f"{segment:04}" + (f"_retry_{retry:03}" if retry is not None else "")
        stem = f"segment_{suffix}"
        files = {"recipe": data_folder / f"{stem}.h3segment.json",
                 "video_path": run_folder / f"{stem}.mp4",
                 "context": data_folder / f"{stem}.h3context",
                 "seam_path": data_folder / f"seam_{suffix}.mp4",
                 "context_video_path": data_folder / f"context_preview_{suffix}.mp4",
                 "latent": data_folder / f"{stem}.h3latent"}
        record = {"format": review.RECORD_FORMAT, "segment": segment, "seed": 123,
                  "width": 32, "height": 32, "fps": 24, "frames": 119,
                  "prompt": "a quiet conversation", "settings": {"duration": 5.0},
                  "sigmas_source": "internal", "sigmas": [1.0, 0.0], "diagnostics": None,
                  **{key: self.annotated(files[key], kind) for key in
                     ("video_path", "context", "seam_path", "context_video_path")}}
        for key, path in files.items():
            if key != "recipe":
                path.write_bytes(f"owned {key}".encode())
        review.save_record(record, str(files["recipe"]))
        return self.annotated(files["recipe"], kind), record, files

    def save(self, files, record):
        review.save_record(record, str(files["recipe"]))

    def assert_record_list(self, result, expected):
        self.assertEqual({entry["recipe_path"] for entry in result["segments"]}, set(expected))

    def assert_only_record_data_cleaned(self, files):
        self.assertEqual(files["video_path"].read_bytes(), b"owned video_path")
        self.assertNotIn(files["video_path"], self.moved)
        self.assertTrue(files["video_path"].parent.is_dir())
        # Saved .h3latent files are user output and survive cleanup.
        self.assertEqual(files["latent"].read_bytes(), b"owned latent")
        self.assertNotIn(files["latent"], self.moved)
        for key, path in files.items():
            if key not in KEPT:
                self.assertFalse(path.exists(), f"Record data should be recycled: {path}")
                self.assertIn(path, self.moved)

    def cleanup_request(self, payload, *, queue_remaining=0, bad_json=None):
        async def json_body():
            if bad_json is not None:
                raise bad_json
            return payload

        queue = SimpleNamespace(mutex=threading.RLock(), get_tasks_remaining=lambda: queue_remaining)
        with patch.object(legacy.package.PromptServer, "instance", SimpleNamespace(prompt_queue=queue), create=True):
            return asyncio.run(review.cleanup_segments(SimpleNamespace(json=json_body)))

    def test_selected_record_data_recycles_while_formal_video_and_other_outputs_stay(self):
        name, _, files = self.create_record()
        keep_name, _, keep_files = self.create_record(3)
        foreign = files["video_path"].parent / "H3_combined.mp4"
        foreign.write_bytes(b"keep combined")
        unrelated = files["recipe"].parent / "notes.json"
        unrelated.write_text("{}", encoding="utf-8")
        result = review.recycle_segment_records([name])
        self.assertFalse(result["errors"])
        self.assert_only_record_data_cleaned(files)
        self.assertEqual(set(self.moved), {path for key, path in files.items() if key not in KEPT})
        self.assertTrue(all(path.is_file() for path in keep_files.values()))
        self.assertEqual(foreign.read_bytes(), b"keep combined")
        self.assertTrue(unrelated.is_file())
        self.assert_record_list(result, [keep_name])

    def test_empty_list_cleans_nothing_and_returns_current_records(self):
        name, _, files = self.create_record()
        result = review.recycle_segment_records([])
        self.assertEqual(self.moved, [])
        self.assertTrue(all(path.is_file() for path in files.values()))
        self.assert_record_list(result, [name])

    def test_flat_old_records_and_retry_versions_cleanup_with_their_own_names(self):
        old_name, _, old_files = self.create_record(flat=True, batch="old_flat")
        retry_name, _, retry_files = self.create_record(retry=7)
        keep_name, _, keep_files = self.create_record(3, batch="old_flat", flat=True)
        result = review.recycle_segment_records([old_name, retry_name])
        self.assertFalse(result["errors"])
        self.assert_only_record_data_cleaned(old_files)
        self.assert_only_record_data_cleaned(retry_files)
        self.assertTrue(all(path.is_file() for path in keep_files.values()))
        self.assert_record_list(result, [keep_name])

    def test_original_retry_legacy_and_temp_formal_videos_remain_byte_identical(self):
        records = [self.create_record(1), self.create_record(1, retry=1),
                   self.create_record(1, retry=2), self.create_record(2),
                   self.create_record(2, flat=True, batch="old_flat"),
                   self.create_record(2, retry=3, flat=True, batch="old_flat"),
                   self.create_record(1, kind="temp")]
        videos = {}
        for index, (_, _, files) in enumerate(records):
            video = files["video_path"]
            video.write_bytes(f"formal video {index}".encode())
            videos[video] = video.read_bytes()
            pending_video = Path(str(video) + ".tmp")
            pending_video.write_bytes(f"pending video {index}".encode())
            videos[pending_video] = pending_video.read_bytes()
        result = review.recycle_segment_records([name for name, _, _ in records])
        self.assertFalse(result["errors"])
        self.assertEqual(result["recycled_records"], len(records))
        self.assert_record_list(result, [])
        self.assertEqual({path: path.read_bytes() for path in videos}, videos)
        self.assertTrue(all(path.parent.is_dir() for path in videos))
        self.assertFalse(set(videos).intersection(self.moved))
        for _, _, files in records:
            self.assertTrue(all(not path.exists() for key, path in files.items()
                                if key not in KEPT))

    def test_input_imports_are_skipped_and_temp_generated_records_can_be_recycled(self):
        imported, _, imported_files = self.create_record(kind="input")
        temporary, _, temporary_files = self.create_record(kind="temp")
        result = review.recycle_segment_records([imported, temporary])
        self.assertTrue(all(path.is_file() for path in imported_files.values()))
        self.assert_only_record_data_cleaned(temporary_files)
        self.assertTrue(result["skipped_records"])
        self.assert_record_list(result, [imported])

    def test_manifest_recycles_only_after_all_records_in_batch_are_cleared(self):
        first, _, first_files = self.create_record(1)
        second, _, second_files = self.create_record(2)
        manifest = first_files["recipe"].parent / "segments.jsonl"
        manifest.write_text("\n".join(json.dumps({"recipe": name}) for name in (first, second)) + "\n",
                            encoding="utf-8")
        review.recycle_segment_records([first])
        self.assertTrue(manifest.is_file())
        # Saved latents are user output; once the user removes them the manifest goes too.
        first_files["latent"].unlink()
        second_files["latent"].unlink()
        result = review.recycle_segment_records([second])
        self.assertFalse(result["errors"])
        self.assertFalse(manifest.exists())
        self.assertIn(manifest, self.moved)
        for files in (first_files, second_files):
            self.assertEqual(files["video_path"].read_bytes(), b"owned video_path")
            self.assertNotIn(files["video_path"], self.moved)
            self.assertTrue(all(not path.exists() for key, path in files.items() if key not in KEPT))
        self.assert_record_list(result, [])

    def test_manifest_survives_while_saved_latents_remain(self):
        first, _, first_files = self.create_record(1)
        second, _, second_files = self.create_record(2)
        manifest = first_files["recipe"].parent / "segments.jsonl"
        manifest.write_text(json.dumps({"recipe": first}) + "\n", encoding="utf-8")
        result = review.recycle_segment_records([first, second])
        self.assertFalse(result["errors"])
        self.assertTrue(manifest.is_file())
        self.assertNotIn(manifest, self.moved)
        self.assertTrue(first_files["latent"].is_file())
        self.assertTrue(second_files["latent"].is_file())
        self.assert_record_list(result, [])

    @unittest.skipUnless(os.name == "nt", "Windows junction regression")
    def test_output_root_behind_directory_junction_can_be_cleaned(self):
        real = self.directory / "real-output"
        real.mkdir()
        link = self.directory / "linked-output"
        created = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(real)],
                                 capture_output=True, check=False)
        if created.returncode:
            self.skipTest("Host does not permit creating directory junctions")
        try:
            self.roots["output"] = link
            with patch.object(legacy.folder_paths, "get_output_directory", return_value=str(link)):
                name, _, files = self.create_record()
                result = review.recycle_segment_records([name])
            self.assertFalse(result["errors"])
            self.assertEqual(result["recycled_records"], 1)
            self.assertFalse(files["recipe"].exists())
            self.assertEqual(files["video_path"].read_bytes(), b"owned video_path")
            self.assertTrue(files["latent"].is_file())
        finally:
            if link.is_junction():
                link.rmdir()

    def test_shared_context_referenced_by_unselected_record_survives(self):
        selected, _, selected_files = self.create_record(2)
        keep, keep_record, keep_files = self.create_record(3)
        keep_record["context"] = self.annotated(selected_files["context"])
        keep_record["context_video_path"] = self.annotated(selected_files["context_video_path"])
        self.save(keep_files, keep_record)
        result = review.recycle_segment_records([selected])
        self.assertFalse(result["errors"])
        self.assertTrue(selected_files["context"].is_file())
        self.assertTrue(selected_files["context_video_path"].is_file())
        self.assertFalse(selected_files["recipe"].exists())
        self.assertEqual(selected_files["video_path"].read_bytes(), b"owned video_path")
        self.assertNotIn(selected_files["video_path"], self.moved)
        self.assertTrue(all(path.is_file() for path in keep_files.values()))
        self.assert_record_list(result, [keep])
        self.assertTrue(review.record_entry(review.load_record(keep), keep)["context_available"])

    def test_missing_files_and_duplicate_paths_do_not_fail_or_recycle_twice(self):
        name, _, files = self.create_record()
        files["context_video_path"].unlink()
        alias = name.replace("H3_00001/", "H3_00001/./")
        result = review.recycle_segment_records([name, name, alias])
        self.assertFalse(result["errors"])
        self.assertEqual(result["recycled_records"], 1)
        self.assertEqual(len(self.moved), len(set(self.moved)))
        self.assertFalse(files["recipe"].exists())
        self.assert_record_list(result, [])

    def test_cleanup_invalidates_normal_generation_cache_only_when_files_are_recycled(self):
        def fingerprint():
            return legacy.LongVideo.fingerprint_inputs(segments=3, seed_mode="fixed", seed=42)
        before = fingerprint()
        name, _, files = self.create_record()
        self.assertEqual(fingerprint(), before)
        imported, _, imported_files = self.create_record(kind="input")
        skipped = review.recycle_segment_records([imported])
        self.assertEqual(skipped["skipped_records"], 1)
        self.assertEqual(fingerprint(), before)
        self.assertEqual(review.cleanup_revision, before)
        self.assertTrue(all(path.is_file() for path in imported_files.values()))
        rejected = self.cleanup_request({"paths": ["../outside.h3segment.json [output]"]})
        self.assertEqual(rejected.status, 400)
        self.assertEqual(fingerprint(), before)
        self.assertEqual(review.cleanup_revision, before)
        review.recycle_segment_records([])
        self.assertEqual(fingerprint(), before)
        result = review.recycle_segment_records([name])
        self.assertGreater(result["recycled_files"], 0)
        self.assertEqual(fingerprint(), before + 1)
        self.assertEqual(review.cleanup_revision, before + 1)
        self.assertEqual(fingerprint(), fingerprint())
        self.assert_only_record_data_cleaned(files)

    def test_already_deleted_recipe_is_skipped_without_touching_its_old_siblings(self):
        name, _, files = self.create_record()
        files["recipe"].unlink()
        result = review.recycle_segment_records([name])
        self.assertEqual(self.moved, [])
        self.assertTrue(all(path.is_file() for key, path in files.items() if key != "recipe"))
        self.assertFalse(result["segments"])

    def test_recycle_failure_retains_recipe_and_file_for_safe_retry(self):
        name, _, files = self.create_record()
        def fail_context(paths):
            names = [paths] if isinstance(paths, (str, Path)) else paths
            for path in names:
                if Path(path) == files["context"]:
                    raise OSError("Recycle Bin is unavailable")
                self.fake_trash(path)
        with patch.object(review, "send2trash", fail_context):
            result = review.recycle_segment_records([name])
        self.assertTrue(result["errors"])
        self.assertTrue(files["recipe"].is_file())
        self.assertTrue(files["context"].is_file())
        self.assertEqual(files["video_path"].read_bytes(), b"owned video_path")
        self.assertNotIn(files["recipe"], self.moved)
        self.assert_record_list(result, [name])
        retried = review.recycle_segment_records([name])
        self.assertFalse(retried["errors"])
        self.assert_only_record_data_cleaned(files)

    def test_all_recycle_failures_never_fall_back_to_permanent_deletion(self):
        name, _, files = self.create_record()
        before = {path: path.read_bytes() for path in files.values()}
        with patch.object(review, "send2trash", side_effect=OSError("Access denied")):
            result = review.recycle_segment_records([name])
        self.assertTrue(result["errors"])
        self.assertEqual({path: path.read_bytes() for path in files.values()}, before)
        self.assertEqual(self.moved, [])

    def test_dependency_missing_keeps_every_file_and_returns_service_unavailable(self):
        name, _, files = self.create_record()
        with patch.object(review, "send2trash", None):
            response = self.cleanup_request({"paths": [name]})
        self.assertEqual(response.status, 503)
        self.assertTrue(json.loads(response.body)["error"])
        self.assertTrue(all(path.is_file() for path in files.values()))
        self.assertEqual(self.moved, [])

    def test_untrusted_recipe_and_root_paths_cannot_recycle_anything(self):
        name, _, files = self.create_record()
        outside = self.directory / "segment_9999.h3segment.json"
        outside.write_bytes(files["recipe"].read_bytes())
        foreign = self.roots["output"] / "anything.h3segment.json"
        foreign.write_bytes(files["recipe"].read_bytes())
        invalid = ["../segment_9999.h3segment.json [output]", str(outside) + " [output]",
                   ". [output]", "H3_00001 [output]", self.annotated(foreign)]
        for path in invalid:
            with self.subTest(path=path):
                response = self.cleanup_request({"paths": [path]})
                self.assertEqual(response.status, 400)
                self.assertEqual(self.moved, [])
        self.assertTrue(outside.is_file())
        self.assertTrue(foreign.is_file())
        self.assertTrue(all(path.is_file() for path in files.values()))

    def test_record_paths_cannot_expand_cleanup_to_an_unrelated_video_or_context(self):
        name, record, files = self.create_record()
        foreign = files["video_path"].parent / "my_final.mp4"
        foreign.write_bytes(b"do not touch")
        record["video_path"] = self.annotated(foreign)
        self.save(files, record)
        response = self.cleanup_request({"paths": [name]})
        self.assertEqual(response.status, 400)
        self.assertEqual(foreign.read_bytes(), b"do not touch")
        self.assertTrue(files["recipe"].is_file())
        self.assertEqual(self.moved, [])

    def test_traversal_inside_record_context_rejects_entire_request_before_mutating(self):
        good, _, good_files = self.create_record(1)
        bad, record, bad_files = self.create_record(2)
        record["context"] = "../outside.h3context [output]"
        self.save(bad_files, record)
        response = self.cleanup_request({"paths": [good, bad]})
        self.assertEqual(response.status, 400)
        self.assertEqual(self.moved, [])
        self.assertTrue(all(path.is_file() for path in (*good_files.values(), *bad_files.values())))

    def test_symlinked_owned_file_outside_batch_is_not_recycled(self):
        name, _, files = self.create_record()
        outside = self.directory / "valuable.mp4"
        outside.write_bytes(b"precious")
        files["video_path"].unlink()
        try:
            files["video_path"].symlink_to(outside)
        except OSError:
            self.skipTest("Host does not permit creating symlinks")
        response = self.cleanup_request({"paths": [name]})
        self.assertEqual(response.status, 400)
        self.assertEqual(outside.read_bytes(), b"precious")
        self.assertTrue(files["recipe"].is_file())
        self.assertEqual(self.moved, [])

    def test_foreign_sidecar_names_in_data_are_never_recycled(self):
        name, _, files = self.create_record()
        foreign = files["recipe"].parent / "segment_9999.h3context"
        foreign.write_bytes(b"unrelated")
        result = review.recycle_segment_records([name])
        self.assertFalse(result["errors"])
        self.assertEqual(foreign.read_bytes(), b"unrelated")
        self.assertNotIn(foreign, self.moved)

    def test_symlinked_file_into_another_batch_is_not_recycled(self):
        name, _, files = self.create_record()
        _, _, other_files = self.create_record(batch="H3_00002")
        files["video_path"].unlink()
        try:
            files["video_path"].symlink_to(other_files["video_path"])
        except OSError:
            self.skipTest("Host does not permit creating symlinks")
        response = self.cleanup_request({"paths": [name]})
        self.assertEqual(response.status, 400)
        self.assertEqual(other_files["video_path"].read_bytes(), b"owned video_path")
        self.assertTrue(files["recipe"].is_file())
        self.assertEqual(self.moved, [])

    def test_unlisted_invalid_record_keeps_batch_manifest_and_directory(self):
        name, _, files = self.create_record()
        malformed = files["recipe"].parent / "segment_0009.h3segment.json"
        malformed.write_text("{", encoding="utf-8")
        manifest = files["recipe"].parent / "segments.jsonl"
        manifest.write_text(json.dumps({"recipe": name}) + "\n", encoding="utf-8")
        result = review.recycle_segment_records([name])
        self.assertFalse(result["errors"])
        self.assertTrue(manifest.is_file())
        self.assertEqual(malformed.read_text(encoding="utf-8"), "{")
        self.assertNotIn(manifest, self.moved)

    @unittest.skipUnless(os.name == "nt", "Windows junction regression")
    def test_windows_directory_junction_cannot_escape_output_root(self):
        name, _, files = self.create_record()
        batch = files["video_path"].parent
        target = self.directory / "outside-output"
        self.assertTrue(batch.is_relative_to(self.directory))
        self.assertTrue(target.is_relative_to(self.directory))
        batch.rename(target)
        created = subprocess.run(["cmd", "/c", "mklink", "/J", str(batch), str(target)],
                                 capture_output=True, check=False)
        if created.returncode:
            self.skipTest("Host does not permit creating directory junctions")
        try:
            self.assertTrue(batch.is_junction())
            response = self.cleanup_request({"paths": [name]})
            self.assertEqual(response.status, 400)
            self.assertEqual(self.moved, [])
            self.assertTrue(all((target / path.relative_to(batch)).is_file() for path in files.values()))
        finally:
            if batch.is_junction():
                batch.rmdir()

    def test_cleanup_holds_queue_mutex_during_validation_and_recycling(self):
        name, _, _ = self.create_record()
        lock = threading.RLock()
        def queue_remaining():
            self.assertTrue(lock._is_owned())
            return 0
        def recycle(paths):
            self.assertTrue(lock._is_owned())
            self.assertEqual(paths, [name])
            return {"recycled_records": 0, "recycled_files": 0, "skipped_records": 0,
                    "errors": [], "segments": []}
        async def json_body():
            return {"paths": [name]}
        server = SimpleNamespace(prompt_queue=SimpleNamespace(mutex=lock, get_tasks_remaining=queue_remaining))
        with patch.object(legacy.package.PromptServer, "instance", server, create=True), \
                patch.object(review, "recycle_segment_records", side_effect=recycle):
            response = asyncio.run(review.cleanup_segments(SimpleNamespace(json=json_body)))
        self.assertEqual(response.status, 200)

    def test_busy_queue_route_returns_conflict_and_never_enters_cleanup(self):
        name, _, files = self.create_record()
        with patch.object(review, "recycle_segment_records") as recycle:
            response = self.cleanup_request({"paths": [name]}, queue_remaining=1)
        self.assertEqual(response.status, 409)
        recycle.assert_not_called()
        self.assertTrue(all(path.is_file() for path in files.values()))

    def test_route_is_registered_as_post_and_success_returns_new_list(self):
        name, _, files = self.create_record()
        routes = review.web.RouteTableDef()
        with patch.object(legacy.package.PromptServer, "instance", SimpleNamespace(routes=routes), create=True):
            asyncio.run(legacy.package.H3LatentGuideExtension().on_load())
        registered_routes = {(route.method, route.path) for route in routes}
        self.assertIn(("POST", review.SEGMENT_CLEANUP_ROUTE), registered_routes)
        self.assertNotIn(("POST", "/h3-sgun/segments/cleanup"), registered_routes)
        response = self.cleanup_request({"paths": [name]})
        self.assertEqual(response.status, 200)
        result = json.loads(response.body)
        self.assertFalse(result["errors"])
        self.assert_record_list(result, [])
        self.assert_only_record_data_cleaned(files)

    def test_invalid_json_and_payloads_return_400_without_any_mutation(self):
        name, _, files = self.create_record()
        for payload in (None, [], {}, {"paths": name}, {"paths": [None]},
                        {"paths": [name, 17]}):
            with self.subTest(payload=payload):
                response = self.cleanup_request(payload)
                self.assertEqual(response.status, 400)
                self.assertEqual(self.moved, [])
        response = self.cleanup_request({}, bad_json=json.JSONDecodeError("invalid JSON", "{", 1))
        self.assertEqual(response.status, 400)
        self.assertTrue(all(path.is_file() for path in files.values()))


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]], verbosity=2)
