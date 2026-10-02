import asyncio
import gc
import importlib.util
import sys
import unittest
import weakref
from pathlib import Path

import torch


COMFY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(COMFY_ROOT))

from comfy.ldm.minimax.model import FRAME_PER_TOKEN, FRAME_RESCALE, PackedLayout
from comfy.nested_tensor import NestedTensor


spec = importlib.util.spec_from_file_location("h3_latent_guide", Path(__file__).resolve().parents[1] / "__init__.py")
package = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = package
spec.loader.exec_module(package)
Guide = package.MiniMaxH3LatentContinuationGuide


def av_latent(frames, height=4, width=4, dtype=torch.float32, device="cpu"):
    video_t = (frames - 5) // 17 * 5 + 2
    audio_t = round(frames * FRAME_RESCALE)
    video = torch.arange(video_t, dtype=dtype, device=device).reshape(1, 1, video_t, 1, 1).expand(1, 24, video_t, height, width).clone()
    audio = torch.arange(audio_t, dtype=dtype, device=device).reshape(1, 1, 1, audio_t).expand(1, 32, 2, audio_t).clone()
    return {"samples": NestedTensor((video, audio))}


def conditioning(metadata=None):
    return [[torch.zeros(1, 7, 4), metadata or {}]]


def check_registered_nodes_and_guide_sockets():
    extension = asyncio.run(package.comfy_entrypoint())
    assert asyncio.run(extension.get_node_list()) == [Guide, package.H3LoopPromptSchedule,
                                                    package.H3LongVideo, package.H3VideoPromptPlan, package.H3LoadLatent]
    schema = Guide.define_schema()
    assert [i.id for i in schema.inputs] == ["positive", "latent", "previous_latent", "context_frames"]
    assert [o.io_type for o in schema.outputs] == ["CONDITIONING", "INT"]


def check_tail_is_exact_independent_copy(device, dtype):
    previous = av_latent(243, dtype=dtype, device=device)
    target = av_latent(141)
    source_video, source_audio = previous["samples"].tensors
    source_copies = [t.clone() for t in previous["samples"].tensors]
    positive = conditioning({"marker": "keep"})
    result, trim = Guide.execute(positive, target, previous).result
    video_guide, audio_guide = result[0][1]["minimax_keyframes"]
    video = video_guide["latent"]
    audio = audio_guide["audio_latent"]
    assert trim == 22
    assert torch.equal(video, source_video[:, :, -7:])
    assert torch.equal(audio, source_audio[..., -40:])
    assert video.untyped_storage().data_ptr() != source_video.untyped_storage().data_ptr()
    assert audio.untyped_storage().data_ptr() != source_audio.untyped_storage().data_ptr()
    assert video.dtype == audio.dtype == dtype
    assert video.device == audio.device == source_video.device
    assert "minimax_keyframes" not in positive[0][1]
    for original, copy in zip(previous["samples"].tensors, source_copies):
        assert torch.equal(original, copy)


def check_native_layout_has_synchronized_context(previous_frames, context_frames):
    target = av_latent(141)
    previous = av_latent(previous_frames)
    refs = [{"kind": "audio", "ref_audio_t": 8, "audio_latent": torch.zeros(1, 32, 2, 8)}]
    result, trim = Guide.execute(conditioning({"minimax_refs": refs}), target, previous, context_frames).result
    guides = result[0][1]["minimax_keyframes"]
    video, audio = target["samples"].tensors
    layout = PackedLayout(7, video.shape[2], video.shape[3], video.shape[4], audio.shape[-1], keyframes=guides, refs=refs)
    assert result[0][1]["minimax_refs"] is refs
    cond_a, cond_b, _ = next(seg for seg in layout.segments if seg[2] == "cond")
    target_a, _, _ = layout.segments[-1]
    frame_rows = video.shape[-1] * video.shape[-2] // 4
    guide_video_t = guides[0]["latent"].shape[2]
    assert (previous["samples"].tensors[0].shape[2] - guide_video_t) % 5 == 0
    assert sum(FRAME_PER_TOKEN[k % 5] for k in range(guide_video_t)) == trim
    assert torch.equal(layout.position_ids[cond_a:cond_b], layout.position_ids[target_a:target_a + guide_video_t * frame_rows])
    ca, cb, _ = next(seg for seg in layout.segments if seg[2] == "cond_audio")
    origin = float(layout.position_ids[target_a, 0])
    times = layout.position_ids[ca:cb, 0] - origin
    assert torch.allclose(times, times.round())
    rt = guides[1]["audio_latent"].shape[-1]
    end = float(times.min()) + rt
    previous_audio_t = previous["samples"].tensors[1].shape[-1]
    overhang = previous_audio_t - previous_frames * FRAME_RESCALE
    assert abs(end - round(trim * FRAME_RESCALE + overhang)) < 1e-9
    assert not bool(layout.img_update[:guide_video_t * frame_rows].any())
    assert not bool(layout.audio_update[:rt * 2].any())
    assert bool(layout.img_update[-video.shape[2] * frame_rows:].all())


def check_scheduled_conditions_keep_their_own_guides_and_references():
    first = {"resolved_frame_index": 0, "latent": torch.zeros(1, 24, 1, 4, 4)}
    last_a = {"resolved_frame_index": 140, "latent": torch.zeros(1, 24, 1, 4, 4)}
    last_b = {"resolved_frame_index": 123, "latent": torch.ones(1, 24, 1, 4, 4)}
    first_metadata = {"minimax_keyframes": [first, last_a], "start_percent": 0.0, "end_percent": 0.5, "minimax_refs": [{"kind": "image"}]}
    second_metadata = {"minimax_keyframes": [last_b], "start_percent": 0.5, "end_percent": 1.0}
    positive = conditioning(first_metadata) + conditioning(second_metadata)
    result, _ = Guide.execute(positive, av_latent(141), av_latent(243)).result
    assert result[0][1]["minimax_keyframes"][-1] is last_a
    assert result[1][1]["minimax_keyframes"][-1] is last_b
    assert result[0][1]["minimax_refs"] is first_metadata["minimax_refs"]
    assert result[0][1]["end_percent"] == 0.5
    assert result[1][1]["start_percent"] == 0.5
    assert first_metadata["minimax_keyframes"] == [first, last_a]
    assert second_metadata["minimax_keyframes"] == [last_b]


def check_context_snaps_down_and_fits_both_clips(requested, previous_frames, target_frames, expected):
    _, trim = Guide.execute(conditioning(), av_latent(target_frames), av_latent(previous_frames), requested).result
    assert trim == expected


class TestLatentGuide(unittest.TestCase):
    def test_registered_nodes_and_guide_sockets(self):
        check_registered_nodes_and_guide_sockets()

    def test_tail_is_exact_independent_copy(self):
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        for device in devices:
            for dtype in (torch.float16, torch.float32, torch.bfloat16):
                with self.subTest(device=device, dtype=dtype):
                    check_tail_is_exact_independent_copy(device, dtype)

    def test_native_layout_has_synchronized_context(self):
        for previous_frames in (5, 22, 56, 124, 243, 260, 362):
            for context_frames in (5, 22, 39, 56):
                with self.subTest(previous_frames=previous_frames, context_frames=context_frames):
                    check_native_layout_has_synchronized_context(previous_frames, context_frames)

    def test_compact_tail_preserves_full_latent_guides(self):
        target = av_latent(141)
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        for device in devices:
            for dtype in (torch.float16, torch.float32, torch.bfloat16):
                for previous_frames in (5, 22, 56, 124, 141, 158, 243, 260, 362):
                    for context_frames in (5, 22, 23, 39, 56, 99):
                        with self.subTest(device=device, dtype=dtype, source=previous_frames, context=context_frames):
                            previous = av_latent(previous_frames, dtype=dtype, device=device)
                            compact = package._copy_context_tail(previous, context_frames)
                            before, trim = Guide.execute(conditioning(), target, previous, context_frames).result
                            after, compact_trim = Guide.execute(conditioning(), target, compact, context_frames).result
                            self.assertEqual(trim, compact_trim)
                            for old, new, key in zip(before[0][1]["minimax_keyframes"],
                                                     after[0][1]["minimax_keyframes"], ("latent", "audio_latent")):
                                self.assertEqual(old["resolved_frame_index"], new["resolved_frame_index"])
                                self.assertTrue(torch.equal(old[key].cpu(), new[key]))
                                self.assertEqual(old[key].dtype, new[key].dtype)
                                self.assertEqual(new[key].device.type, "cpu")
                            tv, ta = target["samples"].tensors
                            old_layout = PackedLayout(7, tv.shape[2], tv.shape[3], tv.shape[4], ta.shape[-1],
                                                      keyframes=before[0][1]["minimax_keyframes"])
                            new_layout = PackedLayout(7, tv.shape[2], tv.shape[3], tv.shape[4], ta.shape[-1],
                                                      keyframes=after[0][1]["minimax_keyframes"])
                            self.assertTrue(torch.equal(old_layout.position_ids, new_layout.position_ids))

    def test_compact_tail_releases_full_storage_and_preserves_audio_rounding(self):
        previous = av_latent(124)
        references = [weakref.ref(stream) for stream in previous["samples"].tensors]
        original_bytes = sum(stream.numel() * stream.element_size() for stream in previous["samples"].tensors)
        compact = package._copy_context_tail(previous, 22)
        video, audio, frames = package._av_streams(compact, "cached context")
        self.assertEqual(frames, 39)
        self.assertEqual(video.shape[2], 12)
        self.assertEqual(audio.shape[-1], 65)
        self.assertEqual(compact["_h3_audio_overhang"], round(124 * FRAME_RESCALE) - 124 * FRAME_RESCALE)
        self.assertLess(sum(stream.numel() * stream.element_size() for stream in compact["samples"].tensors), original_bytes)
        for copied, original in zip(compact["samples"].tensors, previous["samples"].tensors):
            self.assertNotEqual(copied.untyped_storage().data_ptr(), original.untyped_storage().data_ptr())
            self.assertEqual(copied.untyped_storage().nbytes(), copied.numel() * copied.element_size())
        del previous, original
        gc.collect()
        self.assertTrue(all(reference() is None for reference in references))

    def test_scheduled_conditions_keep_their_own_guides_and_references(self):
        with self.assertLogs(level="WARNING") as messages:
            check_scheduled_conditions_keep_their_own_guides_and_references()
        self.assertIn("已替换", messages.output[0])

    def test_context_snaps_down_and_fits_both_clips(self):
        for case in ((23, 243, 141, 22), (99, 56, 141, 56), (99, 243, 22, 22)):
            with self.subTest(case=case):
                check_context_snaps_down_and_fits_both_clips(*case)

    def test_resolution_mismatch_is_reported(self):
        with self.assertRaisesRegex(ValueError, "分辨率"):
            Guide.execute(conditioning(), av_latent(141, width=8), av_latent(243))

    def test_video_only_latent_is_reported(self):
        with self.assertRaisesRegex(ValueError, "联合音视频"):
            Guide.execute(conditioning(), av_latent(141), {"samples": torch.zeros(1, 24, 7, 4, 4)})

    def test_wrong_audio_duration_is_reported(self):
        previous = av_latent(243)
        previous["samples"].tensors[1] = previous["samples"].tensors[1][..., :-1]
        with self.assertRaisesRegex(ValueError, "音视频长度不匹配"):
            Guide.execute(conditioning(), av_latent(141), previous)

    def test_wrong_video_phase_is_reported(self):
        previous = av_latent(243)
        previous["samples"].tensors[0] = previous["samples"].tensors[0][:, :, :-1]
        with self.assertRaisesRegex(ValueError, "时间长度"):
            Guide.execute(conditioning(), av_latent(141), previous)


class TestLoopPrompts(unittest.TestCase):
    def test_crlf_delimiters_and_prompt_order(self):
        first, continuation = package.H3LoopPromptSchedule.execute(
            "shared description", "first action\r\n --- \r\nsecond action\r\n---\r\nthird action", 2).result
        self.assertEqual(first, "shared description\n\nfirst action")
        self.assertEqual(continuation, ["shared description\n\nsecond action", "shared description\n\nthird action"])

    def test_zero_continuations_and_unused_blocks(self):
        self.assertEqual(package.H3LoopPromptSchedule.execute("", "opening\n---\n", 0).result, ("opening", []))

    def test_missing_prompts_fail_before_sampling(self):
        with self.assertRaisesRegex(ValueError, "需要至少 4 段.*目前只有 2 段"):
            package.H3LoopPromptSchedule.execute("shared", "opening\n---\nsecond", 3)

    def test_empty_used_segment_is_reported(self):
        with self.assertRaisesRegex(ValueError, "第 2 段提示词为空"):
            package.H3LoopPromptSchedule.execute("shared", "opening\n---\n\n---\nthird", 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
