"""Standalone boundary regressions extracted from the real local node sources.

Only Python's standard library is required. AST extraction avoids importing
ComfyUI or torch; the small tensor doubles model slicing, shapes and allocation
metadata only. These are NOT native ComfyUI, PyTorch, GPU or encoding integration
tests. Run with: python -B -X utf8 tests/test_boundary_regressions.py
"""

import ast
from array import array
import logging
import math
from pathlib import Path
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]


def load_definitions(filename, functions=(), methods=None, namespace=None):
    """Compile selected, unchanged function bodies from the checked-out source."""
    path = ROOT / filename
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    selected = []
    found = set()
    methods = methods or {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in functions:
            selected.append(node)
            found.add(node.name)
        elif isinstance(node, ast.ClassDef) and node.name in methods:
            wanted = methods[node.name]
            body = [item for item in node.body
                    if isinstance(item, ast.FunctionDef) and item.name in wanted]
            found.update(f"{node.name}.{item.name}" for item in body)
            selected.append(ast.ClassDef(name=node.name, bases=[], keywords=[],
                                         body=body, decorator_list=[]))
    expected = set(functions) | {
        f"{owner}.{name}" for owner, names in methods.items() for name in names
    }
    if found != expected:
        raise AssertionError(f"Missing source definitions: {expected - found}")
    code = ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))
    scope = dict(namespace or {})
    exec(compile(code, str(path), "exec"), scope)
    return scope


class AudioTensor:
    """Last-axis views share storage, so input-mutation assertions are meaningful."""

    def __init__(self, rows, *, leading=None, dtype="float32", device="cpu"):
        self.rows = tuple(array("f", row) for row in rows)
        self.leading = tuple(leading or (1, len(self.rows)))
        assert math.prod(self.leading) == len(self.rows)
        self.offset = 0
        self.length = len(self.rows[0])
        assert all(len(row) == self.length for row in self.rows)
        self.dtype, self.device = dtype, device

    @property
    def shape(self):
        return (*self.leading, self.length)

    def _bounds(self, key):
        assert len(key) == 2 and key[0] is Ellipsis
        start, stop, step = key[1].indices(self.length)
        assert step == 1
        return start, max(start, stop)

    def __getitem__(self, key):
        start, stop = self._bounds(key)
        view = object.__new__(type(self))
        view.__dict__.update(self.__dict__)
        view.offset += start
        view.length = stop - start
        return view

    def __setitem__(self, key, value):
        start, stop = self._bounds(key)
        assert value.shape == (*self.leading, stop - start)
        for row, values in zip(self.rows, value.values()):
            row[self.offset + start:self.offset + stop] = values

    def new_zeros(self, shape):
        return type(self)(
            [array("f", [0.0]) * shape[-1] for _ in range(math.prod(shape[:-1]))],
            leading=shape[:-1], dtype=self.dtype, device=self.device,
        )

    def values(self):
        return tuple(row[self.offset:self.offset + self.length] for row in self.rows)


class ShapeTensor:
    """Shape-only latent supporting the guide's slices and observable clones."""

    def __init__(self, shape, clones, label, forbid_clone=False):
        self.shape = tuple(shape)
        self.ndim = len(shape)
        self.clones, self.label, self.forbid_clone = clones, label, forbid_clone

    def __getitem__(self, key):
        selectors = list(key)
        if Ellipsis in selectors:
            index = selectors.index(Ellipsis)
            selectors[index:index + 1] = [slice(None)] * (
                self.ndim - len(selectors) + 1)
        selectors += [slice(None)] * (self.ndim - len(selectors))
        shape = [len(range(*selector.indices(size)))
                 for selector, size in zip(selectors, self.shape)]
        return type(self)(shape, self.clones, self.label, self.forbid_clone)

    def clone(self):
        self.clones.append(self.label)
        if self.forbid_clone:
            raise AssertionError("Guide cloned data before rejecting full overlap")
        return type(self)(self.shape, self.clones, self.label)


class NestedTensor:
    def __init__(self, tensors):
        self.tensors = tuple(tensors)


class NodeOutput:
    def __init__(self, *result):
        self.result = result


LONG = load_definitions("long_video.py", functions=("_trim_audio",),
                        methods={"H3LongVideo": ("fingerprint_inputs",)})
GUIDE = load_definitions(
    "__init__.py", functions=("_av_streams",),
    methods={"MiniMaxH3LatentContinuationGuide": ("execute",)},
    namespace={"NestedTensor": NestedTensor, "FRAME_RESCALE": 40 / 24,
               "io": SimpleNamespace(NodeOutput=NodeOutput), "logging": logging},
)
trim_audio = LONG["_trim_audio"]
fingerprint = LONG["H3LongVideo"].fingerprint_inputs
guide = GUIDE["MiniMaxH3LatentContinuationGuide"].execute


def latent(frames, clones, forbid_clone=False):
    video = ShapeTensor((1, 24, (frames - 5) // 17 * 5 + 2, 2, 2),
                        clones, "video", forbid_clone)
    audio = ShapeTensor((1, 32, 2, round(frames * 40 / 24)),
                        clones, "audio", forbid_clone)
    return {"samples": NestedTensor((video, audio))}


class TestAudioBoundaries(unittest.TestCase):
    def test_real_158_frame_decode_fills_266_sample_gap(self):
        # H3: 40 latent tokens/s, decoder hop 800, output rate 32000 Hz.
        source_samples = 263 * 800
        waveform = AudioTensor([
            ((index % 16) / 16 for index in range(source_samples)),
            (-0.5 for _ in range(source_samples)),
        ])
        before = waveform.values()
        audio = {"waveform": waveform, "sample_rate": 32000}
        output = trim_audio(audio, 22, 158 - 22)["waveform"]
        self.assertEqual(output.shape, (1, 2, 181333))
        self.assertEqual(source_samples - 29333, 181067)
        for original, actual in zip(before, output.values()):
            self.assertEqual(actual[:181067], original[29333:])
            self.assertEqual(actual[181067:], array("f", [0]) * 266)
        self.assertIs(audio["waveform"], waveform)
        self.assertEqual(waveform.values(), before)

    def test_long_audio_is_cropped_without_changing_samples(self):
        waveform = AudioTensor([[1, 2, 3, 4, 5, 6, 7], [-1, -2, -3, -4, -5, -6, -7]])
        before = waveform.values()
        output = trim_audio({"waveform": waveform, "sample_rate": 24}, 2, 3)
        self.assertEqual(output["waveform"].values(),
                         (array("f", [3, 4, 5]), array("f", [-3, -4, -5])))
        self.assertEqual(waveform.values(), before)

    def test_short_audio_preserves_metadata_and_pads_only_tail(self):
        waveform = AudioTensor([[1, 2, 3], [4, 5, 6]])
        metadata = object()
        audio = {"waveform": waveform, "sample_rate": 24, "metadata": metadata}
        output = trim_audio(audio, 1, 4)
        self.assertEqual(output["waveform"].values(),
                         (array("f", [2, 3, 0, 0]), array("f", [5, 6, 0, 0])))
        self.assertEqual(waveform.values(),
                         (array("f", [1, 2, 3]), array("f", [4, 5, 6])))
        self.assertIs(output["metadata"], metadata)
        self.assertIsNot(output, audio)

    def test_empty_or_fully_trimmed_audio_becomes_silence(self):
        for rows, trim in (([[], []], 0), ([[1, 2], [3, 4]], 5)):
            with self.subTest(rows=rows, trim=trim):
                waveform = AudioTensor(rows)
                before = waveform.values()
                output = trim_audio({"waveform": waveform, "sample_rate": 24}, trim, 3)
                self.assertEqual(output["waveform"].shape, (1, 2, 3))
                self.assertEqual(output["waveform"].values(),
                                 (array("f", [0, 0, 0]), array("f", [0, 0, 0])))
                self.assertEqual(waveform.values(), before)

    def test_allocation_preserves_leading_dimensions_dtype_and_device(self):
        # Device/dtype are contract markers, not real GPU or low-precision tests.
        for dtype, device in (("float32", "cpu"), ("float16", "cuda:1")):
            with self.subTest(dtype=dtype, device=device):
                waveform = AudioTensor([[1], [2], [3], [4]], leading=(2, 2),
                                       dtype=dtype, device=device)
                output = trim_audio({"waveform": waveform, "sample_rate": 24}, 0, 2)
                padded = output["waveform"]
                self.assertEqual(padded.shape, (2, 2, 2))
                self.assertEqual((padded.dtype, padded.device), (dtype, device))
                self.assertEqual(padded.values(),
                                 tuple(array("f", [value, 0]) for value in range(1, 5)))


class TestRandomCacheFingerprint(unittest.TestCase):
    def test_random_multiple_segments_get_distinct_fingerprints(self):
        first = fingerprint(segments=4, seed_mode="randomize", seed=100, model=object())
        second = fingerprint(segments=4, seed_mode="randomize", seed=100, model=object())
        self.assertTrue(math.isnan(first) and math.isnan(second))
        self.assertNotEqual(first, second)

    def test_default_single_segment_and_deterministic_modes_stay_cacheable(self):
        self.assertIsNone(fingerprint())
        for mode in ("fixed", "increment", "decrement", "randomize"):
            counts = (1,) if mode == "randomize" else (1, 4)
            for count in counts:
                with self.subTest(mode=mode, count=count):
                    self.assertIsNone(fingerprint(segments=count, seed_mode=mode,
                                                  clip=None, prompts=None))
                    self.assertEqual(fingerprint(segments=count, seed_mode=mode),
                                     fingerprint(segments=count, seed_mode=mode))

    def test_linked_inputs_are_conservative_without_disabling_known_safe_cases(self):
        for count, mode in ((None, "randomize"), (4, None), (None, None)):
            with self.subTest(count=count, mode=mode):
                self.assertTrue(math.isnan(fingerprint(segments=count, seed_mode=mode)))
        for count, mode in ((1, None), (None, "fixed"), (None, "increment"),
                            (None, "decrement")):
            with self.subTest(count=count, mode=mode):
                self.assertIsNone(fingerprint(segments=count, seed_mode=mode))


class TestGuideBoundaries(unittest.TestCase):
    def test_full_overlap_is_rejected_before_any_clone(self):
        for previous_frames, target_frames, context in (
                (124, 22, 22), (124, 22, 39), (22, 22, 3600), (5, 5, 5)):
            with self.subTest(previous=previous_frames, target=target_frames, context=context):
                clones = []
                with self.assertRaises(ValueError):
                    guide([[object(), {}]], latent(target_frames, clones, True),
                          latent(previous_frames, clones, True), context)
                self.assertEqual(clones, [])

    def test_valid_context_is_aligned_and_keeps_new_frames(self):
        for previous_frames, target_frames, context, expected in (
                (124, 22, 5, 5), (124, 39, 22, 22),
                (124, 56, 40, 39), (22, 124, 56, 22)):
            with self.subTest(previous=previous_frames, target=target_frames, context=context):
                clones = []
                output = guide([[object(), {}]], latent(target_frames, clones),
                               latent(previous_frames, clones), context)
                self.assertEqual(output.result[1], expected)
                self.assertLess(output.result[1], target_frames)
                self.assertEqual(clones, ["video", "audio"])

    def test_default_context_retains_input_conditioning_and_metadata(self):
        clones = []
        embedding, reference = object(), object()
        metadata = {"reference": reference}
        positive = [[embedding, metadata]]
        source, target = latent(124, clones), latent(141, clones)
        source_shapes = tuple(stream.shape for stream in source["samples"].tensors)
        result, trim = guide(positive, target, source).result
        self.assertEqual(trim, 22)
        self.assertIs(result[0][0], embedding)
        self.assertIs(result[0][1]["reference"], reference)
        self.assertIsNot(result[0][1], metadata)
        self.assertEqual(metadata, {"reference": reference})
        keyframes = result[0][1]["minimax_keyframes"]
        self.assertEqual(keyframes[0]["latent"].shape, (1, 24, 7, 2, 2))
        self.assertEqual(keyframes[1]["audio_latent"].shape, (1, 32, 2, 40))
        self.assertEqual(tuple(stream.shape for stream in source["samples"].tensors), source_shapes)
        self.assertEqual(clones, ["video", "audio"])


if __name__ == "__main__":
    unittest.main(verbosity=2)

