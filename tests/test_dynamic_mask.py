import gc
import importlib.util
import sys
import unittest
import weakref
from pathlib import Path
from types import SimpleNamespace

import torch

COMFY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(COMFY_ROOT))
sys.argv = [sys.argv[0], "--cpu"]

import comfy.model_base
import comfy.model_patcher
import comfy.sampler_helpers
import comfy.samplers
import comfy.utils
from comfy.nested_tensor import NestedTensor

spec = importlib.util.spec_from_file_location("h3_dynamic_test", Path(__file__).resolve().parents[1] / "__init__.py")
package = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = package
spec.loader.exec_module(package)
dynamic = sys.modules[spec.name + ".dynamic_mask"]


def latent(frames, dtype=torch.float32):
    return {"samples": NestedTensor((torch.arange(24 * ((frames - 5) // 17 * 5 + 2) * 6, dtype=torch.float32).reshape(
        1, 24, (frames - 5) // 17 * 5 + 2, 2, 3).to(dtype) / 1000,
        torch.arange(32 * 2 * round(frames * 5 / 3), dtype=torch.float32).reshape(
        1, 32, 2, round(frames * 5 / 3)).to(dtype) / 1000))}


def model():
    # Instantiate the native mask/inpaint owner without allocating H3 weights.
    core = comfy.model_base.MiniMaxH3.__new__(comfy.model_base.MiniMaxH3)
    torch.nn.Module.__init__(core)
    core.diffusion_model = torch.nn.Module()
    core.diffusion_model.patch_size = (1, 2, 2)
    core.model_sampling = SimpleNamespace(audio_scale=1., shift=3., audio_shift=3.)
    core.latent_shapes = None
    return comfy.model_patcher.ModelPatcher(core, torch.device("cpu"), torch.device("cpu"), size=1)


def prepare(patcher, source_frames=124, context=22, dtype=torch.float32, sigmas=None):
    source, target = latent(source_frames, dtype), latent(141, dtype)
    for stream in target["samples"].tensors:
        stream.zero_()
    positive = [[torch.zeros(1, 2, 4), {"minimax_refs": object(), "minimax_keyframes": [
        {"resolved_frame_index": 0, "latent": torch.ones(1, 24, 2, 2, 3)},
        {"resolved_frame_index": 140, "latent": torch.ones(1, 24, 2, 2, 3)}]}]]
    guided, _ = package.MiniMaxH3LatentContinuationGuide.execute(positive, target, source, context).result
    clone, prepared, condition = dynamic.prepare_dynamic_mask(patcher, target, guided,
        torch.tensor([1., .8, .3, 0.]) if sigmas is None else sigmas)
    return source, target, positive, guided, clone, prepared, condition


class TestDynamicMask(unittest.TestCase):
    def test_prefix_copy_dtypes_audio_alignment_and_conditions(self):
        for dtype in (torch.float16, torch.float32, torch.bfloat16):
            for frames in (124, 141, 158):
                with self.subTest(dtype=dtype, frames=frames):
                    patcher = model()
                    source, target, original, guided, clone, prepared, condition = prepare(patcher, frames, dtype=dtype)
                    video, audio = prepared["samples"].tensors
                    self.assertEqual(video.dtype, dtype)
                    self.assertEqual(audio.dtype, dtype)
                    self.assertTrue(torch.equal(video[:, :, :7], source["samples"].tensors[0][:, :, -7:]))
                    audio_guide = guided[0][1]["minimax_keyframes"][1]
                    first = round(audio_guide["resolved_frame_index"] * package.FRAME_RESCALE)
                    end = first + audio_guide["audio_latent"].shape[-1]
                    self.assertTrue(torch.equal(audio[..., :end], audio_guide["audio_latent"][..., -end:]))
                    self.assertEqual(torch.count_nonzero(video[:, :, 7:]), 0)
                    self.assertEqual(torch.count_nonzero(audio[..., end:]), 0)
                    self.assertTrue(torch.all(prepared["noise_mask"].tensors[0][:, :, :7] == 0))
                    self.assertTrue(torch.all(prepared["noise_mask"].tensors[1][..., :end] == 0))
                    self.assertIs(condition[0][1]["minimax_refs"], original[0][1]["minimax_refs"])
                    self.assertEqual(len(condition[0][1]["minimax_keyframes"]), 1)
                    self.assertEqual(condition[0][1]["minimax_keyframes"][0]["resolved_frame_index"], 140)
                    self.assertEqual(len(original[0][1]["minimax_keyframes"]), 2)
                    self.assertTrue(all(torch.count_nonzero(stream) == 0 for stream in target["samples"].tensors))
                    self.assertNotIn("denoise_mask_function", patcher.model_options)
                    self.assertIsNot(clone, patcher)
                    self.assertIs(clone.model, patcher.model)

    def test_custom_schedule_live_masks_and_native_token_labels(self):
        sigmas = torch.tensor([1., .9882, .973, .8, .1579, 0.])
        snapshot = sigmas.clone()
        patcher = model()
        *_, clone, prepared, _ = prepare(patcher, sigmas=sigmas)
        shapes = [tuple(stream.shape) for stream in prepared["samples"].tensors]
        masks = [comfy.sampler_helpers.prepare_mask(mask, shape, torch.device("cpu"))
                 for mask, shape in zip(prepared["noise_mask"].tensors, shapes)]
        packed, _ = comfy.utils.pack_latents(masks)
        original_mask = packed.clone()
        hook = clone.model_options["denoise_mask_function"]
        state = hook.__self__
        for sigma, following in ((1., .9882), (.9882, .973), (.9, .8), (.1579, 0.)):
            live = hook(torch.tensor([sigma]), packed)
            video, audio = comfy.utils.unpack_latents(live, shapes)
            self.assertAlmostEqual(float(video[0, 0, 0, 0, 0]), following / sigma, places=6)
            self.assertTrue(torch.all(video[:, :, 6] == 0), "seam must stay locked")
            self.assertTrue(torch.all(video[:, :, 7:] == 1), "new content was masked")
            self.assertTrue(torch.equal(audio, masks[1]), "audio prefix policy changed with sigma")
            kwargs = state.apply_model(lambda **kw: kw, denoise_mask=torch.ones_like(video[:, :1]))
            expected = patcher.model._denoise_mask_values(live, shapes)
            for name, values in expected.items():
                self.assertTrue(torch.equal(kwargs[name], values))
            self.assertTrue(torch.equal(kwargs["denoise_mask"] * 256, (kwargs["denoise_mask"] * 256).round()))
        self.assertTrue(torch.equal(packed, original_mask))
        self.assertTrue(torch.equal(sigmas, snapshot))
        self.assertIs(state.core, patcher.model)

    def test_native_inpaint_blending_preserves_seam_and_audio(self):
        patcher = model()
        *_, clone, prepared, _ = prepare(patcher)
        state = clone.model_options["denoise_mask_function"].__self__
        clean, shapes = comfy.utils.pack_latents(prepared["samples"].tensors)
        patcher.model.latent_shapes = shapes
        masks = [comfy.sampler_helpers.prepare_mask(mask, shape, torch.device("cpu"))
                 for mask, shape in zip(prepared["noise_mask"].tensors, shapes)]
        packed, _ = comfy.utils.pack_latents(masks)
        seen = []
        class Predictor:
            inner_model = patcher.model
            def __call__(self, x, sigma, **kwargs):
                def predict(**mask_kwargs):
                    expected = patcher.model._denoise_mask_values(state.current_mask, shapes)
                    self_outer.assertTrue(all(torch.equal(mask_kwargs[name], value) for name, value in expected.items()))
                    seen.append(mask_kwargs)
                    return torch.full_like(x, .7)
                return state.apply_model(predict)
        self_outer = self
        inpaint = comfy.samplers.KSamplerX0Inpaint(Predictor(), torch.tensor([1., .8, .3, 0.]))
        inpaint.latent_image, inpaint.noise = clean, torch.zeros_like(clean)
        for sigma in (1., .8, .3):
            result = inpaint(torch.randn_like(clean), torch.tensor([sigma]), packed, clone.model_options)
            video, audio = comfy.utils.unpack_latents(result, shapes)
            original_video, original_audio = prepared["samples"].tensors
            self.assertTrue(torch.equal(video[:, :, 6], original_video[:, :, 6]))
            audio_end = int((masks[1][0, 0, 0] == 0).sum())
            self.assertTrue(torch.equal(audio[..., :audio_end], original_audio[..., :audio_end]))
            self.assertTrue(torch.all(video[:, :, 7:] == .7))
        self.assertEqual(len(seen), 3)

    def test_cpu_tail_and_full_source_produce_identical_dynamic_inputs(self):
        patcher, source, target = model(), latent(158), latent(141)
        compact = package._copy_context_tail(source, 22)
        conditions = [[torch.zeros(1, 2, 4), {}]]
        results = []
        for previous in (source, compact):
            guided, _ = package.MiniMaxH3LatentContinuationGuide.execute(conditions, target, previous, 22).result
            results.append(dynamic.prepare_dynamic_mask(patcher, target, guided, torch.tensor([1., .8, 0.]))[1])
        for key in ("samples", "noise_mask"):
            for full, tail in zip(results[0][key].tensors, results[1][key].tensors):
                self.assertTrue(torch.equal(full, tail))
        reference = weakref.ref(source["samples"].tensors[0])
        del source, previous
        gc.collect()
        self.assertIsNone(reference(), "dynamic inputs retain full predecessor storage")

    def test_support_check_rejects_missing_core_and_conflicting_hooks(self):
        with self.assertRaisesRegex(ValueError, "原生H3"):
            dynamic.check_dynamic_mask_support(None)
        patcher = model()
        dynamic.check_dynamic_mask_support(patcher)
        patcher.set_model_denoise_mask_function(lambda *args, **kwargs: None)
        with self.assertRaisesRegex(ValueError, "其他动态掩码"):
            dynamic.check_dynamic_mask_support(patcher)
        legacy = comfy.model_patcher.ModelPatcher(torch.nn.Module(), torch.device("cpu"), torch.device("cpu"), size=1)
        with self.assertRaisesRegex(ValueError, "原生H3"):
            dynamic.check_dynamic_mask_support(legacy)


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]], verbosity=2)
