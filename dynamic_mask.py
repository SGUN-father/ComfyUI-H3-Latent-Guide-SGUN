import torch

import comfy.utils
from comfy.nested_tensor import NestedTensor
from comfy.patcher_extension import WrappersMP

from . import FRAME_RESCALE, _av_streams


def check_dynamic_mask_support(model):
    if (model is None or not callable(getattr(model, "clone", None))
            or not callable(getattr(model, "set_model_denoise_mask_function", None))
            or not callable(getattr(model, "add_wrapper_with_key", None))
            or not callable(getattr(model.model, "_denoise_mask_values", None))):
        raise ValueError("动态掩码需要带原生H3音视频掩码支持的ComfyUI和MODEL；请更新，或关闭动态掩码。")
    if model.model_options.get("denoise_mask_function") is not None:
        raise ValueError("MODEL已安装其他动态掩码（如Differential Diffusion）；请移除它，或关闭本节点动态掩码。")


class H3DynamicMask:
    def __init__(self, core, shapes, prefix_steps, sigmas):
        self.core = core
        self.shapes = shapes
        self.prefix_steps = prefix_steps
        self.sigmas = tuple(sigmas.detach().reshape(-1).cpu().tolist())
        self.current_mask = None

    def update(self, sigma, denoise_mask, extra_options=None):
        current = float(sigma.reshape(-1)[0])
        following = next((value for value in self.sigmas
                          if value < current - max(1e-7, abs(current) * 1e-6)), 0.0)
        release = min(1.0, max(0.0, following / current)) if current > 0 else 0.0
        output = denoise_mask.clone()
        video, _ = comfy.utils.unpack_latents(output, self.shapes)
        # Repaint the disposable prefix more freely away from the locked seam.
        weights = torch.linspace(release, 0.0, self.prefix_steps, device=video.device, dtype=video.dtype)
        video[:, :, :self.prefix_steps] = weights.view(1, 1, -1, 1, 1)
        self.current_mask = output
        return output

    def apply_model(self, executor, *args, **kwargs):
        if self.current_mask is not None:
            # Use the same native token-grid masks as scale_latent_inpaint.
            kwargs.update(self.core._denoise_mask_values(self.current_mask, self.shapes))
        return executor(*args, **kwargs)


def prepare_dynamic_mask(model, latent, positive, sigmas):
    video, audio, _ = _av_streams(latent, "本段 LATENT")
    video_guide, audio_guide = positive[0][1]["minimax_keyframes"][:2]
    prefix = video_guide["latent"]
    prefix_steps = prefix.shape[2]
    result_video, result_audio = video.clone(), audio.clone()
    result_video[:, :, :prefix_steps] = prefix.to(device=video.device, dtype=video.dtype)
    audio_context = audio_guide["audio_latent"]
    audio_start = round(audio_guide["resolved_frame_index"] * FRAME_RESCALE)
    audio_end = min(audio.shape[-1], audio_start + audio_context.shape[-1])
    start = max(0, audio_start)
    result_audio[..., start:audio_end] = audio_context[..., start - audio_start:audio_end - audio_start].to(
        device=audio.device, dtype=audio.dtype)
    video_mask = torch.ones_like(video[:, :1], dtype=torch.float32)
    audio_mask = torch.ones_like(audio[:, :1], dtype=torch.float32)
    video_mask[:, :, :prefix_steps] = 0
    audio_mask[..., :audio_end] = 0
    prepared = latent.copy()
    prepared["samples"] = NestedTensor((result_video, result_audio))
    prepared["noise_mask"] = NestedTensor((video_mask, audio_mask))
    conditioning = []
    for embedding, metadata in positive:
        metadata = metadata.copy()
        metadata["minimax_keyframes"] = metadata["minimax_keyframes"][2:]
        conditioning.append([embedding, metadata])
    patched = model.clone()
    state = H3DynamicMask(patched.model, [tuple(video.shape), tuple(audio.shape)], prefix_steps, sigmas)
    patched.set_model_denoise_mask_function(state.update)
    patched.add_wrapper_with_key(WrappersMP.APPLY_MODEL, "h3_sgun_dynamic_mask", state.apply_model)
    return patched, prepared, conditioning
