import torch

from comfy.nested_tensor import NestedTensor

from . import _av_streams


def _channel_stats(video, start, end):
    means, stds = [], []
    # Reduce one channel at a time to avoid a full float32 latent copy.
    for channel in range(video.shape[1]):
        variance, mean = torch.var_mean(video[:, channel, start:end].detach().float(), correction=0)
        means.append(mean)
        stds.append(variance.sqrt())
    return torch.stack(means).cpu(), torch.stack(stds).cpu()


def drift_reference(latent, context_frames):
    video, _, _ = _av_streams(latent, "防漂移参考 LATENT")
    steps = (context_frames - 5) // 17 * 5 + 2
    start, end = steps, video.shape[2] - steps
    if end <= start:
        start, end = 0, video.shape[2]
    return _channel_stats(video, start, end)


def correct_video_drift(latent, reference, trim_frames, context_frames):
    video, audio, _ = _av_streams(latent, "防漂移 LATENT")
    start = (trim_frames - 5) // 17 * 5 + 2 if trim_frames else 0
    tail = (context_frames - 5) // 17 * 5 + 2
    end = video.shape[2] - tail
    length = end - start
    if length < 3:
        return latent

    means, stds = _channel_stats(video, start, end)
    means, stds = means.to(video.device), stds.to(video.device)
    target_means, target_stds = (stat.to(video.device) for stat in reference)
    gain = (target_stds / stds.clamp(min=1e-6)).clamp(0.8, 1.25)
    shift = (target_means - means * gain).clamp(-0.5 * stds, 0.5 * stds)
    gain = torch.where(stds > 1e-6, gain, 1.0)
    shift = torch.where(stds > 1e-6, shift, 0.0)

    # Fade within the body; copied head and future continuation tail stay exact.
    ramp = min(5, (length - 1) // 2)
    position = torch.arange(length, device=video.device, dtype=torch.float32)
    weight = (torch.minimum(position, length - 1 - position) / ramp).clamp(max=1) * 0.35
    weight = weight.view(1, length, 1, 1)
    corrected = video.detach().clone()
    for channel in range(video.shape[1]):
        body = video[:, channel, start:end].detach().float()
        corrected[:, channel, start:end] = torch.lerp(body, body * gain[channel] + shift[channel], weight).to(video.dtype)
    out = latent.copy()
    out["samples"] = NestedTensor((corrected, audio))
    return out
