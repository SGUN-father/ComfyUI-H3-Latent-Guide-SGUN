import base64
import io
import logging

import torch
from PIL import Image
from safetensors import SafetensorError

from comfy.ldm.minimax.model import FRAME_PER_TOKEN
import comfy.model_management as mm
import comfy.model_patcher
import comfy.model_prefetch
import comfy.sd
from comfy.taesd.taesd import Block, Clamp, conv
import comfy.utils
from comfy_execution.utils import get_executing_context
import folder_paths
from server import PromptServer


# Each update ships its frames as JPEG data URLs in one websocket message, so a
# long segment at 768px could reach several MB per update. Send at most this
# many evenly spaced frames; frame_indices still covers the full timeline.
MAX_PREVIEW_FRAMES = 60

# Latent frames the frame-by-frame decoder runs at once.
DECODE_CHUNK = 8


def preview_timeline(count, limit=MAX_PREVIEW_FRAMES):
    """Return (decoded frame offsets to encode, batch index for every timeline frame)."""
    if count <= limit:
        return list(range(count)), list(range(count))
    picks = [round(index * (count - 1) / (limit - 1)) for index in range(limit)]
    timeline = [round(frame * (limit - 1) / (count - 1)) for frame in range(count)]
    return picks, timeline


class FramewiseDecoder(torch.nn.Sequential):
    """The frame-by-frame taeh3: a 16x image decoder that draws one picture per latent frame."""

    def __init__(self):
        super().__init__(
            Clamp(), conv(24, 96), torch.nn.ReLU(),
            Block(96, 96), Block(96, 96), Block(96, 96), torch.nn.Upsample(scale_factor=2), conv(96, 96, bias=False),
            Block(96, 96), Block(96, 96), Block(96, 96), torch.nn.Upsample(scale_factor=2), conv(96, 96, bias=False),
            Block(96, 64), Block(64, 64), Block(64, 64), torch.nn.Upsample(scale_factor=2), conv(64, 64, bias=False),
            Block(64, 64), Block(64, 64), torch.nn.Upsample(scale_factor=2), conv(64, 64, bias=False),
            Block(64, 64), conv(64, 3),
        )

    def decode(self, samples):
        frames = torch.cat([self(chunk) for chunk in samples[0].movedim(0, 1).split(DECODE_CHUNK)])
        return frames.movedim(0, 1)[None]


class FramewiseTAE:
    """Carries the comfy.sd.VAE attributes the preview reads for the frame-by-frame decoder."""

    def __init__(self, state):
        self.first_stage_model = FramewiseDecoder()
        self.first_stage_model.load_state_dict(state)
        self.device = mm.get_torch_device()
        self.vae_dtype = torch.float32
        self.patcher = comfy.model_patcher.ModelPatcher(self.first_stage_model, self.device, mm.unet_offload_device())

    def memory_used_decode(self, shape, dtype):
        return DECODE_CHUNK * 64 * 4 * 256 * shape[-2] * shape[-1] * mm.dtype_size(dtype)


class H3SamplingPreview:
    def __init__(self, node_id, interval=2, max_resolution=768, fps=10):
        self.node_id = str(node_id)
        self.interval = max(1, int(interval))
        self.max_resolution = max(16, min(2048, int(max_resolution)))
        self.fps = max(1, min(60, int(fps)))
        self.source_frames = 0
        self.trim_frames = 0
        self.vae = None
        self.patcher = None
        self.framewise = False
        self.failed = False
        self.closed = False
        self.segment = 0
        self.step = 0
        self.total_steps = 0
        context = get_executing_context()
        self.prompt_id = context.prompt_id if context is not None else None
        self.client_id = PromptServer.instance.client_id if PromptServer.instance else None

    def set_segment(self, total_frames, trim_frames=0):
        self.source_frames = int(total_frames)
        self.trim_frames = int(trim_frames)

    def _send(self, status, frames=None, frame_indices=None, message=None):
        if PromptServer.instance is None or self.client_id is None or self.prompt_id is None:
            return
        payload = {"node_id": self.node_id, "prompt_id": self.prompt_id,
                   "segment": self.segment, "step": self.step,
                   "total_steps": self.total_steps, "frames": frames or [],
                   "frame_indices": frame_indices or [],
                   "frame_count": self.source_frames - self.trim_frames,
                   "source_frames": self.source_frames, "trim_frames": self.trim_frames,
                   "source_fps": 24, "fps": self.fps, "status": status}
        if message:
            payload["message"] = message
        PromptServer.instance.send_sync("h3-sgun-preview", payload, self.client_id)

    def _load(self):
        if self.patcher is not None:
            return
        path = (folder_paths.get_full_path("vae_approx", "taeh3_temporal.safetensors")
                or folder_paths.get_full_path("vae_approx", "taeh3.safetensors"))
        if path is None:
            raise FileNotFoundError("未找到 models/vae_approx/taeh3_temporal.safetensors 或 taeh3.safetensors")
        state = comfy.utils.load_torch_file(path, safe_load=True)
        if ("decoder.1.weight" in state and "decoder.22.bias" in state
                and state["decoder.1.weight"].shape[:2] == (256, 24)
                and state["decoder.22.bias"].shape == (12,)):
            self.vae = comfy.sd.VAE(sd=state)
        elif ("1.weight" in state and "23.bias" in state
                and state["1.weight"].shape[:2] == (96, 24)
                and state["23.bias"].shape == (3,)):
            self.vae = FramewiseTAE(state)
            self.framewise = True
        else:
            raise ValueError("采样预览需要 H3 的 taeh3 权重（时序版或逐帧二维版）")
        self.patcher = self.vae.patcher

    def _frames(self, video, model_patcher):
        self._load()
        height, width = video.shape[-2:]
        scale = min(1.0, self.max_resolution / (16 * max(height, width)))
        size = (max(1, int(height * scale)), max(1, int(width * scale)))
        samples = video[:1]
        if size != (height, width):
            samples = torch.nn.functional.interpolate(samples, size=(video.shape[2], *size), mode="trilinear", align_corners=False)
        samples = model_patcher.model.latent_format.process_out(samples)
        memory = self.vae.memory_used_decode(samples.shape, self.vae.vae_dtype)
        # Keep the active sampler resident while loading the preview decoder.
        with comfy.model_prefetch.pause_malloc_graph():
            mm.load_models_gpu([self.patcher, model_patcher], memory_required=memory)
        mm.throw_exception_if_processing_interrupted()
        with mm.cuda_device_context(self.vae.device), comfy.model_prefetch.malloc_graph_scope(self.patcher.load_device):
            samples = samples.to(device=self.vae.device, dtype=self.vae.vae_dtype)
            decoded = self.vae.first_stage_model.decode(samples)
        del samples
        mm.throw_exception_if_processing_interrupted()
        # picture index behind every video frame; a frame-by-frame decoder draws one picture per latent frame
        owners = range(decoded.shape[2])
        if self.framewise:
            owners = [k for k in owners for _ in range(FRAME_PER_TOKEN[k % 5])]
        if not self.source_frames:
            self.source_frames = len(owners)
        if len(owners) != self.source_frames or not 0 <= self.trim_frames < self.source_frames:
            raise ValueError("采样预览帧数与 H3 视频 latent 的时长不匹配")
        frames = []
        picks, timeline = preview_timeline(self.source_frames - self.trim_frames)
        for offset in picks:
            mm.throw_exception_if_processing_interrupted()
            frame = decoded[0, :, owners[self.trim_frames + offset]].clamp(0, 1).mul(255).to(torch.uint8)
            pixels = frame.movedim(0, -1).contiguous().cpu().numpy()
            image = Image.fromarray(pixels)
            encoded = io.BytesIO()
            image.save(encoded, format="JPEG", quality=80)
            frames.append("data:image/jpeg;base64," + base64.b64encode(encoded.getvalue()).decode("ascii"))
            del frame, pixels, image
        return frames, timeline

    def update(self, segment, step, x0, total_steps, model_patcher):
        if self.closed or self.failed:
            return
        mm.throw_exception_if_processing_interrupted()
        self.segment, self.step, self.total_steps = int(segment), int(step) + 1, int(total_steps)
        if step % self.interval and step != total_steps - 1:
            return
        try:
            video = x0.tensors[0] if x0.is_nested else x0
            if video.ndim != 5 or video.shape[1] != 24 or video.shape[2] == 0:
                raise ValueError("采样预览需要 H3 视频 latent [B, 24, T, H, W]")
            frames, frame_indices = self._frames(video, model_patcher)
            self._send("sampling", frames, frame_indices)
        except Exception as error:
            # The preview is optional: any failure (missing weights, an API this ComfyUI
            # lacks, OOM, ...) only disables it. User interrupts derive from BaseException
            # (mm.InterruptProcessingException) and still propagate.
            self.failed = True
            logging.warning("H3 sampling preview disabled for this execution: %r", error)
            try:
                self._send("error", message=f"采样预览已停用：{error}")
            except Exception as send_error:
                logging.warning("H3 sampling preview could not report its error: %r", send_error)
            self.close()

    def close(self):
        self.closed = True
        if self.patcher is not None:
            patcher, self.patcher, self.vae = self.patcher, None, None
            try:
                patcher.detach()
                mm.cleanup_models()
            except Exception as error:
                logging.warning("H3 sampling preview cleanup failed: %r", error)

    def finish(self):
        if self.closed:
            return
        try:
            self._send("complete")
        except Exception as error:
            logging.warning("H3 sampling preview could not report completion: %r", error)
        finally:
            self.close()
