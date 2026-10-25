"""Check that the running ComfyUI provides every official API this node pack needs.

The checks look for the features themselves, not a version number, so custom
builds and the desktop app are judged by what they actually ship.
"""

import importlib
import inspect
import logging

MINIMUM = "2026-09-21"
MINIMUM_RELEASE = "v0.38.0"


def _has(module, *names):
    try:
        root = importlib.import_module(module)
    except Exception:
        return False
    for name in names:
        # Every dotted name is looked up from the module itself.
        target = root
        try:
            for part in name.split("."):
                target = getattr(target, part)
        except AttributeError:
            return False
    return True


def _video_from_list_takes_audio():
    try:
        from comfy_api.latest import InputImpl
        return "complete_audio" in inspect.signature(InputImpl.VideoFromList).parameters
    except Exception:
        return False


# (what is missing, when official ComfyUI added it, check)
REQUIREMENTS = (
    ("MiniMax H3 模型 / MiniMax H3 model (comfy.ldm.minimax)", "2026-08-02",
     lambda: _has("comfy.ldm.minimax.model", "FRAME_RESCALE")),
    ("MiniMax H3 官方节点 / official H3 nodes (comfy_extras.nodes_minimax_h3)", "2026-08-02",
     lambda: _has("comfy_extras.nodes_minimax_h3", "EmptyMiniMaxH3LatentAV",
                  "MiniMaxH3ReferenceToVideo", "align_frame_count")),
    ("联合音视频张量 / NestedTensor (comfy.nested_tensor)", "2026-08-02",
     lambda: _has("comfy.nested_tensor", "NestedTensor")),
    ("视频拼接 / VideoFromList(complete_audio)", "2026-09-11", _video_from_list_takes_audio),
    ("采样预览显存接口 / comfy.model_prefetch.malloc_graph_scope", "2026-09-17",
     lambda: _has("comfy.model_prefetch", "pause_malloc_graph", "malloc_graph_scope")),
    ("音频重采样 / comfy.audio.resample", "2026-09-21",
     lambda: _has("comfy.audio", "resample")),
    ("路径安全检查 / folder_paths.is_within_directory", "2026-07-02",
     lambda: _has("folder_paths", "is_within_directory", "get_directory_by_type")),
    ("Autogrow 动态输入 / io.Autogrow", "2026-06-01",
     lambda: _has("comfy_api.latest", "io.Autogrow.TemplatePrefix", "io.RemoteOptions")),
)


def missing_requirements():
    return [(name, added) for name, added, check in REQUIREMENTS if not check()]


def check_comfyui():
    """Raise ImportError with a readable message when ComfyUI is too old."""
    missing = missing_requirements()
    if not missing:
        return
    items = "\n".join(f"  - {name}（官方 {added} 加入）" for name, added in missing)
    message = (
        f"ComfyUI 版本过旧，H3-Latent-Guide-SGUN 需要 {MINIMUM} 之后的 ComfyUI"
        f"（稳定版 {MINIMUM_RELEASE} 及以上）。请更新 ComfyUI 后重启。\n"
        f"ComfyUI is too old: H3-Latent-Guide-SGUN requires ComfyUI from {MINIMUM} or later"
        f" (stable {MINIMUM_RELEASE}+). Please update ComfyUI and restart.\n"
        f"缺少 / Missing:\n{items}"
    )
    logging.error("[H3-Latent-Guide-SGUN] %s", message)
    raise ImportError(message)
