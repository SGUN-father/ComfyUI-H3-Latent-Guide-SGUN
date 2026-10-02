import asyncio
import hashlib
import json
import os

import comfy.utils
import folder_paths
from comfy.nested_tensor import NestedTensor
from comfy_api.latest import io
from safetensors import safe_open
from aiohttp import web

from . import _av_streams


EMPTY_LATENT_LIST = "未找到 .h3latent（请先开启保存每段 LATENT，再刷新）"
LATENT_LIST_ROUTE = "/h3-sgun/latents"


def list_h3_latents():
    files = []
    for directory_type in ("output", "input", "temp"):
        root = folder_paths.get_directory_by_type(directory_type)
        if not root:
            continue
        for directory, subdirs, names in os.walk(root, followlinks=False):
            subdirs[:] = [name for name in subdirs
                          if folder_paths.is_within_directory(root, os.path.join(directory, name))]
            for name in names:
                if not name.lower().endswith(".h3latent"):
                    continue
                path = os.path.join(directory, name)
                if not folder_paths.is_within_directory(root, path):
                    continue
                try:
                    modified = os.stat(path).st_mtime_ns
                except FileNotFoundError:
                    continue  # 文件可能在扫描时被删除。
                relative = os.path.relpath(path, root).replace(os.sep, "/")
                files.append((modified, f"{relative} [{directory_type}]"))
    files.sort(key=lambda item: (-item[0], item[1]))
    return [name for _, name in files] or [EMPTY_LATENT_LIST]


async def get_h3_latents(request):
    return web.json_response(await asyncio.to_thread(list_h3_latents))


def selected_latent_path(latent_path, latent_file):
    selected = latent_path.strip() or latent_file.strip()
    if not selected or selected == EMPTY_LATENT_LIST:
        raise ValueError("请先开启主节点的保存每段 LATENT，生成完成后刷新列表并选择文件。")
    return folder_paths.get_annotated_filepath(selected)


def save_h3_latent(latent, path, segment_info):
    video, audio, _ = _av_streams(latent, "待保存 LATENT")
    metadata = {"h3_latent_format": "joint_av_v1", "segment_info": json.dumps(segment_info, ensure_ascii=False)}
    if "_h3_audio_overhang" in latent:
        metadata["h3_audio_overhang"] = str(latent["_h3_audio_overhang"])
    temporary = path + ".tmp"
    try:
        comfy.utils.save_torch_file({"video": video.detach().cpu().contiguous(),
                                     "audio": audio.detach().cpu().contiguous()}, temporary, metadata=metadata)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


class H3LoadLatent(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3LoadLatent", display_name="H3 读取音视频 LATENT by神棍", category="video/minimax",
            description="读取本插件保存的 .h3latent，保留音视频张量的原始数值和dtype。路径限ComfyUI input/output/temp目录。",
            inputs=[io.String.Input("latent_path", display_name="手动路径（可留空）", default="", advanced=True,
                                    tooltip="通常留空，直接使用下方文件列表。填写或连接路径时优先读取该路径，兼容旧工作流。"),
                    io.Combo.Input("latent_file", display_name="已保存的 LATENT", options=list_h3_latents(), optional=True,
                                   remote=io.RemoteOptions(route=LATENT_LIST_ROUTE, refresh_button=True,
                                                           control_after_refresh="first"),
                                   tooltip="自动扫描 output/input/temp 及子目录，最新文件在最前面。保存新片段后点击刷新按钮。")],
            outputs=[io.Latent.Output(display_name="音视频 LATENT")],
        )

    @classmethod
    def execute(cls, latent_path="", latent_file=""):
        path = selected_latent_path(latent_path, latent_file)
        with safe_open(path, framework="pt", device="cpu") as file:
            metadata = file.metadata() or {}
            if metadata.get("h3_latent_format") != "joint_av_v1":
                raise ValueError("请选择本插件保存的 .h3latent；普通单流 LATENT 不包含完整H3音视频。")
            latent = {"samples": NestedTensor((file.get_tensor("video"), file.get_tensor("audio")))}
            if "h3_audio_overhang" in metadata:
                latent["_h3_audio_overhang"] = float(metadata["h3_audio_overhang"])
        _av_streams(latent, "加载的 LATENT")
        return io.NodeOutput(latent)

    @classmethod
    def validate_inputs(cls, latent_file=""):
        # 远程列表会变化；手动路径优先，最终路径与格式由 execute 检查。
        return True

    @classmethod
    def fingerprint_inputs(cls, latent_path="", latent_file=""):
        path = selected_latent_path(latent_path, latent_file)
        digest = hashlib.sha256()
        with open(path, "rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
