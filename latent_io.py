import hashlib
import json
import os

import comfy.utils
import folder_paths
from comfy.nested_tensor import NestedTensor
from comfy_api.latest import io
from safetensors import safe_open

from . import _av_streams


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
            inputs=[io.String.Input("latent_path", display_name="LATENT 文件路径", default="",
                                    tooltip="粘贴生成信息中的路径，例如 video/H3-SGUN/H3_00001/segment_0002.h3latent [output]。上传到input的文件填相对路径。")],
            outputs=[io.Latent.Output(display_name="音视频 LATENT")],
        )

    @classmethod
    def execute(cls, latent_path):
        if not latent_path.strip():
            raise ValueError("请填写已保存的 H3 LATENT 文件路径。")
        path = folder_paths.get_annotated_filepath(latent_path.strip())
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
    def fingerprint_inputs(cls, latent_path):
        path = folder_paths.get_annotated_filepath(latent_path.strip())
        digest = hashlib.sha256()
        with open(path, "rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
