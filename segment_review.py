import asyncio
import hashlib
import json
import os
import re
import sys

import folder_paths
from aiohttp import web
from server import PromptServer

try:
    if os.name == "nt":
        from send2trash.win.modern import send2trash
    else:
        from send2trash import send2trash
except ImportError:
    send2trash = None

from .latent_io import load_h3_latent


RECORD_FORMAT = "h3_sgun_segment_v1"
RECORD_SUFFIX = ".h3segment.json"
SEGMENTS_ROUTE = "/h3-sgun/segments"
SEGMENT_ROUTE = "/h3-sgun/segment"
SEGMENT_CLEANUP_ROUTE = "/h3-sgun/segments/cleanup-records"
cleanup_revision = 0


def annotated_path(path, directory_type):
    root = folder_paths.get_directory_by_type(directory_type)
    if not root or not folder_paths.is_within_directory(root, path):
        raise ValueError("片段文件必须位于 ComfyUI input/output/temp 目录内。")
    return f"{os.path.relpath(path, root).replace(os.sep, '/')} [{directory_type}]"


def save_record(record, path):
    temporary = path + ".tmp"
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(record, handle, ensure_ascii=False, allow_nan=False)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def load_record(recipe_path):
    path = folder_paths.get_annotated_filepath(recipe_path)
    if not path.lower().endswith(RECORD_SUFFIX):
        raise ValueError("请选择本插件生成的 .h3segment.json 片段记录。")
    with open(path, encoding="utf-8") as handle:
        record = json.load(handle)
    if (not isinstance(record, dict) or record.get("format") != RECORD_FORMAT
            or not isinstance(record.get("prompt"), str)
            or not isinstance(record.get("settings"), dict)
            or record.get("sigmas_source") not in ("internal", "external")
            or not isinstance(record.get("sigmas"), list)
            or "context" not in record):
        raise ValueError("片段记录不完整；旧版本只保存 LATENT，不能自动恢复重抽参数。")
    return record


def _view_file(name):
    if not name:
        return None
    path = folder_paths.get_annotated_filepath(name)
    if not os.path.isfile(path):
        return None
    _, base = folder_paths.annotated_filepath(name)
    kind = next((kind for kind in ("output", "input", "temp")
                 if base == folder_paths.get_directory_by_type(kind)), "input")
    root = folder_paths.get_directory_by_type(kind)
    relative = os.path.relpath(path, root)
    return {"filename": os.path.basename(relative),
            "subfolder": os.path.dirname(relative).replace(os.sep, "/"), "type": kind}


def record_entry(record, recipe_path):
    context = record["context"]
    available = not context or os.path.isfile(folder_paths.get_annotated_filepath(context))
    return {**{key: record[key] for key in ("segment", "seed", "width", "height", "frames")},
            "recipe_path": recipe_path, "context_available": available,
            "seed_text": str(record["seed"]),
            "video": _view_file(record.get("video_path")),
            "seam": _view_file(record.get("seam_path")),
            "diagnostics": record.get("diagnostics")}


def list_segment_records():
    candidates = []
    for kind in ("output", "input", "temp"):
        root = folder_paths.get_directory_by_type(kind)
        if not root:
            continue
        for directory, subdirs, names in os.walk(root, followlinks=False):
            subdirs[:] = [name for name in subdirs
                          if folder_paths.is_within_directory(root, os.path.join(directory, name))]
            for name in names:
                if not name.lower().endswith(RECORD_SUFFIX):
                    continue
                path = os.path.join(directory, name)
                if folder_paths.is_within_directory(root, path):
                    try:
                        candidates.append((os.stat(path).st_mtime_ns, annotated_path(path, kind)))
                    except FileNotFoundError:
                        continue
    entries = []
    for _, name in sorted(candidates, reverse=True):
        try:
            entries.append(record_entry(load_record(name), name))
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return entries


def load_retry_context(record):
    context = record["context"]
    if not context:
        return None
    path = folder_paths.get_annotated_filepath(context)
    if not os.path.isfile(path):
        raise ValueError("这段的重抽上下文已被清理；请使用保存在 output 中的片段记录，或重新生成前段。")
    return load_h3_latent(path)


def record_fingerprint(recipe_path):
    record = load_record(recipe_path)
    digest = hashlib.sha256()
    for name in (recipe_path, record["context"]):
        if name:
            with open(folder_paths.get_annotated_filepath(name), "rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
    return digest.hexdigest()


def _cleanup_path(path, root):
    path, root = os.path.abspath(path), os.path.abspath(root)
    expected = os.path.normpath(os.path.join(os.path.realpath(root), os.path.relpath(path, root)))
    if (not folder_paths.is_within_directory(root, path) or path == root
            or os.path.normcase(os.path.realpath(path)) != os.path.normcase(expected)):
        raise ValueError("只能回收 ComfyUI 输出或临时目录中的片段文件，不允许目录越界或符号链接。")
    return os.path.realpath(path)


def _recycle_files(paths):
    paths = [path for path in paths if os.path.lexists(path)]
    if not paths:
        return
    send2trash(paths)
    if any(os.path.lexists(path) for path in paths):
        raise OSError("系统回收站未接收全部文件，未完成部分已保留。")


def recycle_segment_records(paths):
    global cleanup_revision
    if not isinstance(paths, list) or any(not isinstance(path, str) or not path for path in paths):
        raise ValueError("请提供要清理的片段记录列表。")
    if send2trash is None or (os.name == "nt" and sys.getwindowsversion().major < 10):
        raise RuntimeError("系统回收站不可用，请安装节点 requirements.txt 中的依赖；未删除任何文件。")
    plan, batches, selected = [], set(), set()
    skipped = 0
    for name in dict.fromkeys(paths):
        kind = "output" if name.endswith("[output]") else "temp" if name.endswith("[temp]") else "input"
        if kind == "input":
            skipped += 1
            continue
        relative, root = folder_paths.annotated_filepath(name)
        path = _cleanup_path(os.path.join(root, relative), root)
        match = re.fullmatch(r"(segment_\d{4,}(?:_retry_\d{3,})?)\.h3segment\.json", os.path.basename(path))
        if not match:
            raise ValueError("只能清理本节点生成的片段记录。")
        if path in selected:
            continue
        if not os.path.isfile(path):
            continue
        record = load_record(name)
        stem, record_folder = match[1], os.path.dirname(path)
        batch = os.path.dirname(record_folder) if os.path.basename(record_folder) == "_data" else record_folder
        suffix = stem.removeprefix("segment_")
        expected = {"video_path": os.path.join(batch, f"{stem}.mp4"),
                    "context": os.path.join(record_folder, f"{stem}.h3context"),
                    "seam_path": os.path.join(record_folder, f"seam_{suffix}.mp4"),
                    "context_video_path": os.path.join(record_folder, f"context_preview_{suffix}.mp4")}
        expected = {field: _cleanup_path(target, root) for field, target in expected.items()}
        for field, target in expected.items():
            reference = record.get(field)
            if reference:
                actual = folder_paths.get_annotated_filepath(reference)
                if field == "video_path" and os.path.normcase(actual) != os.path.normcase(os.path.realpath(target)):
                    raise ValueError("片段视频路径与记录不匹配，未清理任何文件。")
        # Formal segment videos remain user output after retry records are cleared.
        candidates = {target for field, target in expected.items() if field != "video_path"}
        candidates |= {os.path.join(record_folder, f"{stem}.h3latent"),
                       os.path.join(record_folder, f"{stem}.reserve")}
        candidates |= {candidate + ".tmp" for candidate in candidates} | {path + ".tmp"}
        candidates = {_cleanup_path(candidate, root) for candidate in candidates}
        selected.add(path)
        plan.append((path, candidates))
        batches.add((batch, record_folder, root))
    protected = set()
    for entry in list_segment_records():
        path = folder_paths.get_annotated_filepath(entry["recipe_path"])
        if path in selected:
            continue
        record = load_record(entry["recipe_path"])
        protected.add(path)
        for key in ("video_path", "context", "seam_path", "context_video_path"):
            if record.get(key):
                protected.add(folder_paths.get_annotated_filepath(record[key]))
    protected = {os.path.normcase(path) for path in protected}
    existing = {path for recipe, files in plan for path in files | {recipe} if os.path.isfile(path)}
    errors, recycled = [], 0
    shared = {path for _, files in plan for path in files if os.path.normcase(path) in protected}
    for recipe, files in plan:
        if os.path.normcase(recipe) in protected:
            skipped += 1
            continue
        try:
            _recycle_files(sorted(path for path in files if os.path.normcase(path) not in protected))
            _recycle_files([recipe])
            recycled += 1
        except OSError as error:
            errors.append(f"{os.path.basename(recipe)}: {error}")
    for batch, record_folder, root in batches:
        directories = {batch, record_folder, os.path.join(batch, "_data")}
        try:
            if not any(name.endswith(RECORD_SUFFIX) for directory in directories if os.path.isdir(directory)
                       for name in os.listdir(directory)):
                manifests = [_cleanup_path(os.path.join(directory, "segments.jsonl"), root)
                             for directory in directories if os.path.isdir(directory)]
                existing.update(path for path in manifests if os.path.isfile(path))
                _recycle_files(manifests)
            for directory in sorted(directories, key=len, reverse=True):
                if (os.path.realpath(directory) != os.path.realpath(root) and os.path.isdir(directory)
                        and not os.listdir(directory)):
                    _recycle_files([_cleanup_path(directory, root)])
        except (OSError, ValueError) as error:
            errors.append(str(error))
    recycled_files = sum(not os.path.exists(path) for path in existing)
    if recycled_files:
        cleanup_revision += 1
    return {"recycled_records": recycled, "recycled_files": recycled_files,
            "skipped_records": skipped, "retained_shared_files": len(shared), "errors": errors,
            "segments": list_segment_records()}


async def cleanup_segments(request):
    try:
        body = await request.json()
        if not isinstance(body, dict) or "paths" not in body:
            raise ValueError("请提供要清理的片段记录列表。")

        def when_idle():
            queue = PromptServer.instance.prompt_queue
            with queue.mutex:
                if queue.get_tasks_remaining():
                    return None
                return recycle_segment_records(body["paths"])

        result = await asyncio.to_thread(when_idle)
        if result is None:
            return web.json_response({"error": "任务运行或排队期间不能清理片段，请等待完成。"}, status=409)
        return web.json_response(result)
    except RuntimeError as error:
        return web.json_response({"error": str(error)}, status=503)
    except (OSError, ValueError, TypeError, KeyError) as error:
        return web.json_response({"error": str(error)}, status=400)


async def get_segments(request):
    return web.json_response({"segments": await asyncio.to_thread(list_segment_records)})


async def get_segment(request):
    try:
        name = request.query.get("path", "")
        record = await asyncio.to_thread(load_record, name)
        entry = await asyncio.to_thread(record_entry, record, name)
        return web.json_response({**record, **entry})
    except (OSError, ValueError, KeyError, TypeError) as error:
        return web.json_response({"error": str(error)}, status=400)
