import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { createDirectorPanel } from "./director_panel.js";
import { t, onLocaleChange } from "./i18n.js";

const settingNames = ["duration", "aspect_ratio", "megapixels", "steps", "sampler_name",
  "scheduler", "context_frames", "denoise", "ref_image_size", "dynamic_mask", "drift_correction"];
const nativePreviewWidgets = new Set(["$$canvas-image-preview", "$$comfy_animation_preview", "video-preview"]);

function suppressNativePreview() {
  this.hideOutputImages = true;
  this.imgs = undefined;
  let removed = false;
  for (const widget of [...(this.widgets ?? [])]) {
    if (!nativePreviewWidgets.has(widget.name)) continue;
    if (this.removeWidget) this.removeWidget(widget);
    else {
      widget.onRemove?.();
      this.widgets.splice(this.widgets.indexOf(widget), 1);
    }
    removed = true;
  }
  if (removed) this._h3SgunSegmentReview?.resize();
}

async function getJson(path, options) {
  const response = await api.fetchApi(path, options);
  const data = await response.json();
  if (!response.ok) {
    if (data.error) throw new Error(data.error);
    throw localizedError(`片段操作失败（${response.status}）`, `Segment operation failed (${response.status}).`);
  }
  return data;
}

function localizedError(zh, en) {
  const error = new Error(t(zh, en));
  error.h3Message = [zh, en];
  return error;
}

function diagnosticMessage(message) {
  const known = {
    "大动作或换光也可能触发，请结合预览判断。": "Large movements or lighting changes can also trigger this hint. Check the preview.",
    "前段视频不可用，本段接缝尚未检查；重抽上下文仍已保存。": "The previous video is unavailable, so this seam has not been checked. Retry context was saved.",
  };
  if (known[message]) return t(message, known[message]);
  const event = /^新段 ([\d.-]+) 秒处(疑似画面跳变|疑似亮度或颜色台阶)。$/.exec(message);
  if (event) return t(message, `${event[2] === "疑似画面跳变" ? "Possible visual jump" : "Possible brightness or color step"} at ${event[1]} s in the new segment.`);
  if (message.startsWith("接缝预览未完成：")) return t(message, `Seam preview could not finish: ${message.slice("接缝预览未完成：".length)}`);
  return message;
}

function createPanel(node, nodeData) {
  const director = createDirectorPanel(node, nodeData);
  const select = document.createElement("select");
  select.style.cssText = "width:100%;margin-bottom:6px";
  const previewSelect = document.createElement("select");
  previewSelect.style.cssText = select.style.cssText;
  const buttons = document.createElement("div");
  buttons.style.cssText = "display:flex;gap:4px;flex-wrap:wrap;margin-bottom:6px";
  const refresh = document.createElement("button");
  const restore = document.createElement("button");
  const exit = document.createElement("button");
  const cleanup = document.createElement("button");
  for (const button of [refresh, restore, exit, cleanup]) {
    button.type = "button";
    buttons.append(button);
  }
  const status = document.createElement("div");
  status.setAttribute("role", "status");
  status.setAttribute("aria-live", "polite");
  status.style.cssText = "white-space:pre-wrap;margin-bottom:6px";
  const details = document.createElement("div");
  details.style.cssText = "white-space:pre-wrap;margin-bottom:6px";
  const video = document.createElement("video");
  video.controls = true;
  video.playsInline = true;
  video.preload = "metadata";
  video.style.cssText = "width:100%;height:220px;object-fit:contain;background:#111";
  const help = document.createElement("div");
  help.style.cssText = "opacity:.8;margin-top:6px";
  director.sections.retry.prepend(select, buttons, status);
  director.sections.retry.append(help);
  director.sections.preview.prepend(previewSelect, details, video);
  const info = document.createElement("pre");
  info.style.cssText = "white-space:pre-wrap;overflow-wrap:anywhere;margin:0;font:12px/1.6 monospace";
  director.sections.info.append(info);

  let entries = [];
  let beforeRetry = null;
  let busy = false;
  let queueBusy = false;
  let removed = false;
  let statusMessage = ["", ""];
  let hasInfo = false;

  function setStatus(zh, en) {
    statusMessage = [zh, en];
    status.textContent = t(zh, en);
  }

  function entryLabel(entry) {
    return t(`第 ${entry.segment} 段 · ${entry.seed_text ?? entry.seed} · ${entry.recipe_path}`,
      `Segment ${entry.segment} · ${entry.seed_text ?? entry.seed} · ${entry.recipe_path}`);
  }

  function updateButtons() {
    select.disabled = busy || !entries.length;
    previewSelect.disabled = busy || !entries.length;
    refresh.disabled = busy;
    restore.disabled = busy || !entries.length;
    exit.disabled = busy || !node.widgets.find((item) => item.name === "retry_segment")?.value;
    cleanup.disabled = busy || queueBusy || !entries.length;
  }

  function onStatus(event) {
    queueBusy = event.detail == null || Number(event.detail.exec_info?.queue_remaining ?? 0) > 0;
    updateButtons();
  }
  api.addEventListener("status", onStatus);

  function showSelection(updateVideo = true) {
    const entry = entries.find((item) => item.recipe_path === select.value);
    if (updateVideo) {
      video.pause();
      video.removeAttribute("src");
      video.hidden = true;
      video.load();
    }
    if (!entry) {
      details.textContent = t("完成生成后可选择片段；也可刷新已保存记录。", "Select a segment after generation, or refresh saved records.");
      director.setSummary(t("尚无片段记录", "No segment records yet"));
      return;
    }
    const diagnostics = entry.diagnostics;
    const lines = [t(`第 ${entry.segment} 段 · 种子 ${entry.seed_text ?? entry.seed} · ${entry.width}×${entry.height} · ${entry.frames} 帧`,
      `Segment ${entry.segment} · Seed ${entry.seed_text ?? entry.seed} · ${entry.width}×${entry.height} · ${entry.frames} frames`)];
    if (entry.seam) {
      const { filename, subfolder, type } = entry.seam;
      if (updateVideo) {
        video.src = api.apiURL(`/view?${new URLSearchParams({ filename, subfolder, type })}`);
        video.hidden = false;
      }
      if (diagnostics?.fps && Number.isFinite(diagnostics.join_frame)) {
        lines.push(t(`接缝位于预览第 ${diagnostics.join_frame} 帧（${(diagnostics.join_frame / diagnostics.fps).toFixed(2)} 秒）。`,
          `The seam is at preview frame ${diagnostics.join_frame} (${(diagnostics.join_frame / diagnostics.fps).toFixed(2)} s).`));
      }
    } else {
      lines.push(entry.segment === 1 && !diagnostics ? t("首段没有前段接缝。", "The first segment has no preceding seam.") : t("此记录没有接缝预览。", "This record has no seam preview."));
    }
    if (diagnostics) {
      const label = { ok: t("未发现明显异常", "No obvious issues detected"), warning: t("接缝可能异常", "Possible seam issue"),
        error: t("接缝检查失败", "Seam check failed"), unavailable: t("未检查", "Not checked") }[diagnostics.status]
        ?? diagnostics.status;
      if (label) lines.push(t(`接缝检查：${label}`, `Seam check: ${label}`));
      lines.push(...(diagnostics.messages ?? []).map(diagnosticMessage));
    }
    lines.push(t("异常提示仅辅助审片，请播放接缝确认。", "Hints assist review. Play the seam to confirm."));
    details.textContent = lines.join("\n");
    director.setSummary(t(`第 ${entry.segment} 段 · ${entry.width}×${entry.height} · ${entry.frames} 帧`,
      `Segment ${entry.segment} · ${entry.width}×${entry.height} · ${entry.frames} frames`));
  }

  function setEntries(next, newestFirst = false, selected = select.value) {
    entries = next;
    select.replaceChildren();
    previewSelect.replaceChildren();
    for (const entry of entries) {
      const option = document.createElement("option");
      option.value = entry.recipe_path;
      option.textContent = entryLabel(entry);
      select.append(option);
      const previewOption = document.createElement("option");
      previewOption.value = entry.recipe_path;
      previewOption.textContent = option.textContent;
      previewSelect.append(previewOption);
    }
    if (entries.some((item) => item.recipe_path === selected)) select.value = selected;
    else select.value = (newestFirst ? entries[0] : entries.at(-1))?.recipe_path ?? "";
    previewSelect.value = select.value;
    showSelection();
    updateButtons();
    director.syncWidgets();
  }

  async function perform(action) {
    if (busy) return;
    busy = true;
    updateButtons();
    try {
      await action();
    } catch (error) {
      if (!removed) setStatus(...(error.h3Message ?? [error.message, error.message]));
    } finally {
      busy = false;
      if (!removed) updateButtons();
    }
  }

  refresh.addEventListener("click", () => perform(async () => {
    setStatus("正在读取已保存片段…", "Loading saved segments…");
    const data = await getJson("/h3-sgun/segments");
    if (removed) return;
    setEntries(data.segments, true);
    if (data.segments.length) setStatus(`已载入 ${data.segments.length} 个片段记录。`, `Loaded ${data.segments.length} segment records.`);
    else setStatus("没有已保存片段。", "No saved segments.");
  }));
  restore.addEventListener("click", () => perform(async () => {
    setStatus("正在核对片段与重抽上下文…", "Checking the segment and retry context…");
    const path = select.value;
    const record = await getJson(`/h3-sgun/segment?${new URLSearchParams({ path })}`);
    if (removed) return;
    if (!record.context_available) throw localizedError("该片段缺少重抽上下文，请重新生成并保存片段记录。", "This segment has no retry context. Generate it again and save its record.");
    const values = Object.fromEntries(settingNames.map((name) => [name, record.settings?.[name]]));
    const originalSeed = beforeRetry?.find(([item]) => item.name === "seed")?.[1]
      ?? node.widgets.find((item) => item.name === "seed")?.value;
    Object.assign(values, { segments: 1, seed_mode: "fixed",
      seed: Number.isSafeInteger(record.seed) ? record.seed : originalSeed,
      [director.seedControl?.name ?? "control_after_generate"]: "fixed", retry_segment: record.recipe_path, retry_prompt: record.prompt,
      retry_seed: record.seed_text ?? String(record.seed) });
    const linked = node.inputs.filter((input) => input.link != null
      && (Object.hasOwn(values, input.name) || Object.hasOwn(values, input.widget?.name)));
    if (linked.length) throw localizedError(`无法回填：${linked.map((input) => input.name).join("、")} 由连线控制。请先转回节点控件或断开对应连线，再点击恢复。`,
      `Cannot restore: ${linked.map((input) => input.name).join(", ")} are controlled by connections. Convert them back to widgets or disconnect them, then restore again.`);
    const widgets = Object.keys(values).map((name) => node.widgets.find((item) => item.name === name));
    const seedMode = node.widgets.find((item) => item.name === "retry_seed_mode");
    const missingWidgets = Object.keys(values).filter((name, index) => !widgets[index]);
    if (!seedMode) missingWidgets.push("retry_seed_mode");
    if (missingWidgets.length) {
      throw localizedError(`当前节点缺少控件：${missingWidgets.join("、")}。请刷新页面或重新载入工作流后重试，已保存片段无需重做。`,
        `Node widgets are missing: ${missingWidgets.join(", ")}. Refresh the page or reload the workflow, then try again. Saved segments do not need to be regenerated.`);
    }
    const missingValues = Object.keys(values).filter((name) => values[name] == null);
    if (missingValues.length) {
      throw localizedError(`片段记录参数不完整：${missingValues.join("、")}。请重新生成该片段。`,
        `Segment settings are incomplete: ${missingValues.join(", ")}. Generate this segment again.`);
    }
    if (!beforeRetry) beforeRetry = [...widgets, seedMode].map((item) => [item, item.value]);
    for (const item of widgets) item.value = values[item.name];
    director.syncWidgets();
    node.setDirtyCanvas(true, true);
    setStatus(`已恢复第 ${record.segment} 段参数。按所选重抽种子方式运行，可修改重抽提示词。\n`
      + (record.sigmas_source === "external" ? "自定义 SIGMAS：重抽使用原记录序列，当前 SIGMAS 连线不参与重抽。" : "SIGMAS：按恢复的步数、调度器与降噪重新计算。"),
    `Restored settings for segment ${record.segment}. Run with the selected retry seed mode, or edit the retry prompt.\n`
      + (record.sigmas_source === "external" ? "Custom SIGMAS: retry uses the recorded sequence. The current SIGMAS connection is not used." : "SIGMAS: recalculated from the restored steps, scheduler and denoise."));
  }));
  function resetRetry() {
    const restored = beforeRetry !== null;
    if (beforeRetry) for (const [item, value] of beforeRetry) item.value = value;
    for (const item of node.widgets) if (["retry_segment", "retry_prompt", "retry_seed"].includes(item.name)) item.value = "";
    beforeRetry = null;
    director.syncWidgets();
    return restored;
  }
  exit.addEventListener("click", () => {
    if (busy) return;
    const restored = resetRetry();
    if (restored) setStatus("已退出重抽，恢复进入重抽前的参数。", "Exited retry and restored the previous settings.");
    else setStatus("已退出重抽。", "Exited retry.");
    node.setDirtyCanvas(true, true);
    updateButtons();
  });
  cleanup.addEventListener("click", () => {
    if (queueBusy || !entries.length) return;
    // The refreshed list can span every batch in output; confirm the real scope first.
    const batches = new Set(entries.map((entry) => String(entry.recipe_path)
      .replace(/\s*\[\w+\]$/, "").replace(/\/[^/]*$/, "").replace(/\/_data$/, "")));
    if (typeof globalThis.confirm === "function" && !globalThis.confirm(t(
      `将清理列表中 ${entries.length} 条片段记录（涉及 ${batches.size} 个批次）的重抽数据。片段视频和保存的 LATENT 保留。继续吗？`,
      `Clean up retry data for ${entries.length} segment records in this list (${batches.size} batches)? Segment videos and saved LATENTs are kept.`))) return;
    return perform(async () => {
      setStatus("正在将重抽记录和辅助数据移到回收站，保留片段视频…", "Moving retry records and supporting data to the Recycle Bin; keeping segment videos…");
      video.pause();
      video.removeAttribute("src");
      video.hidden = true;
      video.load();
      let data;
      try {
        data = await getJson("/h3-sgun/segments/cleanup-records", {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ paths: entries.map((entry) => entry.recipe_path) }),
        });
      } catch (error) {
        if (!removed) showSelection();
        throw error;
      }
      if (removed) return;
      const activeRetry = node.widgets.find((item) => item.name === "retry_segment")?.value;
      if (!activeRetry || !data.segments.some((entry) => entry.recipe_path === activeRetry)) resetRetry();
      setEntries(data.segments, true);
      hasInfo = false;
      info.textContent = t("生成后显示实际种子、尺寸、时长与缓存路径。", "Actual seeds, dimensions, duration and cache paths appear after generation.");
      const zh = `已将 ${data.recycled_records} 个片段记录及 ${data.recycled_files} 个辅助文件移到回收站，片段视频保留，保存的 LATENT 也保留。`;
      const en = `Moved ${data.recycled_records} segment records and ${data.recycled_files} supporting files to the Recycle Bin. Segment videos and saved LATENTs were kept.`;
      setStatus(zh + (data.skipped_records ? ` 保留 ${data.skipped_records} 条导入记录。` : "")
          + (data.errors.length ? `\n部分文件未能清理：\n${data.errors.join("\n")}` : ""),
        en + (data.skipped_records ? ` Kept ${data.skipped_records} imported records.` : "")
          + (data.errors.length ? `\nSome files could not be cleaned up:\n${data.errors.join("\n")}` : ""));
      node.setDirtyCanvas(true, true);
    });
  });
  select.addEventListener("change", () => {
    previewSelect.value = select.value;
    showSelection();
  });
  previewSelect.addEventListener("change", () => {
    select.value = previewSelect.value;
    showSelection();
  });
  function localize() {
    select.ariaLabel = t("选择已生成片段", "Select a generated segment");
    previewSelect.ariaLabel = t("选择预览片段", "Select a segment to preview");
    refresh.textContent = t("刷新已保存片段", "Refresh saved segments");
    restore.textContent = t("恢复参数重抽", "Restore settings for retry");
    exit.textContent = t("退出重抽", "Exit retry");
    cleanup.textContent = t("一键清理", "Clean up");
    cleanup.title = t("仅将列表中的重抽记录及辅助数据移到系统回收站，保留原片段和重抽片段视频及保存的 LATENT。刷新后列表包含 output 中所有批次。导入记录保留，运行或排队时不可清理。", "Move only retry records and supporting data in this list to the system Recycle Bin. Keep original and retry segment videos and saved LATENTs. After a refresh the list covers every batch in output. Imported records are kept. Unavailable while tasks are running or queued.");
    help.textContent = t("默认使用随机种子重抽，每次运行重新随机。选择“原种子”使用片段记录的实际种子；选择“自定义种子”可填写完整整数。重抽提示词留空沿用原文。模型、CLIP、VAE 和参考素材沿用当前连线；仅重抽选中一段。",
      "Retry uses a new random seed on every run by default. Original Seed uses the actual seed from the segment record; Custom Seed lets you enter an exact integer. A blank retry prompt uses the original text. Model, CLIP, VAE and references use the current connections. Only the selected segment is generated again.");
    if (!hasInfo) info.textContent = t("生成后显示实际种子、尺寸、时长与缓存路径。", "Actual seeds, dimensions, duration and cache paths appear after generation.");
    status.textContent = t(...statusMessage);
    for (const list of [select, previewSelect]) {
      for (const [index, option] of [...list.children].entries()) option.textContent = entryLabel(entries[index]);
    }
    showSelection(false);
  }
  const removeLocaleListener = onLocaleChange(localize);
  localize();
  showSelection();
  updateButtons();
  director.resize();
  return {
    addEntries(next) {
      const combined = new Map([...entries, ...next].map((entry) => [entry.recipe_path, entry]));
      setEntries([...combined.values()], false, next.at(-1)?.recipe_path ?? select.value);
    },
    resize() {
      director.resize();
    },
    configure() {
      director.syncWidgets();
      director.resize();
      updateButtons();
    },
    setInfo(value) {
      hasInfo = true;
      info.textContent = Array.isArray(value) ? value.join("\n") : String(value);
    },
    remove() {
      removed = true;
      removeLocaleListener();
      api.removeEventListener("status", onStatus);
      video.pause();
      video.removeAttribute("src");
      video.load();
      entries = [];
      beforeRetry = null;
      director.remove();
    },
  };
}

app.registerExtension({
  name: "H3LatentGuideSGUN.SegmentReview",
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "H3LongVideo") return;
    const onCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function (...args) {
      const result = onCreated?.apply(this, args);
      this.onDrawBackground = suppressNativePreview;
      suppressNativePreview.call(this);
      this._h3SgunSegmentReview = createPanel(this, nodeData);
      return result;
    };
    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (...args) {
      const result = onConfigure?.apply(this, args);
      this.onDrawBackground = suppressNativePreview;
      suppressNativePreview.call(this);
      this._h3SgunSegmentReview?.configure();
      return result;
    };
    nodeType.prototype.onDrawBackground = suppressNativePreview;
    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message, ...args) {
      const result = onExecuted?.call(this, message, ...args);
      if (Array.isArray(message?.h3_sgun_segments)) this._h3SgunSegmentReview?.addEntries(message.h3_sgun_segments);
      if (message?.h3_sgun_info) this._h3SgunSegmentReview?.setInfo(message.h3_sgun_info);
      return result;
    };
    const onRemoved = nodeType.prototype.onRemoved;
    nodeType.prototype.onRemoved = function (...args) {
      this._h3SgunSegmentReview?.remove();
      return onRemoved?.apply(this, args);
    };
  },
});
