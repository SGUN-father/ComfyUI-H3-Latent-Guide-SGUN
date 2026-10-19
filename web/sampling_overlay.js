import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { t, onLocaleChange } from "./i18n.js";

const promptOwners = new Map();
const startedPrompts = new Map();
const finishedPrompts = new Map();
let startSequence = 0;

function remember(map, id, value) {
  map.set(id, value);
  if (map.size > 64) map.delete(map.keys().next().value);
}

// The native promptQueued event has no prompt ID. Capture ownership at the
// queue call, before an await can switch the active workflow tab.
const queuePrompt = api.queuePrompt;
api.queuePrompt = async function (number, data, ...options) {
  const workflow = app.extensionManager?.workflow.activeWorkflow;
  const workflowId = data?.workflow?.id;
  const root = app.rootGraph ?? app.graph?.rootGraph ?? app.graph;
  const owner = workflow?.instanceId && workflowId && root?.id === workflowId
    ? { workflowId, instanceId: workflow.instanceId } : null;
  const response = await queuePrompt.call(this, number, data, ...options);
  if (owner && response?.prompt_id && !finishedPrompts.has(response.prompt_id)) {
    remember(promptOwners, response.prompt_id, owner);
  }
  return response;
};

function currentNode(node) {
  const root = app.rootGraph ?? app.graph?.rootGraph ?? app.graph;
  const graph = node.graph;
  return graph && (graph.rootGraph ?? graph) === root && graph.getNodeById(node.id) === node;
}

function matchesNode(node, id) {
  if (id == null) return false;
  const locator = node.graph?.isRootGraph === false ? `${node.graph.id}:${node.id}` : String(node.id);
  if (String(id) === locator) return true;
  const parts = String(id).split(":");
  let graph = app.rootGraph ?? app.graph?.rootGraph ?? app.graph;
  for (const [index, part] of parts.entries()) {
    const found = graph?.getNodeById?.(/^\d+$/.test(part) ? Number(part) : part);
    if (index === parts.length - 1) return found === node;
    graph = found?.subgraph;
    if (!graph) return false;
  }
  return false;
}

export function createSamplingOverlay(node, panel) {
  const element = document.createElement("div");
  element.className = "h3-sampling-overlay";
  element.hidden = true;
  const image = document.createElement("img");
  const toolbar = document.createElement("div");
  toolbar.className = "h3-sampling-toolbar";
  const status = document.createElement("span");
  status.setAttribute("role", "status");
  const close = document.createElement("button");
  close.type = "button";
  toolbar.append(status, close);
  element.append(image, toolbar);
  panel.append(element);

  let frames = [];
  let timer = null;
  let playbackStart = null;
  let playbackSegment = null;
  let playbackFps = null;
  let dismissed = false;
  let removed = false;
  let activePromptId = null;
  let activeSequence = 0;
  const retiredPrompts = new Map();
  let statusMessage = ["", ""];
  function localize() {
    image.alt = t("当前采样预览", "Current sampling preview");
    close.textContent = t("返回参数", "Back to settings");
    status.textContent = t(...statusMessage);
  }
  const removeLocaleListener = onLocaleChange(localize);
  localize();
  function stop() {
    if (timer !== null) clearInterval(timer);
    timer = null;
  }
  function hide() {
    stop();
    frames = [];
    playbackStart = null;
    playbackSegment = null;
    playbackFps = null;
    statusMessage = ["", ""];
    image.removeAttribute("src");
    element.hidden = true;
  }
  close.addEventListener("click", () => {
    dismissed = true;
    hide();
  });
  function ownsPrompt(data) {
    if (!data?.prompt_id || !currentNode(node)) return false;
    const owner = promptOwners.get(data.prompt_id);
    const root = app.rootGraph ?? app.graph?.rootGraph ?? app.graph;
    const workflow = app.extensionManager?.workflow.activeWorkflow;
    return owner && owner.workflowId === root.id && owner.instanceId === workflow?.instanceId
      && (data.workflow_id == null || data.workflow_id === owner.workflowId);
  }
  function beginPrompt(id) {
    if (activePromptId === id) return;
    if (activePromptId) remember(retiredPrompts, activePromptId, true);
    activePromptId = id;
    activeSequence = startedPrompts.get(id) ?? 0;
    dismissed = false;
    hide();
  }
  function preview(event) {
    const data = event.detail;
    if (removed || !ownsPrompt(data) || retiredPrompts.has(data.prompt_id)
      || ![data.node_id, data.display_node_id, data.display_node].some((id) => matchesNode(node, id))) return;
    if (activePromptId !== data.prompt_id) {
      // A start message can arrive before the queue HTTP response. Adopt it
      // on the first confirmed frame, while rejecting late frames from old runs.
      if (activePromptId && (startedPrompts.get(data.prompt_id) ?? 0) <= activeSequence) return;
      beginPrompt(data.prompt_id);
    }
    if (data.status === "complete") {
      remember(retiredPrompts, data.prompt_id, true);
      hide();
      return;
    }
    if (dismissed) return;
    if (data.status === "error") {
      hide();
      const message = data.message || "采样预览失败，生成继续运行。";
      let english = message;
      if (message === "采样预览失败，生成继续运行。") english = "Sampling preview failed. Generation continues.";
      else if (message.startsWith("采样预览已停用：")) {
        const reason = message.slice("采样预览已停用：".length)
          .replace("未找到 models/vae_approx/taeh3_temporal.safetensors", "models/vae_approx/taeh3_temporal.safetensors was not found")
          .replace("采样预览需要 H3 时序 TAEHV 权重，不能使用逐帧二维 TAE", "Sampling preview requires H3 temporal TAEHV weights; frame-by-frame 2D TAE weights are unsupported")
          .replace("采样预览需要 H3 视频 latent [B, 24, T, H, W]", "Sampling preview requires H3 video latents [B, 24, T, H, W]")
          .replace("采样预览帧数与 H3 视频 latent 的时长不匹配", "The preview frame count does not match the H3 video latent duration");
        english = `Sampling preview disabled: ${reason}`;
      }
      statusMessage = [message, english];
      localize();
      element.hidden = false;
      return;
    }
    if (data.status !== "sampling") return;
    stop();
    const batch = Array.isArray(data.frames) ? data.frames : [];
    const timeline = Array.isArray(data.frame_indices)
      ? data.frame_indices.filter(Number.isInteger).map((index) => batch[index]) : batch;
    frames = timeline.filter((frame) => typeof frame === "string" && frame.startsWith("data:image/jpeg;base64,"));
    if (!frames.length) {
      hide();
      return;
    }
    const fps = Number.isFinite(data.fps) ? Math.max(1, Math.min(60, data.fps)) : 10;
    if (playbackStart === null || playbackSegment !== data.segment || playbackFps !== fps) {
      playbackStart = performance.now();
      playbackSegment = data.segment;
      playbackFps = fps;
    }
    for (const video of panel.querySelectorAll("video")) video.pause();
    statusMessage = [`第 ${data.segment} 段 · 采样 ${data.step}/${data.total_steps} · ${frames.length} 帧 · ${fps} fps`,
      `Segment ${data.segment} · Sampling ${data.step}/${data.total_steps} · ${frames.length} frames · ${fps} fps`];
    localize();
    element.hidden = false;
    let displayedFrame = null;
    function render() {
      const index = Math.floor((performance.now() - playbackStart) * fps / 1000) % frames.length;
      const frame = frames[index];
      if (frame !== displayedFrame) image.src = frame;
      displayedFrame = frame;
    }
    render();
    if (frames.length > 1) timer = setInterval(render, 1000 / fps);
  }
  function start(event) {
    const data = event.detail;
    if (!data?.prompt_id || finishedPrompts.has(data.prompt_id)) return;
    if (!startedPrompts.has(data.prompt_id)) remember(startedPrompts, data.prompt_id, ++startSequence);
    if (ownsPrompt(data) && !retiredPrompts.has(data.prompt_id)) beginPrompt(data.prompt_id);
  }
  function finish(event) {
    const id = event.detail?.prompt_id;
    if (!id) return;
    promptOwners.delete(id);
    startedPrompts.delete(id);
    remember(finishedPrompts, id, true);
    if (id !== activePromptId) return;
    remember(retiredPrompts, id, true);
    hide();
  }
  const listeners = [
    ["h3-sgun-preview", preview], ["execution_start", start],
    ["execution_success", finish], ["execution_error", finish], ["execution_interrupted", finish],
  ];
  for (const [name, callback] of listeners) api.addEventListener(name, callback);
  return {
    hide,
    remove() {
      if (removed) return;
      removed = true;
      removeLocaleListener();
      hide();
      for (const [name, callback] of listeners) api.removeEventListener(name, callback);
      element.remove();
    },
  };
}
