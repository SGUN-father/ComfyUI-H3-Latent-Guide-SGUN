import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { test } from "node:test";
import vm from "node:vm";

class Element {
  constructor(tag) { this.tagName = tag; this.children = []; this.listeners = {}; }
  append(...children) { this.children.push(...children); }
  setAttribute(name, value) { this[name] = value; }
  removeAttribute(name) { delete this[name]; }
  addEventListener(name, callback) { this.listeners[name] = callback; }
  querySelectorAll(tag) { return this.children.flatMap((item) => [item, ...item.querySelectorAll(tag)]).filter((item) => item.tagName === tag); }
  pause() { this.paused = true; }
  remove() { this.removed = true; }
  click() { this.listeners.click(); }
}
const source = await readFile(new URL("../web/sampling_overlay.js", import.meta.url), "utf8");
async function createView(node = { id: 42, graph: { isRootGraph: true } }, app = {}) {
  app.graph ??= node.graph;
  app.rootGraph ??= app.graph.rootGraph ?? app.graph;
  app.rootGraph.id ??= "workflow-1";
  node.graph.rootGraph ??= app.rootGraph;
  node.graph.getNodeById ??= (id) => String(id) === String(node.id) ? node : null;
  app.extensionManager ??= { workflow: { activeWorkflow: { instanceId: "instance-1", activeState: { id: app.rootGraph.id } } } };
  const listeners = new Map();
  const localeListeners = new Set();
  let locale = "zh";
  const timers = new Map();
  let nextTimer = 1;
  let now = 0;
  let queueResponse = { prompt_id: "prompt-1" };
  let queueError = null;
  const queueCalls = [];
  const api = {
    addEventListener(name, callback) { if (!listeners.has(name)) listeners.set(name, new Set()); listeners.get(name).add(callback); },
    removeEventListener(name, callback) { listeners.get(name)?.delete(callback); },
    async queuePrompt(...args) {
      queueCalls.push({ receiver: this, args });
      if (queueError) throw queueError;
      return await queueResponse;
    },
  };
  const context = vm.createContext({
    document: { createElement: (name) => new Element(name) },
    performance: { now: () => now },
    setInterval(callback, delay) { const id = nextTimer++; timers.set(id, { callback, delay }); return id; },
    clearInterval(id) { timers.delete(id); },
  });
  const modules = {
    "../../scripts/app.js": new vm.SyntheticModule(["app"], function () { this.setExport("app", app); }, { context }),
    "../../scripts/api.js": new vm.SyntheticModule(["api"], function () { this.setExport("api", api); }, { context }),
    "./i18n.js": new vm.SyntheticModule(["t", "onLocaleChange"], function () {
      this.setExport("t", (zh, en) => locale.startsWith("zh") ? zh : en);
      this.setExport("onLocaleChange", (callback) => {
        localeListeners.add(callback);
        return () => localeListeners.delete(callback);
      });
    }, { context }),
  };
  const module = new vm.SourceTextModule(source, { context });
  await module.link((specifier) => modules[specifier]);
  await module.evaluate();
  await api.queuePrompt(0, { workflow: { id: app.rootGraph.id } });
  const panel = new Element("div");
  const video = new Element("video");
  panel.append(video);
  const overlay = module.namespace.createSamplingOverlay(node, panel);
  const element = panel.children.at(-1);
  const [image, toolbar] = element.children;
  const [status, close] = toolbar.children;
  return {
    overlay, element, image, status, close, panel, video, timers, listeners, localeListeners, app, node, api, queueCalls,
    queue(id, workflowId = app.rootGraph.id, ...options) {
      queueResponse = typeof id === "string" ? { prompt_id: id } : id;
      return api.queuePrompt(0, { workflow: { id: workflowId } }, ...options);
    },
    setQueueError(error) { queueError = error; },
    setLocale(value) { locale = value; for (const callback of localeListeners) callback(); },
    emit(name, detail) { for (const callback of listeners.get(name) ?? []) callback({ detail }); },
    preview(data = {}) { this.emit("h3-sgun-preview", { node_id: "42", prompt_id: "prompt-1", workflow_id: "workflow-1", segment: 1, step: 2, total_steps: 8, frames: ["data:image/jpeg;base64,A", "data:image/jpeg;base64,B"], fps: 8, status: "sampling", ...data }); },
    tick(elapsed = [...timers.values()][0]?.delay ?? 0) { now += elapsed; for (const { callback } of timers.values()) callback(); },
  };
}

test("sampling overlays the existing fixed panel and animates only the latest batch", async () => {
  const view = await createView();
  assert.equal(view.element.hidden, true);
  view.preview();
  assert.equal(view.element.hidden, false);
  assert.equal(view.image.src, "data:image/jpeg;base64,A");
  assert.match(view.status.textContent, /第 1 段.*2\/8/);
  assert.equal(view.video.paused, true);
  assert.equal(view.panel.children.length, 2);
  assert.equal([...view.timers.values()][0].delay, 125);
  view.tick();
  assert.equal(view.image.src, "data:image/jpeg;base64,B");
  view.preview({ frames: ["data:image/jpeg;base64,C"] });
  assert.equal(view.image.src, "data:image/jpeg;base64,C");
  assert.equal(view.timers.size, 0);
});

test("returning to parameters releases frames and prevents auto reopen for the same run", async () => {
  const view = await createView();
  view.preview();
  view.close.click();
  assert.equal(view.timers.size, 0);
  assert.equal(view.element.hidden, true);
  assert.equal(view.image.src, undefined);
  view.preview({ step: 4 });
  view.preview({ status: "complete" });
  view.preview({ segment: 2 });
  assert.equal(view.element.hidden, true);
  await view.queue("prompt-2");
  view.emit("execution_start", { prompt_id: "prompt-2", workflow_id: "workflow-1" });
  view.preview({ prompt_id: "prompt-2" });
  assert.equal(view.element.hidden, false);
});

test("completion and execution termination stop timers and return the existing tabs", async () => {
  for (const event of ["execution_success", "execution_error", "execution_interrupted"]) {
    const view = await createView();
    view.preview();
    view.emit(event, { prompt_id: "prompt-1" });
    assert.equal(view.timers.size, 0);
    assert.equal(view.image.src, undefined);
    assert.equal(view.element.hidden, true);
    await view.queue("prompt-2");
    view.emit("execution_start", { prompt_id: "prompt-2" });
    view.preview({ prompt_id: "prompt-2" });
    view.preview({ prompt_id: "prompt-2", status: "complete" });
    assert.equal(view.timers.size, 0);
    assert.equal(view.element.hidden, true);
  }
});

test("preview errors show plain closable status without keeping stale animation", async () => {
  const view = await createView();
  view.preview();
  view.preview({ status: "error", message: "<script>decode error</script>" });
  assert.equal(view.element.hidden, false);
  assert.equal(view.status.textContent, "<script>decode error</script>");
  assert.equal(view.status.children.length, 0);
  assert.equal(view.timers.size, 0);
  assert.equal(view.image.src, undefined);
  view.close.click();
  assert.equal(view.element.hidden, true);
});

test("node identity follows qualified subgraph paths without confusing equal leaf ids", async () => {
  const nested = { id: 42, graph: { isRootGraph: false, id: "graph-uuid" } };
  const rootNode = { id: 42, graph: { isRootGraph: true } };
  const subgraph = { getNodeById(id) { return id === 42 ? nested : null; } };
  const app = { graph: { getNodeById(id) { return id === 7 ? { subgraph } : id === 42 ? rootNode : null; } } };
  const view = await createView(nested, app);
  view.preview();
  assert.equal(view.element.hidden, true);
  view.preview({ node_id: "7:42" });
  assert.equal(view.element.hidden, false);
  view.preview({ node_id: "graph-uuid:42", status: "complete" });
  assert.equal(view.element.hidden, true);
  await view.queue("prompt-2");
  view.emit("execution_start", { prompt_id: "prompt-2" });
  view.preview({ prompt_id: "prompt-2", node_id: "expanded-id", display_node_id: "7:42" });
  assert.equal(view.element.hidden, false);
});

test("removal unregisters every API listener and discards the active frame batch", async () => {
  const view = await createView();
  view.preview();
  view.overlay.remove();
  assert.equal(view.timers.size, 0);
  assert.equal(view.element.removed, true);
  assert.equal(view.image.src, undefined);
  assert.ok([...view.listeners.values()].every((callbacks) => callbacks.size === 0));
  view.preview();
  view.overlay.remove();
  assert.equal(view.element.hidden, true);
});

test("unrelated nodes and invalid sources never open the overlay", async () => {
  const view = await createView();
  view.preview({ node_id: "43" });
  view.preview({ frames: ["https://example.com/image.jpg", "data:text/html;base64,ignored"] });
  assert.equal(view.element.hidden, true);
  assert.equal(view.timers.size, 0);
});

test("language changes update active sampling text without interrupting the frame batch", async () => {
  const view = await createView();
  view.preview();
  const timer = [...view.timers.keys()][0];
  view.tick();
  view.setLocale("en");
  assert.equal(view.image.alt, "Current sampling preview");
  assert.equal(view.close.textContent, "Back to settings");
  assert.equal(view.status.textContent, "Segment 1 · Sampling 2/8 · 2 frames · 8 fps");
  assert.equal(view.element.hidden, false);
  assert.equal(view.image.src, "data:image/jpeg;base64,B");
  assert.equal([...view.timers.keys()][0], timer);
  view.tick();
  assert.equal(view.image.src, "data:image/jpeg;base64,A");
  view.setLocale("zh");
  assert.equal(view.status.textContent, "第 1 段 · 采样 2/8 · 2 帧 · 8 fps");
  view.overlay.remove();
  assert.equal(view.localeListeners.size, 0);
});

test("known preview errors and fallback messages are translated while preserving technical details", async () => {
  const view = await createView();
  view.preview({ status: "error", message: "采样预览已停用：未找到 models/vae_approx/taeh3_temporal.safetensors" });
  view.setLocale("en");
  assert.equal(view.status.textContent, "Sampling preview disabled: models/vae_approx/taeh3_temporal.safetensors was not found");
  view.preview({ status: "error", message: "" });
  assert.equal(view.status.textContent, "Sampling preview failed. Generation continues.");
  view.setLocale("zh");
  assert.equal(view.status.textContent, "采样预览失败，生成继续运行。");
});

test("complete segment timelines keep frames beyond the old 48-frame limit", async () => {
  const view = await createView();
  const frames = Array.from({ length: 243 }, (_, index) => `data:image/jpeg;base64,${index}`);
  view.preview({ frames, fps: 24 });
  assert.match(view.status.textContent, /243 帧 · 24 fps/);
  assert.equal([...view.timers.values()][0].delay, 1000 / 24);
  view.tick(5000);
  assert.equal(view.image.src, frames[120]);
  view.tick(5000);
  assert.equal(view.image.src, frames[240]);
  view.tick(125);
  assert.equal(view.image.src, frames[0]);
});

test("native temporal spans animate the actual frame timeline without duplicating JPEG payloads", async () => {
  const view = await createView();
  const frames = ["data:image/jpeg;base64,A", "data:image/jpeg;base64,B", "data:image/jpeg;base64,C"];
  view.preview({ frames, frame_indices: [0, 1, 1, 1, 1, 2, 2, 2, 2], fps: 24 });
  assert.match(view.status.textContent, /9 帧 · 24 fps/);
  assert.equal(view.image.src, frames[0]);
  view.tick(50);
  assert.equal(view.image.src, frames[1]);
  view.tick(150);
  assert.equal(view.image.src, frames[1]);
  view.tick(50);
  assert.equal(view.image.src, frames[2]);
  view.tick(125);
  assert.equal(view.image.src, frames[0]);
});

test("malformed frame indexes do not display invalid sources or shift valid JPEG indexes", async () => {
  const view = await createView();
  view.preview({ frames: ["https://example.com/ignored.jpg", "data:image/jpeg;base64,B"], frame_indices: [1, 1, 0, -1, 2, 0.5], fps: 24 });
  assert.equal(view.image.src, "data:image/jpeg;base64,B");
  assert.match(view.status.textContent, /2 帧/);
  view.preview({ frame_indices: [9] });
  assert.equal(view.element.hidden, true);
  assert.equal(view.image.src, undefined);
  assert.equal(view.timers.size, 0);
});

test("preview playback accepts 60 fps and catches up after delayed browser ticks", async () => {
  const view = await createView();
  const frames = Array.from({ length: 120 }, (_, index) => `data:image/jpeg;base64,${index}`);
  view.preview({ frames, fps: 60 });
  assert.equal([...view.timers.values()][0].delay, 1000 / 60);
  view.tick(100);
  assert.equal(view.image.src, frames[6]);
  view.tick(900);
  assert.equal(view.image.src, frames[60]);
  view.tick(1000);
  assert.equal(view.image.src, frames[0]);
});

test("new sampling steps replace images without restarting the same segment timeline", async () => {
  const view = await createView();
  const frames = Array.from({ length: 240 }, (_, index) => `data:image/jpeg;base64,A${index}`);
  view.preview({ frames, fps: 24 });
  view.tick(2500);
  assert.equal(view.image.src, frames[60]);
  const latest = frames.map((frame) => frame.replace(",A", ",B"));
  view.preview({ frames: latest, step: 4, fps: 24 });
  assert.equal(view.image.src, latest[60]);
  assert.equal(view.timers.size, 1);
  view.tick(2500);
  assert.equal(view.image.src, latest[120]);
  view.preview({ frames: latest, segment: 2, fps: 24 });
  assert.equal(view.image.src, latest[0]);
  view.tick(1000);
  view.preview({ frames: latest, segment: 2, fps: 12 });
  assert.equal(view.image.src, latest[0]);
});

test("missing preview fps uses 10 while requested limits remain usable", async () => {
  const view = await createView();
  view.preview({ fps: undefined });
  assert.equal([...view.timers.values()][0].delay, 1000 / 10);
  view.preview({ fps: 1 });
  assert.equal([...view.timers.values()][0].delay, 1000);
  view.preview({ fps: 90 });
  assert.equal([...view.timers.values()][0].delay, 1000 / 60);
});

test("unknown prompts are rejected even when workflow and node IDs match", async () => {
  const view = await createView();
  view.preview({ prompt_id: undefined });
  view.preview({ prompt_id: "unknown-prompt" });
  assert.equal(view.element.hidden, true);
  view.emit("execution_start", { prompt_id: "unknown-prompt", workflow_id: "workflow-1" });
  view.preview({ prompt_id: "unknown-prompt" });
  assert.equal(view.element.hidden, true);
  view.preview({ workflow_id: "other-workflow" });
  assert.equal(view.element.hidden, true);
  // Local queue ownership is sufficient when older servers omit workflow metadata.
  view.preview({ workflow_id: undefined });
  assert.equal(view.element.hidden, false);
});

test("workflow tab instances remain isolated when serialized workflow IDs are equal", async () => {
  const view = await createView();
  const original = view.app.extensionManager.workflow.activeWorkflow;
  view.app.extensionManager.workflow.activeWorkflow = { instanceId: "copied-tab", activeState: { id: "workflow-1" } };
  view.emit("execution_start", { prompt_id: "prompt-1", workflow_id: "workflow-1" });
  view.preview();
  assert.equal(view.element.hidden, true);
  await view.queue("copied-prompt");
  view.preview({ prompt_id: "copied-prompt" });
  assert.equal(view.element.hidden, false);
  view.close.click();
  view.app.extensionManager.workflow.activeWorkflow = original;
  view.preview({ prompt_id: "copied-prompt" });
  assert.equal(view.element.hidden, true);
  view.preview();
  assert.equal(view.element.hidden, false);
});

test("a detached node cannot receive another workflow's equal node ID", async () => {
  const view = await createView();
  const oldNode = view.node;
  const replacement = { id: 42, graph: view.app.rootGraph };
  view.app.rootGraph.getNodeById = () => replacement;
  view.preview();
  assert.equal(view.element.hidden, true);
  oldNode.graph = null;
  view.preview();
  assert.equal(view.element.hidden, true);
});

test("queue ownership is captured before a tab switch and preserves native responses and arguments", async () => {
  const view = await createView();
  let resolve;
  const pending = new Promise((done) => { resolve = done; });
  const options = { partialExecutionTargets: [42] };
  const result = view.queue(pending, "workflow-1", options);
  const workflow = view.app.extensionManager.workflow.activeWorkflow;
  view.app.extensionManager.workflow.activeWorkflow = { instanceId: "other-tab", activeState: { id: "workflow-2" } };
  view.app.rootGraph.id = "workflow-2";
  const response = { prompt_id: "pending-prompt", number: 17, node_errors: {} };
  resolve(response);
  assert.equal(await result, response);
  const call = view.queueCalls.at(-1);
  assert.equal(call.receiver, view.api);
  assert.equal(call.args[0], 0);
  assert.equal(call.args[1].workflow.id, "workflow-1");
  assert.equal(call.args[2], options);
  view.preview({ prompt_id: "pending-prompt", workflow_id: "workflow-1" });
  assert.equal(view.element.hidden, true);
  view.app.extensionManager.workflow.activeWorkflow = workflow;
  view.app.rootGraph.id = "workflow-1";
  view.preview({ prompt_id: "pending-prompt" });
  assert.equal(view.element.hidden, false);
  const error = new Error("native queue validation error");
  view.setQueueError(error);
  await assert.rejects(view.queue("failed-prompt"), (caught) => caught === error);
});

test("foreign starts and terminal events leave the active preview and dismissal untouched", async () => {
  const view = await createView();
  view.emit("execution_start", { prompt_id: "prompt-1", workflow_id: "workflow-1" });
  view.preview();
  const image = view.image.src;
  for (const event of ["execution_start", "execution_success", "execution_error", "execution_interrupted"]) {
    view.emit(event, { prompt_id: "foreign-prompt", workflow_id: "foreign-workflow" });
    view.emit(event, {});
    assert.equal(view.element.hidden, false);
    assert.equal(view.image.src, image);
    assert.equal(view.timers.size, 1);
  }
  view.close.click();
  view.emit("execution_start", { prompt_id: "unmapped-prompt", workflow_id: "workflow-1" });
  view.preview({ step: 3 });
  assert.equal(view.element.hidden, true);
});

test("old task frames and termination cannot replace a newer task's preview", async () => {
  const view = await createView();
  view.emit("execution_start", { prompt_id: "prompt-1" });
  view.preview();
  await view.queue("prompt-2");
  view.emit("execution_start", { prompt_id: "prompt-2" });
  view.preview({ prompt_id: "prompt-2", frames: ["data:image/jpeg;base64,NEW"] });
  view.preview({ frames: ["data:image/jpeg;base64,OLD"] });
  view.emit("execution_start", { prompt_id: "prompt-1" });
  view.emit("execution_error", { prompt_id: "prompt-1" });
  assert.equal(view.image.src, "data:image/jpeg;base64,NEW");
  assert.equal(view.element.hidden, false);
  view.emit("execution_success", { prompt_id: "prompt-2" });
  view.preview({ prompt_id: "prompt-2" });
  assert.equal(view.element.hidden, true);
  assert.equal(view.image.src, undefined);
});

test("a start arriving before the HTTP response takes ownership lazily without reviving older tasks", async () => {
  const view = await createView();
  view.emit("execution_start", { prompt_id: "prompt-1" });
  view.preview();
  let resolve;
  const result = view.queue(new Promise((done) => { resolve = done; }));
  view.emit("execution_start", { prompt_id: "prompt-2", workflow_id: "workflow-1" });
  view.preview({ prompt_id: "prompt-2", frames: ["data:image/jpeg;base64,EARLY"] });
  assert.equal(view.image.src, "data:image/jpeg;base64,A");
  resolve({ prompt_id: "prompt-2" });
  await result;
  view.preview({ prompt_id: "prompt-2", frames: ["data:image/jpeg;base64,NEW"] });
  assert.equal(view.image.src, "data:image/jpeg;base64,NEW");
  view.preview();
  assert.equal(view.image.src, "data:image/jpeg;base64,NEW");
});

test("a task completed before its HTTP response never registers a stale preview", async () => {
  const view = await createView();
  let resolve;
  const result = view.queue(new Promise((done) => { resolve = done; }));
  view.emit("execution_start", { prompt_id: "brief-prompt", workflow_id: "workflow-1" });
  view.emit("execution_success", { prompt_id: "brief-prompt", workflow_id: "workflow-1" });
  resolve({ prompt_id: "brief-prompt" });
  await result;
  view.preview({ prompt_id: "brief-prompt" });
  assert.equal(view.element.hidden, true);
});

test("native temporal weight format failures explain the required weights in English", async () => {
  const view = await createView();
  view.setLocale("en");
  view.preview({ status: "error", message: "采样预览已停用：采样预览需要 H3 时序 TAEHV 权重，不能使用逐帧二维 TAE" });
  assert.equal(view.status.textContent, "Sampling preview disabled: Sampling preview requires H3 temporal TAEHV weights; frame-by-frame 2D TAE weights are unsupported");
});
