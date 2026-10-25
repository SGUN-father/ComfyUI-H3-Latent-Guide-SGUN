// Run with: node --experimental-vm-modules tests/test_segment_review.mjs
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import vm from "node:vm";
import { test } from "node:test";

class Element {
  constructor(tagName) {
    this.tagName = tagName;
    this.style = {};
    this.children = [];
    this.listeners = {};
    this.attributes = {};
    this.value = "";
  }
  append(...children) { this.children.push(...children); }
  prepend(...children) { this.children.unshift(...children); }
  replaceChildren(...children) { this.children = children; }
  addEventListener(name, callback) { this.listeners[name] = callback; }
  setAttribute(name, value) { this.attributes[name] = value; }
  getAttribute(name) { return this.attributes[name] ?? this[name] ?? null; }
  removeAttribute(name) { delete this[name]; }
  pause() { this.paused = true; }
  load() { this.loads = (this.loads ?? 0) + 1; }
  fire(name) { return this.listeners[name]?.({}); }
}

let extension;
let response;
let locale = "zh";
const localeListeners = new Set();
function setLocale(value) {
  locale = value;
  for (const callback of localeListeners) callback();
}
const requests = [];
const apiListeners = new Map();
function queueStatus(remaining) {
  const detail = remaining == null ? null : { exec_info: { queue_remaining: remaining } };
  for (const callback of apiListeners.get("status") ?? []) callback({ detail });
}
const app = { registerExtension(value) { extension = value; } };
const api = {
  async fetchApi(path, options) {
    requests.push(path);
    const selected = response;
    selected.onFetch?.(path, options);
    return { ok: selected.ok ?? true, status: selected.status ?? 200, async json() { return selected.body; } };
  },
  addEventListener(name, callback) {
    if (!apiListeners.has(name)) apiListeners.set(name, new Set());
    apiListeners.get(name).add(callback);
  },
  removeEventListener(name, callback) { apiListeners.get(name)?.delete(callback); },
  apiURL(path) { return `/comfy${path}`; },
};
function createDirectorPanel(node, nodeData) {
  const sections = Object.fromEntries(["generate", "retry", "preview", "info"].map((name) => [name, new Element("section")]));
  const element = new Element("div");
  element.append(...Object.values(sections));
  node.addDOMWidget("h3_sgun_director", "div", element, { serialize: false });
  const director = {
    sections,
    seedControl: node.widgets.find((item) => item.name === "control_after_generate")
      ?? node.widgets.find((item) => item.name === nodeData.input?.required?.seed?.[1]?.control_after_generate),
    activeTab: "generate",
    setTab(value) { this.activeTab = value; },
    syncWidgets() { this.syncs = (this.syncs ?? 0) + 1; },
    setSummary(value) { this.summary = value; },
    resize() { this.resized = true; this.resizes = (this.resizes ?? 0) + 1; },
    remove() { this.removed = true; },
  };
  node.director = director;
  return director;
}
const context = vm.createContext({ document: { createElement: (name) => new Element(name) }, URLSearchParams });
const modules = Object.fromEntries(Object.entries({ "../../scripts/app.js": ["app", app], "../../scripts/api.js": ["api", api] })
  .map(([path, [name, value]]) => [path, new vm.SyntheticModule([name], function () { this.setExport(name, value); }, { context })]));
modules["./director_panel.js"] = new vm.SyntheticModule(["createDirectorPanel"], function () {
  this.setExport("createDirectorPanel", createDirectorPanel);
}, { context });
modules["./i18n.js"] = new vm.SyntheticModule(["t", "onLocaleChange"], function () {
  this.setExport("t", (zh, en) => locale.startsWith("zh") ? zh : en);
  this.setExport("onLocaleChange", (callback) => {
    localeListeners.add(callback);
    return () => localeListeners.delete(callback);
  });
}, { context });
const source = await readFile(new URL("../web/segment_review.js", import.meta.url), "utf8");
const module = new vm.SourceTextModule(source, { context });
await module.link((specifier) => {
  assert.ok(modules[specifier]);
  return modules[specifier];
});
await module.evaluate();
assert.equal(extension.name, "H3LatentGuideSGUN.SegmentReview");

const settings = {
  duration: 5, aspect_ratio: "16:9 (Widescreen)", megapixels: 0.4, seed: 101,
  steps: 8, sampler_name: "res_multistep", scheduler: "simple", context_frames: 22,
  denoise: 1, ref_image_size: "match", dynamic_mask: true, drift_correction: false,
};
function record(overrides = {}) {
  const data = {
    recipe_path: "video/H3/segment_2.json [output]", segment: 2, seed: 938, width: 832, height: 480, frames: 124,
    video: null, seam: { filename: "seam & 2.mp4", subfolder: "video/H3", type: "output" },
    diagnostics: { status: "warning", messages: ["亮度跳变"], join_frame: 12, frame_count: 48, fps: 24 },
    prompt: "A person walks forward.", settings: { ...settings }, sigmas_source: "internal", context_available: true,
    ...overrides,
  };
  data.seed_text ??= String(data.seed);
  return data;
}
function createNode(type = "H3LongVideo", initialWidgets = [], seedControlName = "control_after_generate") {
  class Node {
    constructor() {
      this.id = 42;
      this.widgets = Object.entries({ ...settings, segments: 4, seed_mode: "increment", seed: 333,
        [seedControlName]: "randomize", retry_segment: "", retry_prompt: "", retry_seed: "", filename_prefix: "original/path", retry_seed_mode: "randomize" })
        .map(([name, value]) => ({ name, value, ...(name === "retry_seed_mode"
          ? { type: "combo", options: { values: ["randomize", "original", "custom"] } } : {}) }));
      this.widgets.push(...initialWidgets);
      if (initialWidgets.length) this.imgs = [{ src: "blob:old-preview" }];
      this.inputs = [{ name: "model", link: 4 }, { name: "clip", link: 5 }, { name: "sigmas", link: 6 }];
      this.graph = { links: { 4: { origin_id: 1 }, 5: { origin_id: 2 }, 6: { origin_id: 3 } } };
      this.size = [300, 500];
    }
    onNodeCreated() { this.originalCreated = true; return "created"; }
    onExecuted(message) { this.originalMessage = message; return "executed"; }
    onConfigure() { this.originalConfigured = true; return "configured"; }
    onRemoved() { this.originalRemoved = true; return "removed"; }
    onDrawBackground() { this.nativePreviewDrawn = true; return "drawn"; }
    addDOMWidget(name, type, element, options) {
      this.panel = { name, type, element, options };
      return this.panel;
    }
    computeSize() { return [this.size[0], 860]; }
    setSize(value) { this.size = value; }
    setDirtyCanvas() { this.dirty = true; }
  }
  const originals = { onNodeCreated: Node.prototype.onNodeCreated, onExecuted: Node.prototype.onExecuted,
    onConfigure: Node.prototype.onConfigure, onRemoved: Node.prototype.onRemoved, onDrawBackground: Node.prototype.onDrawBackground };
  extension.beforeRegisterNodeDef(Node, { name: type,
    input: { required: { seed: ["INT", { control_after_generate: seedControlName }] } } });
  const node = new Node();
  node.onNodeCreated();
  if (type !== "H3LongVideo") return { node, originals, prototype: Node.prototype };
  const [select, buttons, status, help] = node.director.sections.retry.children;
  const [previewSelect, details, video] = node.director.sections.preview.children;
  const [info] = node.director.sections.info.children;
  const [refresh, restore, exit, cleanup] = buttons.children;
  return { node, select, previewSelect, refresh, restore, exit, cleanup, status, details, video, help, info };
}
function values(node) { return Object.fromEntries(node.widgets.map((item) => [item.name, item.value])); }
function widget(node, name) { return node.widgets.find((item) => item.name === name); }
function load(node, entries = [record()]) {
  const message = { h3_sgun_segments: entries };
  assert.equal(node.onExecuted(message), "executed");
  assert.equal(node.originalMessage, message);
}

test("restores every recipe setting and actual seed without queuing or changing links", async () => {
  const view = createNode();
  const { node, restore, status, exit } = view;
  const original = values(node);
  const links = structuredClone({ inputs: node.inputs, graph: node.graph });
  load(node);
  response = { body: record() };
  await restore.fire("click");
  const current = values(node);
  for (const name of Object.keys(settings).filter((name) => name !== "seed")) assert.equal(current[name], settings[name]);
  assert.equal(current.seed, 938);
  assert.equal(current.seed_mode, "fixed");
  assert.equal(current.control_after_generate, "fixed");
  assert.equal(current.segments, 1);
  assert.equal(current.retry_segment, record().recipe_path);
  assert.equal(current.retry_prompt, record().prompt);
  assert.equal(current.retry_seed, "938");
  assert.equal(current.retry_seed_mode, "randomize");
  assert.match(status.textContent, /按所选重抽种子方式运行/);
  assert.equal(current.filename_prefix, original.filename_prefix);
  assert.match(status.textContent, /步数、调度器与降噪重新计算/);
  assert.deepEqual({ inputs: node.inputs, graph: node.graph }, links);
  assert.equal(exit.disabled, false);
  widget(node, "seed").value = 772;
  widget(node, "retry_prompt").value = "A new action.";
  exit.fire("click");
  assert.deepEqual(values(node), original);
  assert.equal(exit.disabled, true);
});

test("external SIGMAS are explained and retry selection keeps the original exit snapshot", async () => {
  const { node, restore, exit, status } = createNode();
  const original = values(node);
  const other = createNode();
  const otherValues = values(other.node);
  load(node);
  response = { body: record({ sigmas_source: "external" }) };
  await restore.fire("click");
  assert.match(status.textContent, /原记录序列/);
  assert.match(status.textContent, /当前 SIGMAS 连线不参与/);
  response = { body: record({ seed: 1942 }) };
  await restore.fire("click");
  assert.equal(values(node).seed, 1942);
  exit.fire("click");
  assert.deepEqual(values(node), original);
  assert.deepEqual(values(other.node), otherValues);
});

test("restoring uses the configured native seed control name without renaming or losing its exit value", async () => {
  const { node, restore, exit, status } = createNode("H3LongVideo", [], "randomize");
  const original = values(node);
  load(node);
  response = { body: record({ sigmas_source: "external" }) };
  await restore.fire("click");
  assert.match(status.textContent, /已恢复第 2 段参数/);
  assert.equal(values(node).randomize, "fixed");
  assert.equal(values(node).retry_seed_mode, "randomize");
  assert.equal(values(node).retry_segment, record().recipe_path);
  assert.equal(widget(node, "control_after_generate"), undefined);
  exit.fire("click");
  assert.deepEqual(values(node), original);
  node.onRemoved();
});

test("missing widgets identify the control without asking to regenerate complete segment records", async () => {
  for (const name of ["duration", "control_after_generate", "retry_seed_mode"]) {
    const { node, restore, status } = createNode();
    node.widgets = node.widgets.filter((item) => item.name !== name);
    const original = values(node);
    load(node);
    response = { body: record() };
    await restore.fire("click");
    assert.match(status.textContent, /缺少控件/);
    assert.ok(status.textContent.includes(name));
    assert.doesNotMatch(status.textContent, /重新生成/);
    assert.deepEqual(values(node), original);
    node.onRemoved();
  }
});

test("restoring preserves the selected retry seed mode and exiting restores its entry snapshot", async () => {
  for (const mode of ["randomize", "original", "custom"]) {
    const { node, restore, exit, help, status } = createNode();
    const modeWidget = widget(node, "retry_seed_mode");
    assert.equal(modeWidget.type, "combo");
    assert.deepEqual(modeWidget.options.values, ["randomize", "original", "custom"]);
    modeWidget.value = mode;
    const original = values(node);
    load(node);
    response = { body: record({ seed: Number("18446744073709551615"), seed_text: "18446744073709551615" }) };
    await restore.fire("click");
    assert.equal(values(node).retry_seed_mode, mode);
    assert.equal(values(node).retry_seed, "18446744073709551615");
    assert.match(help.textContent, /默认使用随机种子重抽，每次运行重新随机/);
    assert.match(status.textContent, /按所选重抽种子方式运行/);
    modeWidget.value = mode === "custom" ? "randomize" : "custom";
    response = { body: record({ seed: 1942 }) };
    await restore.fire("click");
    assert.equal(values(node).retry_seed_mode, mode === "custom" ? "randomize" : "custom");
    exit.fire("click");
    assert.deepEqual(values(node), original);
    node.onRemoved();
  }
});

test("linked scalar inputs reject the whole restore before any value changes", async () => {
  for (const input of [{ name: "seed", link: 10 }, { name: "converted", widget: { name: "duration" }, link: 11 },
    { name: "retry_prompt", link: 12 }, { name: "segments", link: 13 }, { name: "retry_seed", link: 14 }]) {
    const { node, restore, status, exit } = createNode();
    node.inputs.push(input);
    const original = values(node);
    load(node);
    response = { body: record() };
    await restore.fire("click");
    assert.deepEqual(values(node), original);
    assert.match(status.textContent, /由连线控制/);
    assert.match(status.textContent, /转回节点控件或断开/);
    assert.equal(exit.disabled, true);
  }
});

test("missing context, incomplete records and server errors never partially overwrite widgets", async () => {
  for (const data of [
    { body: record({ context_available: false }) },
    { body: record({ settings: { duration: 8 } }) },
    { ok: false, status: 400, body: { error: "旧记录缺少必要参数，请重新生成" } },
  ]) {
    const { node, restore, status } = createNode();
    const original = values(node);
    load(node);
    response = data;
    await restore.fire("click");
    assert.deepEqual(values(node), original);
    assert.match(status.textContent, /上下文|不完整|旧记录/);
  }
});

test("record paths and video filenames are encoded, messages are rendered as text", async () => {
  const { node, restore, select, video, details } = createNode();
  const item = record({ recipe_path: "weird & path?#.json [output]",
    diagnostics: { ...record().diagnostics, messages: ["<img onerror='bad'>"] } });
  load(node, [item]);
  assert.equal(select.value, item.recipe_path);
  response = { body: item };
  await restore.fire("click");
  const url = new URL(requests.at(-1), "http://localhost");
  assert.equal(url.pathname, "/h3-sgun/segment");
  assert.deepEqual([...url.searchParams.keys()], ["path"]);
  assert.equal(url.searchParams.get("path"), item.recipe_path);
  const videoUrl = new URL(video.src, "http://localhost");
  assert.equal(videoUrl.searchParams.get("filename"), "seam & 2.mp4");
  assert.equal(video.controls, true);
  assert.equal(video.hidden, false);
  assert.match(details.textContent, /0.50 秒/);
  assert.match(details.textContent, /<img onerror='bad'>/);
  assert.equal(details.children.length, 0);
});

test("refresh loads history, shows first-segment explanation, and keeps preview state outside workflow", async () => {
  const { node, refresh, restore, details, video, select } = createNode();
  assert.equal(restore.disabled, true);
  assert.equal(node.panel.options.serialize, false);
  assert.equal(node.originalCreated, true);
  response = { body: { segments: [record({ segment: 1, seam: null, diagnostics: null })] } };
  await refresh.fire("click");
  assert.equal(requests.at(-1), "/h3-sgun/segments");
  assert.equal(restore.disabled, false);
  assert.match(details.textContent, /首段没有前段接缝/);
  assert.equal(video.hidden, true);
  load(node, [record()]);
  assert.equal(video.hidden, false);
  response = { body: { segments: [] } };
  await refresh.fire("click");
  assert.equal(select.children.length, 0);
  assert.equal(restore.disabled, true);
  assert.equal(video.hidden, true);
  assert.equal(video.src, undefined);
  assert.equal(node.onRemoved(), "removed");
  assert.equal(video.paused, true);
  assert.equal(node.originalRemoved, true);
});

test("an unavailable seam is shown as unchecked, including continuation from an external LATENT", () => {
  const { node, details, video } = createNode();
  load(node, [record({ segment: 1, seam: null,
    diagnostics: { status: "unavailable", messages: ["缺少前段视频"], join_frame: null, frame_count: 0, fps: 24 } })]);
  assert.match(details.textContent, /接缝检查：未检查/);
  assert.match(details.textContent, /缺少前段视频/);
  assert.doesNotMatch(details.textContent, /首段没有前段接缝/);
  assert.equal(video.hidden, true);
});

test("a loaded workflow can exit retry without claiming to restore an in-memory snapshot", () => {
  const { node, exit, status } = createNode();
  widget(node, "retry_segment").value = "loaded/recipe.json [output]";
  widget(node, "retry_prompt").value = "Loaded prompt";
  widget(node, "retry_seed").value = "18446744073709551615";
  assert.equal(node.onConfigure(), "configured");
  assert.equal(node.originalConfigured, true);
  assert.equal(exit.disabled, false);
  exit.fire("click");
  assert.equal(values(node).retry_segment, "");
  assert.equal(values(node).retry_prompt, "");
  assert.equal(values(node).retry_seed, "");
  assert.equal(values(node).seed, 333);
  assert.equal(status.textContent, "已退出重抽。");
});

test("history defaults to newest record, while execution defaults to last generated segment", async () => {
  const { node, refresh, select } = createNode();
  const first = record({ recipe_path: "first.json [output]", segment: 1 });
  const last = record({ recipe_path: "last.json [output]", segment: 2 });
  response = { body: { segments: [last, first] } };
  await refresh.fire("click");
  assert.equal(select.value, last.recipe_path);
  load(node, []);
  load(node, [first, last]);
  assert.equal(select.value, last.recipe_path);
});

test("a one-segment retry keeps the original segments and selects the new version without duplicates", async () => {
  const { node, select, previewSelect, restore, exit, refresh, details } = createNode();
  const originals = [1, 2, 3].map((segment) => record({ segment,
    recipe_path: `H3_00001/segment_000${segment}.h3segment.json [output]` }));
  load(node, originals);
  select.value = originals[1].recipe_path;
  select.fire("change");
  response = { body: originals[1] };
  await restore.fire("click");
  const retry = record({ segment: 2, seed: 1942,
    recipe_path: "H3_00002/segment_0001.h3segment.json [output]" });
  load(node, [retry]);
  assert.deepEqual(select.children.map((item) => item.value), [...originals, retry].map((item) => item.recipe_path));
  assert.deepEqual(previewSelect.children.map((item) => item.value), select.children.map((item) => item.value));
  assert.equal(select.value, retry.recipe_path);
  assert.equal(previewSelect.value, retry.recipe_path);
  assert.match(details.textContent, /种子 1942/);
  exit.fire("click");
  assert.equal(select.children.length, 4);
  load(node, [retry]);
  assert.equal(select.children.length, 4);
  const updated = { ...retry, seed: 2013, seed_text: "2013" };
  load(node, [updated]);
  assert.equal(select.children.length, 4);
  assert.match(details.textContent, /种子 2013/);
  response = { body: { segments: [updated, originals[2]] } };
  await refresh.fire("click");
  assert.deepEqual(select.children.map((item) => item.value), [updated.recipe_path, originals[2].recipe_path]);
  assert.equal(select.value, retry.recipe_path);
  node.onRemoved();
});

test("retry and preview selectors stay synchronized while reviewing different seams", () => {
  const { node, select, previewSelect, details, video } = createNode();
  const first = record({ recipe_path: "first.json [output]", segment: 1, seam: null, diagnostics: null });
  const last = record({ recipe_path: "last.json [output]", segment: 2 });
  load(node, [first, last]);
  assert.equal(previewSelect.value, last.recipe_path);
  previewSelect.value = first.recipe_path;
  previewSelect.fire("change");
  assert.equal(select.value, first.recipe_path);
  assert.match(details.textContent, /首段没有前段接缝/);
  assert.equal(video.hidden, true);
  select.value = last.recipe_path;
  select.fire("change");
  assert.equal(previewSelect.value, last.recipe_path);
  assert.equal(video.hidden, false);
  assert.match(node.director.summary, /第 2 段/);
});

test("generation information is updated in its own tab as plain text", () => {
  const { node, info, status } = createNode();
  const previousStatus = status.textContent;
  const content = ["actual seed: 18446744073709551615", "<script>untrusted</script>", "temp/segment_2.mp4"];
  node.onExecuted({ h3_sgun_info: content });
  assert.equal(info.textContent, content.join("\n"));
  assert.equal(info.children.length, 0);
  assert.equal(status.textContent, previousStatus);
  node.onExecuted({ h3_sgun_info: "a new run" });
  assert.equal(info.textContent, "a new run");
});

test("loaded workflows refresh director controls and resize without discarding retry parameters", () => {
  const { node, exit } = createNode();
  widget(node, "retry_segment").value = "saved/segment.h3segment.json [output]";
  widget(node, "retry_seed").value = "18446744073709551615";
  const expected = values(node);
  const previousSyncs = node.director.syncs ?? 0;
  node.onConfigure();
  assert.equal(node.director.resized, true);
  assert.ok(node.director.syncs > previousSyncs);
  assert.equal(exit.disabled, false);
  assert.deepEqual(values(node), expected);
  node.onRemoved();
  assert.equal(node.director.removed, true);
});

test("native low resolution sampling previews no longer appear in a separate section", () => {
  const view = createNode();
  assert.equal(view.node.director.sections.preview.children.length, 3);
  app.nodePreviewImages = { 42: ["blob:latest"] };
  view.node.onDrawBackground();
  assert.equal(view.node.nativePreviewDrawn, undefined);
  assert.equal(view.node.director.sections.preview.children.length, 3);
});

test("native preview cleanup removes only native widgets and resizes only when needed", () => {
  const { node } = createNode();
  const original = values(node);
  const removed = [];
  node.removeWidget = function (item) {
    removed.push(item.name);
    item.onRemove?.();
    this.widgets.splice(this.widgets.indexOf(item), 1);
  };
  const previewNames = ["$$canvas-image-preview", "$$comfy_animation_preview", "video-preview"];
  const cleanupCalls = [];
  node.widgets.push(node.panel, ...previewNames.map((name) => ({ name, onRemove() { cleanupCalls.push(name); } })));
  node.imgs = [{ src: "blob:old-preview" }];
  const nativePreviews = app.nodePreviewImages = { 42: ["blob:latest"], 99: ["blob:other"] };
  const previousResizes = node.director.resizes;
  node.onDrawBackground();
  assert.deepEqual(removed, previewNames);
  assert.deepEqual(cleanupCalls, previewNames);
  assert.ok(node.widgets.includes(node.panel));
  assert.equal(node.imgs, undefined);
  assert.equal(node.hideOutputImages, true);
  assert.equal(node.director.resizes, previousResizes + 1);
  assert.equal(app.nodePreviewImages, nativePreviews);
  assert.deepEqual(nativePreviews, { 42: ["blob:latest"], 99: ["blob:other"] });
  const { h3_sgun_director: panelValue, ...current } = values(node);
  assert.deepEqual(current, original);
  node.onDrawBackground();
  assert.equal(node.director.resizes, previousResizes + 1);
});

test("native preview cleanup runs on creation and configure with the older widget API", () => {
  let cleanupCalls = 0;
  const native = { name: "$$canvas-image-preview", onRemove() { cleanupCalls++; } };
  const { node } = createNode("H3LongVideo", [native]);
  assert.equal(cleanupCalls, 1);
  assert.equal(node.widgets.includes(native), false);
  assert.equal(node.imgs, undefined);
  const ownDraw = node.onDrawBackground;
  Object.getPrototypeOf(node).onDrawBackground = function () { this.nativePreviewDrawn = true; };
  node.onDrawBackground();
  assert.equal(node.onDrawBackground, ownDraw);
  assert.equal(node.nativePreviewDrawn, undefined);
  const original = values(node);
  node.widgets.push(native);
  node.imgs = [{ src: "blob:restored-preview" }];
  node.onConfigure();
  assert.equal(cleanupCalls, 2);
  assert.equal(node.widgets.includes(native), false);
  assert.equal(node.imgs, undefined);
  assert.deepEqual(values(node), original);
});

test("uint64 seeds retain their exact decimal text without overwriting the ordinary numeric seed", async () => {
  for (const seedText of ["9007199254740993", "18446744073709551615"]) {
    const { node, restore, select, details, exit, help, status } = createNode();
    const original = values(node);
    load(node);
    response = { body: record() };
    await restore.fire("click");
    assert.equal(values(node).seed, 938);
    const item = record({ seed: Number(seedText), seed_text: seedText });
    load(node, [item]);
    assert.ok(select.children[0].textContent.includes(seedText));
    assert.ok(details.textContent.includes(seedText));
    response = { body: item };
    await restore.fire("click");
    assert.equal(values(node).retry_seed, seedText);
    assert.equal(values(node).seed, original.seed);
    assert.match(help.textContent, /默认使用随机种子重抽/);
    assert.match(help.textContent, /原种子.*片段记录的实际种子/);
    assert.match(help.textContent, /自定义种子.*完整整数/);
    assert.doesNotMatch(help.textContent + status.textContent, /retry_prompt|retry_seed/);
    exit.fire("click");
    assert.deepEqual(values(node), original);
  }
});

test("other node types keep their original hooks", () => {
  const { node, originals, prototype } = createNode("H3DigitalHuman");
  for (const [name, hook] of Object.entries(originals)) assert.equal(prototype[name], hook);
  assert.equal(node.panel, undefined);
});

test("language changes refresh review labels and status without reloading media or editing parameters", async () => {
  const { node, select, previewSelect, details, video, restore, refresh, exit, status, help, info } = createNode();
  const item = record({ diagnostics: { ...record().diagnostics,
    messages: ["新段 0.25 秒处疑似画面跳变。", "大动作或换光也可能触发，请结合预览判断。"] } });
  load(node, [item]);
  response = { body: item };
  await restore.fire("click");
  node.onExecuted({ h3_sgun_info: ["用户原文 user content", "seed: 938"] });
  const previousValues = values(node);
  const previousLoads = video.loads;
  const previousSrc = video.src;
  const previousListeners = localeListeners.size;
  video.currentTime = 0.75;
  try {
    setLocale("en");
    assert.equal(refresh.textContent, "Refresh saved segments");
    assert.equal(restore.textContent, "Restore settings for retry");
    assert.equal(exit.textContent, "Exit retry");
    assert.match(help.textContent, /new random seed on every run by default/);
    assert.match(help.textContent, /Original Seed uses the actual seed from the segment record/);
    assert.match(select.ariaLabel, /generated segment/);
    assert.match(previewSelect.ariaLabel, /segment to preview/);
    assert.match(select.children[0].textContent, /^Segment 2/);
    assert.match(details.textContent, /Seed 938/);
    assert.match(details.textContent, /Seam check: Possible seam issue/);
    assert.match(details.textContent, /Possible visual jump at 0.25 s/);
    assert.match(details.textContent, /Large movements or lighting changes/);
    assert.match(status.textContent, /Restored settings for segment 2/);
    assert.match(status.textContent, /selected retry seed mode/);
    assert.match(node.director.summary, /^Segment 2/);
    assert.equal(info.textContent, "用户原文 user content\nseed: 938");
    assert.equal(select.value, item.recipe_path);
    assert.equal(previewSelect.value, item.recipe_path);
    assert.equal(video.loads, previousLoads);
    assert.equal(video.src, previousSrc);
    assert.equal(video.currentTime, 0.75);
    assert.deepEqual(values(node), previousValues);
    node.onRemoved();
    assert.equal(localeListeners.size, previousListeners - 1);
  } finally {
    setLocale("zh");
  }
});

test("English review placeholders and restore errors follow the locale", async () => {
  setLocale("en");
  const { node, details, info, restore, status } = createNode();
  try {
    assert.match(details.textContent, /Select a segment after generation/);
    assert.match(info.textContent, /Actual seeds/);
    load(node);
    response = { body: record({ context_available: false }) };
    await restore.fire("click");
    assert.match(status.textContent, /no retry context/);
    setLocale("zh");
    assert.match(status.textContent, /缺少重抽上下文/);
  } finally {
    node.onRemoved();
    setLocale("zh");
  }
});

function cleanupResult(overrides = {}) {
  return { recycled_records: 3, recycled_files: 13, skipped_records: 0, errors: [], segments: [], ...overrides };
}

test("cleanup recycles listed records and supporting data, unloads preview and restores normal generation", async () => {
  const { node, restore, cleanup, select, previewSelect, video, info, status, exit } = createNode();
  assert.match(cleanup.title, /仅将列表中的重抽记录及辅助数据/);
  assert.match(cleanup.title, /保留原片段和重抽片段视频/);
  const original = values(node);
  const records = [1, 2, 3].map((segment) => record({ segment,
    recipe_path: `H3_00001/_data/segment_000${segment}.h3segment.json [output]` }));
  load(node, records);
  response = { body: records[2] };
  await restore.fire("click");
  widget(node, "filename_prefix").value = "H3";
  node.onExecuted({ h3_sgun_info: "old paths" });
  response = { body: cleanupResult(), onFetch(path, options) {
    assert.equal(path, "/h3-sgun/segments/cleanup-records");
    assert.equal(options.method, "POST");
    assert.equal(options.headers["Content-Type"], "application/json");
    assert.deepEqual(JSON.parse(options.body), { paths: records.map((entry) => entry.recipe_path) });
    assert.equal(video.paused, true);
    assert.equal(video.src, undefined);
    assert.equal(video.hidden, true);
  } };
  await cleanup.fire("click");
  assert.deepEqual(values(node), { ...original, filename_prefix: "H3" });
  assert.equal(select.children.length, 0);
  assert.equal(previewSelect.children.length, 0);
  assert.equal(video.src, undefined);
  assert.equal(video.hidden, true);
  assert.match(node.director.summary, /尚无片段记录/);
  assert.match(info.textContent, /生成后显示/);
  assert.match(status.textContent, /3 个片段记录及 13 个辅助文件移到回收站，片段视频保留/);
  assert.equal(cleanup.disabled, true);
  assert.equal(restore.disabled, true);
  assert.equal(exit.disabled, true);
  assert.equal(node.dirty, true);
  node.onRemoved();
});

test("cleanup asks for confirmation with the real record and batch scope, and a cancel sends nothing", async () => {
  const { node, cleanup, status } = createNode();
  const records = [
    record({ segment: 1, recipe_path: "H3_00001/_data/segment_0001.h3segment.json [output]" }),
    record({ segment: 2, recipe_path: "H3_00001/_data/segment_0002.h3segment.json [output]" }),
    record({ segment: 1, recipe_path: "H3_00007/_data/segment_0001.h3segment.json [output]" }),
    record({ segment: 1, recipe_path: "old_flat/segment_0001.h3segment.json [temp]" })];
  load(node, records);
  const prompts = [];
  try {
    context.confirm = (message) => { prompts.push(message); return false; };
    const count = requests.length;
    await cleanup.fire("click");
    assert.equal(requests.length, count);
    assert.match(prompts[0], /4 条片段记录（涉及 3 个批次）/);
    assert.match(prompts[0], /片段视频和保存的 LATENT 保留/);
    context.confirm = (message) => { prompts.push(message); return true; };
    response = { body: cleanupResult({ recycled_records: 4 }) };
    await cleanup.fire("click");
    assert.equal(requests.length, count + 1);
    assert.match(status.textContent, /保存的 LATENT 也保留/);
  } finally {
    delete context.confirm;
    node.onRemoved();
  }
});

test("cleanup clears stale retry fields from a loaded workflow without changing its other settings", async () => {
  const { node, cleanup } = createNode();
  widget(node, "retry_segment").value = "deleted/old.h3segment.json [output]";
  widget(node, "retry_prompt").value = "old retry";
  widget(node, "retry_seed").value = "18446744073709551615";
  widget(node, "segments").value = 3;
  const original = values(node);
  load(node);
  response = { body: cleanupResult({ recycled_records: 1 }) };
  await cleanup.fire("click");
  assert.deepEqual(values(node), { ...original, retry_segment: "", retry_prompt: "", retry_seed: "" });
  node.onRemoved();
});

test("partial cleanup keeps failed records playable and preserves an active retry that still exists", async () => {
  const { node, restore, cleanup, select, previewSelect, video, exit, status } = createNode();
  const original = values(node);
  const failed = record();
  const other = record({ recipe_path: "other.json [output]", segment: 3 });
  load(node, [other, failed]);
  response = { body: failed };
  await restore.fire("click");
  const retry = values(node);
  response = { body: cleanupResult({ recycled_records: 1, recycled_files: 4,
    errors: ["<img> File is in use"], segments: [failed] }) };
  await cleanup.fire("click");
  assert.deepEqual(values(node), retry);
  assert.deepEqual(select.children.map((entry) => entry.value), [failed.recipe_path]);
  assert.equal(previewSelect.value, failed.recipe_path);
  assert.equal(video.hidden, false);
  assert.ok(video.src);
  assert.match(status.textContent, /部分文件未能清理/);
  assert.match(status.textContent, /<img> File is in use/);
  assert.equal(status.children.length, 0);
  assert.equal(cleanup.disabled, false);
  assert.equal(exit.disabled, false);
  exit.fire("click");
  assert.deepEqual(values(node), original);
  node.onRemoved();
});

test("cleanup exits a removed retry while keeping imported records in the list", async () => {
  const { node, cleanup, restore, select, status, exit } = createNode();
  const original = values(node);
  load(node);
  response = { body: record() };
  await restore.fire("click");
  const imported = record({ recipe_path: "imported.json [input]" });
  response = { body: cleanupResult({ recycled_records: 1, skipped_records: 1, segments: [imported] }) };
  await cleanup.fire("click");
  assert.deepEqual(values(node), original);
  assert.equal(select.value, imported.recipe_path);
  assert.equal(exit.disabled, true);
  assert.match(status.textContent, /保留 1 条导入记录/);
  node.onRemoved();
});

test("partial cleanup exits a removed active retry even when a different record could not be recycled", async () => {
  const { node, cleanup, restore, select, exit } = createNode();
  const original = values(node);
  const removed = record();
  const failed = record({ recipe_path: "failed.json [output]", segment: 3 });
  load(node, [failed, removed]);
  response = { body: removed };
  await restore.fire("click");
  response = { body: cleanupResult({ recycled_records: 1, errors: ["failed file is busy"], segments: [failed] }) };
  await cleanup.fire("click");
  assert.deepEqual(values(node), original);
  assert.equal(select.value, failed.recipe_path);
  assert.equal(exit.disabled, true);
  node.onRemoved();
});

test("a rejected cleanup preserves entries, parameters and information and restores the preview", async () => {
  for (const statusCode of [400, 404, 409, 500]) {
    const { node, cleanup, select, video, status, info } = createNode();
    load(node);
    node.onExecuted({ h3_sgun_info: "saved information" });
    const original = values(node);
    const src = video.src;
    response = { ok: false, status: statusCode, body: { error: `cleanup rejected ${statusCode}` } };
    await cleanup.fire("click");
    assert.equal(select.children.length, 1);
    assert.equal(video.src, src);
    assert.equal(video.hidden, false);
    assert.equal(info.textContent, "saved information");
    assert.deepEqual(values(node), original);
    assert.match(status.textContent, new RegExp(`cleanup rejected ${statusCode}`));
    assert.equal(cleanup.disabled, false);
    node.onRemoved();
  }
});

test("cleanup is disabled while tasks run, wait in the queue or the server is disconnected", async () => {
  const { node, cleanup } = createNode();
  load(node);
  const count = requests.length;
  for (const remaining of [1, 4, null]) {
    queueStatus(remaining);
    assert.equal(cleanup.disabled, true);
    await cleanup.fire("click");
    assert.equal(requests.length, count);
  }
  queueStatus(0);
  assert.equal(cleanup.disabled, false);
  node.onRemoved();
});

test("cleanup cannot run twice while its request is pending", async () => {
  const { node, cleanup, refresh, restore } = createNode();
  load(node);
  let finish;
  response = { body: new Promise((resolve) => { finish = resolve; }) };
  const pending = cleanup.fire("click");
  const count = requests.length;
  assert.equal(cleanup.disabled, true);
  assert.equal(refresh.disabled, true);
  assert.equal(restore.disabled, true);
  await cleanup.fire("click");
  assert.equal(requests.length, count);
  finish(cleanupResult());
  await pending;
  node.onRemoved();
});

test("removing a node during cleanup removes its queue listener and ignores the late UI result", async () => {
  const { node, cleanup, status } = createNode();
  load(node);
  let finish;
  response = { body: new Promise((resolve) => { finish = resolve; }) };
  const pending = cleanup.fire("click");
  const listeners = apiListeners.get("status").size;
  const text = status.textContent;
  node.onRemoved();
  assert.equal(apiListeners.get("status").size, listeners - 1);
  finish(cleanupResult());
  await pending;
  assert.equal(status.textContent, text);
  assert.equal(node.director.removed, true);
  assert.equal(node.dirty, undefined);
});

test("cleanup labels and its result follow language changes, and empty lists never request cleanup", async () => {
  const { node, cleanup, status } = createNode();
  const count = requests.length;
  assert.equal(cleanup.disabled, true);
  await cleanup.fire("click");
  assert.equal(requests.length, count);
  setLocale("en");
  try {
    assert.equal(cleanup.textContent, "Clean up");
    assert.match(cleanup.title, /system Recycle Bin/);
    assert.match(cleanup.title, /Keep original and retry segment videos/);
    load(node);
    response = { body: cleanupResult({ recycled_records: 1 }) };
    await cleanup.fire("click");
    assert.match(status.textContent, /Moved 1 segment records and 13 supporting files to the Recycle Bin. Segment videos and saved LATENTs were kept/);
    setLocale("zh");
    assert.equal(cleanup.textContent, "一键清理");
    assert.match(status.textContent, /移到回收站/);
  } finally {
    node.onRemoved();
    setLocale("zh");
  }
});
