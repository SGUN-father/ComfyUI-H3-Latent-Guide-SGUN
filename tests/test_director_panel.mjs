// Run with: node --experimental-vm-modules --test tests/test_director_panel.mjs
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { test } from "node:test";
import vm from "node:vm";

class Element {
  constructor(tagName) {
    this.tagName = tagName;
    this.children = [];
    this.style = {};
    this.dataset = {};
    this.attributes = {};
    this.listeners = {};
    this.value = "";
    this.validationMessage = "";
    this.classList = { add: (...names) => { this.className = `${this.className ?? ""} ${names.join(" ")}`.trim(); } };
  }
  append(...children) { this.children.push(...children); }
  prepend(...children) { this.children.unshift(...children); }
  removeChild(child) { this.children.splice(this.children.indexOf(child), 1); }
  replaceChildren(...children) { this.children = children; }
  setAttribute(name, value) { this.attributes[name] = value; }
  addEventListener(name, callback) { this.listeners[name] = callback; }
  setCustomValidity(value) { this.validationMessage = value; }
  reportValidity() { this.reported = true; }
  querySelectorAll(selector) { return descendants(this).filter((item) => item.tagName === selector); }
  pause() { this.paused = (this.paused ?? 0) + 1; }
  remove() { this.removed = true; }
  fire(name) { return this.listeners[name]?.({ type: name, target: this }); }
}
function descendants(element) { return [element, ...element.children.flatMap(descendants)]; }

const document = {
  head: new Element("head"), activeElement: null,
  createElement: (name) => new Element(name),
  getElementById(id) { return descendants(this.head).find((item) => item.id === id); },
};
const context = vm.createContext({ document });
let locale = "zh";
const localeListeners = new Set();
const app = { canvas: { graph_mouse: [0, 0] }, ui: { settings: {
  getSettingValue() { return locale; },
  addEventListener(name, callback) { assert.equal(name, "Comfy.Locale.change"); localeListeners.add(callback); },
  removeEventListener(name, callback) { assert.equal(name, "Comfy.Locale.change"); localeListeners.delete(callback); },
} } };
function setLocale(value) { locale = value; for (const callback of [...localeListeners]) callback(); }
const appModule = new vm.SyntheticModule(["app"], function () { this.setExport("app", app); }, { context });
const catalogs = Object.fromEntries(await Promise.all(["en", "zh"].map(async (locale) => [locale, {
  nodeDefs: JSON.parse(await readFile(new URL(`../locales/${locale}/nodeDefs.json`, import.meta.url), "utf8")),
}])));
let translationRequests = 0;
const apiModule = new vm.SyntheticModule(["api"], function () {
  this.setExport("api", { async getCustomNodesI18n() { translationRequests++; return catalogs; } });
}, { context });
const i18nModule = new vm.SourceTextModule(await readFile(new URL("../web/i18n.js", import.meta.url), "utf8"), { context });
await i18nModule.link((specifier) => specifier === "../../scripts/app.js" ? appModule : apiModule);
await i18nModule.evaluate();
const samplingModule = new vm.SyntheticModule(["createSamplingOverlay"], function () {
  this.setExport("createSamplingOverlay", () => ({ remove() {} }));
}, { context });
const source = await readFile(new URL("../web/director_panel.js", import.meta.url), "utf8");
const module = new vm.SourceTextModule(source, { context });
await module.link((specifier) => {
  if (specifier === "./sampling_overlay.js") return samplingModule;
  if (specifier === "./i18n.js") return i18nModule;
  assert.equal(specifier, "../../scripts/app.js");
  return appModule;
});
await module.evaluate();
const { createDirectorPanel } = module.namespace;

const definitions = {
  segments: ["INT", { default: 3, min: 1, max: 10000, display_name: "生成段数" }],
  duration: ["FLOAT", { default: 10, min: 0.01, max: 150, display_name: "每段时长" }],
  aspect_ratio: [["16:9", "9:16"], { default: "16:9" }],
  seed: ["INT", { default: 333, min: 0, max: 18446744073709551615 }],
  context_frames: ["INT", { default: 22, min: 5, step: 17 }],
  dynamic_mask: ["BOOLEAN", { default: true }],
  live_preview: ["BOOLEAN", { default: false, display_name: "预览采样（增加耗时）" }],
  preview_frames: ["INT", { default: 12 }],
  preview_fps: ["INT", { default: 10, min: 1, max: 60, display_name: "预览帧率" }],
  preview_interval: ["INT", { default: 2, min: 1, max: 10000 }],
  preview_max_resolution: ["INT", { default: 768, min: 128, max: 2048 }],
  filename_prefix: ["STRING", { default: "video/H3/custom" }],
  retry_segment: ["STRING", { default: "" }],
  retry_prompt: ["STRING", { default: "", multiline: true }],
  retry_seed: ["STRING", { default: "" }],
  retry_seed_mode: [["randomize", "original", "custom"], { default: "randomize", display_name: "重抽种子方式" }],
};
function createNode(setup = () => {}, nodeDefinitions = definitions) {
  const changes = [];
  const callbacks = [];
  const widgets = Object.entries(nodeDefinitions).map(([name, [type, options]]) => ({
    name, value: options.default, type: Array.isArray(type) ? "combo" : type === "STRING" ? "text" : "number",
    options: Array.isArray(type) ? { values: [...type] } : { ...options },
    callback(value, canvas, node) { callbacks.push({ widget: this, value, canvas, node }); },
  }));
  widgets.push({ name: "control_after_generate", type: "combo", value: "randomize", options: { values: ["fixed", "randomize"] } });
  const unmanaged = { name: "unrelated", value: 42, type: "custom", hidden: false };
  widgets.push(unmanaged);
  const node = {
    widgets, size: [500, 1700], inputs: Array.from({ length: 13 }, (_, index) => ({ name: `ref_${index}`, link: index })),
    outputs: Array.from({ length: 5 }, () => ({})),
    graph: { beforeChange(value) { changes.push(["before", value]); }, afterChange(value) { changes.push(["after", value]); } },
    addDOMWidget(name, type, element, options) {
      const widget = { name, type, element, options };
      this.widgets.push(widget);
      this.domWidget = widget;
      return widget;
    },
    computeSize() {
      const slotHeight = Math.max(this.inputs.length, this.outputs.length) * 20;
      const widgetHeight = this.widgets.reduce((height, widget) => height + (widget.hidden ? 0 : widget.computeSize?.(500)?.[1] ?? 20), 0);
      return [this.size[0], 30 + slotHeight + widgetHeight];
    },
    setSize(value) { this.size = Array.from(value); },
    setDirtyCanvas() { this.dirty = true; },
  };
  const native = [...widgets];
  setup(node);
  const director = createDirectorPanel(node, { input: { required: nodeDefinitions } });
  const control = (name) => descendants(node.domWidget.element).find((item) => item.dataset.widget === name);
  const widget = (name) => native.find((item) => item.name === name);
  return { node, native, director, control, widget, callbacks, changes, unmanaged };
}
function plain(value) { return JSON.parse(JSON.stringify(value)); }

test("all views share a fixed panel while native widget order and serialization stay intact", () => {
  const { node, native, director } = createNode();
  const original = native.map((widget) => [widget, widget.name, widget.type, widget.value, widget.callback, widget.options]);
  const size = plain(node.domWidget.computeSize(500));
  for (const tab of ["generate", "retry", "preview", "info", "generate"]) {
    director.setTab(tab);
    assert.deepEqual(plain(node.domWidget.computeSize(500)), size);
    assert.equal(Object.values(director.sections).filter((section) => !section.hidden).length, 1);
    assert.equal(director.sections[tab].hidden, false);
    for (const button of descendants(node.domWidget.element).filter((item) => item.attributes.role === "tab")) {
      assert.equal(button.attributes["aria-selected"], String(button.dataset.tab === tab));
    }
  }
  assert.equal(node.domWidget.options.serialize, false);
  assert.equal(node.domWidget.options.getMinHeight(), node.domWidget.options.getMaxHeight());
  assert.equal(node.widgets.length, native.length + 1);
  original.forEach(([widget, name, type, value, callback, options], index) => {
    assert.equal(node.widgets[index], widget);
    assert.equal(widget.name, name);
    assert.equal(widget.type, type);
    assert.equal(widget.value, value);
    assert.equal(widget.callback, callback);
    assert.equal(widget.options, options);
  });
});

test("schema-named seed controls use canonical panel labels without renaming native widgets", async () => {
  const nodeDefinitions = {
    ...definitions,
    seed: ["INT", { ...definitions.seed[1], control_after_generate: "randomize" }],
  };
  let original;
  const { node, native, director, control, widget, callbacks } = createNode((node) => {
    const seedControl = node.widgets.find((item) => item.name === "control_after_generate");
    seedControl.name = "randomize";
    seedControl.options = { values: ["fixed", "increment", "decrement", "randomize"], serialize: false };
    seedControl.callback = function (value) { callbacks.push({ widget: this, value }); };
    original = node.widgets.map((item) => [item, item.name, item.value, item.options, item.callback]);
  }, nodeDefinitions);
  await i18nModule.namespace.loadPanelTranslations();
  const input = control("control_after_generate");
  const row = director.sections.generate.children.find((item) => item.children.includes(input));
  assert.ok(input);
  assert.ok(row);
  assert.equal(input.tagName, "select");
  assert.equal(director.seedControl, widget("randomize"));
  assert.equal(control("randomize"), undefined);
  assert.equal(input.value, "3");
  assert.equal(widget("randomize").hidden, true);
  for (const language of ["en", "zh"]) {
    setLocale(language);
    const translation = catalogs[language].nodeDefs.H3LongVideo.inputs.control_after_generate;
    assert.equal(row.children[0].textContent, translation.name);
    assert.equal(input.attributes["aria-label"], translation.name);
    assert.deepEqual(input.children.map((option) => option.textContent),
      ["fixed", "increment", "decrement", "randomize"].map((value) => translation.options[value]));
  }
  for (const [index, [item, name, value, options, callback]] of original.entries()) {
    assert.equal(node.widgets[index], item);
    assert.equal(item.name, name);
    assert.equal(item.value, value);
    assert.equal(item.options, options);
    assert.equal(item.callback, callback);
  }
  input.value = "0";
  input.fire("change");
  assert.equal(widget("randomize").value, "fixed");
  assert.equal(callbacks.length, 1);
  assert.equal(callbacks[0].widget, widget("randomize"));
  assert.equal(callbacks[0].value, "fixed");
  assert.equal(native.map((item) => item.name).filter((name) => name === "randomize").length, 1);
  director.remove();
  assert.equal(widget("randomize").hidden, undefined);
  assert.equal(widget("randomize").name, "randomize");
});

test("editing numeric, boolean and combo controls updates the owning native widgets and callbacks", () => {
  const { node, control, widget, callbacks, changes } = createNode();
  const duration = control("duration");
  duration.value = "7.5";
  duration.fire("change");
  assert.equal(widget("duration").value, 7.5);
  assert.equal(callbacks[0].widget, widget("duration"));
  assert.equal(callbacks[0].value, 7.5);
  assert.equal(callbacks[0].node, node);
  assert.equal(callbacks[0].canvas, app.canvas);
  assert.deepEqual(changes, [["before", node], ["after", node]]);
  control("dynamic_mask").value = "0";
  control("dynamic_mask").fire("change");
  assert.equal(widget("dynamic_mask").value, false);
  control("aspect_ratio").value = "1";
  control("aspect_ratio").fire("change");
  assert.equal(widget("aspect_ratio").value, "9:16");
  assert.equal(node.dirty, true);
});

test("external native value changes and dynamic combo choices appear when switching views", () => {
  const { director, control, widget } = createNode();
  widget("duration").value = 12.5;
  widget("dynamic_mask").value = false;
  const aspect = widget("aspect_ratio");
  aspect.options.values = function () { assert.equal(this, aspect); return ["16:9", "9:16", "1:1"]; };
  aspect.value = "1:1";
  director.setTab("generate");
  assert.equal(control("duration").value, "12.5");
  assert.equal(control("dynamic_mask").value, "0");
  assert.deepEqual(control("aspect_ratio").children.map((option) => option.textContent), ["16:9", "9:16", "1:1"]);
  assert.equal(control("aspect_ratio").value, "2");
});

test("linked scalar and converted widget controls cannot overwrite upstream values", () => {
  const { node, director, control, widget, callbacks } = createNode();
  for (const input of [{ name: "duration", link: 99 }, { name: "converted", widget: { name: "seed" }, link: 100 }]) {
    node.inputs.push(input);
    director.syncWidgets();
    const name = input.widget?.name ?? input.name;
    const original = widget(name).value;
    assert.equal(control(name).disabled, true);
    control(name).value = "1";
    control(name).fire("change");
    assert.equal(widget(name).value, original);
    assert.equal(callbacks.length, 0);
    input.link = null;
    director.syncWidgets();
    assert.equal(control(name).disabled, false);
  }
});

test("retry seed and multiline prompt preserve their exact text", () => {
  const { control, widget } = createNode();
  control("retry_seed_mode").value = "2";
  control("retry_seed_mode").fire("change");
  const seed = "18446744073709551615";
  const prompt = "  <Picture 1> holds her ground.\nA second line.\n";
  control("retry_seed").value = seed;
  control("retry_seed").fire("change");
  control("retry_prompt").value = prompt;
  control("retry_prompt").fire("change");
  assert.equal(widget("retry_seed").value, seed);
  assert.equal(widget("retry_prompt").value, prompt);
  assert.equal(control("retry_seed").type, "text");
  assert.equal(control("retry_prompt").tagName, "textarea");
});

test("typing valid text and numeric values commits before the field loses focus", () => {
  const { control, widget, callbacks } = createNode();
  for (const [name, text, expected] of [["retry_prompt", "Changed while still focused", "Changed while still focused"],
    ["retry_seed", "18446744073709551615", "18446744073709551615"], ["duration", "12.5", 12.5]]) {
    const input = control(name);
    document.activeElement = input;
    input.value = text;
    input.fire("input");
    assert.equal(widget(name).value, expected);
    const count = callbacks.length;
    input.fire("change");
    assert.equal(callbacks.length, count);
  }
  document.activeElement = null;
});

test("native setValue owns its callback and graph hooks instead of being invoked twice", () => {
  const nativeChanges = [];
  const { node, control, widget, callbacks } = createNode((node) => {
    const native = node.widgets.find((widget) => widget.name === "duration");
    native.setValue = function (value, options) {
      nativeChanges.push({ native: this, value, options });
      this.value = value;
      this.callback(value, options.canvas, options.node);
    };
  });
  control("duration").value = "6.25";
  control("duration").fire("input");
  control("duration").fire("change");
  assert.equal(nativeChanges.length, 1);
  assert.equal(nativeChanges[0].native, widget("duration"));
  assert.equal(nativeChanges[0].options.node, node);
  assert.equal(nativeChanges[0].options.canvas, app.canvas);
  assert.equal(nativeChanges[0].options.e.type, "input");
  assert.equal(callbacks.length, 1);
});

test("DOM widget value setters invoke callbacks once and update graph version", () => {
  const changes = [];
  const { node, control, callbacks } = createNode((node) => {
    const native = node.widgets.find((widget) => widget.name === "retry_prompt");
    native.element = new Element("textarea");
    let value = native.value;
    Object.defineProperty(native, "value", {
      get: () => value,
      set(next) { value = next; native.callback(next, app.canvas, node); },
    });
    node.onWidgetChanged = (...args) => changes.push(args);
    node.graph.incrementVersion = () => { node.version = (node.version ?? 0) + 1; };
  });
  control("retry_prompt").value = "A new prompt.";
  control("retry_prompt").fire("input");
  control("retry_prompt").fire("change");
  assert.equal(callbacks.length, 1);
  assert.equal(changes.length, 1);
  assert.equal(changes[0][0], "retry_prompt");
  assert.equal(changes[0][1], "A new prompt.");
  assert.equal(changes[0][2], "");
  assert.equal(node.version, 1);
});

test("refreshing preserves a focused draft until blur", () => {
  const { director, control, widget } = createNode();
  const input = control("retry_prompt");
  input.value = "a draft still being edited";
  document.activeElement = input;
  widget("retry_prompt").value = "a restored recipe";
  director.syncWidgets();
  assert.equal(input.value, "a draft still being edited");
  document.activeElement = null;
  input.fire("blur");
  assert.equal(input.value, "a restored recipe");
});

test("invalid numeric edits report an error without silently changing the submitted value", () => {
  const { node, director, control, widget, callbacks, changes } = createNode();
  const input = control("segments");
  const original = widget("segments").value;
  for (const value of ["", " ", "NaN", "Infinity", "1.5", "0"]) {
    input.value = value;
    document.activeElement = input;
    input.fire("change");
    assert.equal(widget("segments").value, original);
    assert.ok(input.validationMessage);
    director.syncWidgets();
    assert.equal(input.value, value);
  }
  assert.equal(callbacks.length, 0);
  assert.equal(changes.length, 0);
  input.value = "5";
  input.fire("input");
  input.fire("change");
  assert.equal(widget("segments").value, 5);
  assert.equal(input.validationMessage, "");
  assert.equal(node.dirty, true);
  document.activeElement = null;
});

test("resizing a loaded tall node uses its compact controls and preserves width", () => {
  const { node, director } = createNode();
  const before = node.size[1];
  director.resize();
  assert.equal(node.size[0], 500);
  assert.ok(node.size[1] < before);
  const size = [...node.size];
  for (const tab of ["retry", "preview", "info", "generate"]) {
    director.setTab(tab);
    director.resize();
    assert.deepEqual(node.size, size);
  }
  node.size[0] = 300;
  director.resize();
  assert.ok(node.size[0] >= 460);
});

test("removing the director restores original widget visibility and leaves unrelated widgets alone", () => {
  const { node, native, director, unmanaged } = createNode();
  assert.equal(unmanaged.hidden, false);
  assert.ok(native.filter((widget) => widget !== unmanaged).every((widget) => widget.hidden));
  director.remove();
  assert.equal(node.domWidget.element.removed, true);
  assert.equal(unmanaged.hidden, false);
  assert.ok(native.filter((widget) => widget !== unmanaged).every((widget) => widget.hidden === undefined));
  director.syncWidgets();
  director.resize();
  assert.ok(native.filter((widget) => widget !== unmanaged).every((widget) => widget.hidden === undefined));
});

test("summary text is displayed literally without creating HTML children", () => {
  const { node, director } = createNode();
  const text = "<img src=x onerror=bad> 18446744073709551615";
  director.setSummary(text);
  const status = descendants(node.domWidget.element).find((item) => item.attributes.role === "status");
  assert.equal(status.textContent, text);
  assert.equal(status.children.length, 0);
});

test("leaving the preview tab pauses its videos", () => {
  const { director } = createNode();
  const video = new Element("video");
  director.sections.preview.append(video);
  director.setTab("preview");
  assert.equal(video.paused, undefined);
  director.setTab("info");
  assert.equal(video.paused, 1);
});

test("sampling preview toggle is in advanced settings and defaults off", () => {
  const { director, control, widget } = createNode();
  assert.equal(widget("live_preview").value, false);
  assert.equal(control("live_preview").tagName, "select");
  assert.equal(control("live_preview").value, "0");
  assert.deepEqual(control("live_preview").children.map((option) => option.textContent), ["关", "开"]);
  const advanced = director.sections.generate.children.find((element) => element.tagName === "details");
  assert.ok(advanced.children.some((row) => row.children.includes(control("live_preview"))));
  assert.ok(!director.sections.generate.children.some((row) => row.children.includes(control("live_preview"))));
  control("live_preview").value = "1";
  control("live_preview").fire("change");
  assert.equal(widget("live_preview").value, true);
  control("live_preview").value = "0";
  control("live_preview").fire("change");
  assert.equal(widget("live_preview").value, false);
});

test("all preview controls stay together inside collapsed advanced settings while frame count stays hidden", () => {
  const { director, control, widget } = createNode();
  assert.equal(control("preview_frames"), undefined);
  assert.equal(widget("preview_frames").hidden, true);
  assert.equal(widget("preview_frames").value, 12);
  const advanced = director.sections.generate.children.find((element) => element.tagName === "details");
  assert.ok(!advanced.open);
  const previewRows = advanced.children.slice(-4);
  for (const [index, name] of ["live_preview", "preview_fps", "preview_interval", "preview_max_resolution"].entries()) {
    assert.ok(previewRows[index].children.includes(control(name)));
    assert.ok(!director.sections.generate.children.some((row) => row.children.includes(control(name))));
  }
  assert.equal(control("preview_fps").value, "10");
  control("preview_fps").value = "12";
  control("preview_fps").fire("change");
  assert.equal(widget("preview_fps").value, 12);
  director.remove();
  assert.equal(widget("preview_frames").hidden, undefined);
});

test("retry seed mode stays beside the conditional custom seed field in the retry tab", () => {
  const { node, director, control, widget } = createNode();
  const row = (name) => descendants(node.domWidget.element).find((item) => item.children.includes(control(name)));
  const mode = control("retry_seed_mode");
  const seedRow = row("retry_seed");
  assert.equal(widget("retry_seed_mode").value, "randomize");
  assert.equal(mode.tagName, "select");
  assert.equal(mode.value, "0");
  assert.equal(seedRow.hidden, true);
  const retryRows = director.sections.retry.children;
  assert.equal(retryRows.indexOf(seedRow), retryRows.indexOf(row("retry_seed_mode")) + 1);
  assert.ok(retryRows.includes(row("retry_seed_mode")));
  assert.ok(!descendants(director.sections.generate).includes(mode));
  mode.value = "1";
  mode.fire("change");
  assert.equal(widget("retry_seed_mode").value, "original");
  assert.equal(seedRow.hidden, true);
  mode.value = "2";
  mode.fire("change");
  assert.equal(widget("retry_seed_mode").value, "custom");
  assert.equal(seedRow.hidden, false);
  control("retry_seed").value = "18446744073709551615";
  control("retry_seed").fire("input");
  mode.value = "0";
  mode.fire("change");
  assert.equal(seedRow.hidden, true);
  assert.equal(widget("retry_seed").value, "18446744073709551615");
  widget("retry_seed_mode").value = "custom";
  director.syncWidgets();
  assert.equal(seedRow.hidden, false);
  assert.equal(mode.value, "2");
  assert.equal(control("retry_seed").value, "18446744073709551615");
  director.remove();
});

test("native locale changes translate dropdowns and metadata without changing serialized values", async () => {
  const { node, director, control, native, callbacks } = createNode();
  await i18nModule.namespace.loadPanelTranslations();
  const values = native.map((widget) => widget.value);
  const labels = () => descendants(node.domWidget.element).filter((item) => item.tagName === "label").map((item) => item.textContent);
  const previewRow = descendants(node.domWidget.element).find((row) => row.children.includes(control("live_preview")));
  const listeners = localeListeners.size;
  setLocale("en");
  assert.ok(labels().includes(catalogs.en.nodeDefs.H3LongVideo.inputs.segments.name));
  assert.equal(previewRow.title, catalogs.en.nodeDefs.H3LongVideo.inputs.live_preview.tooltip);
  assert.deepEqual(control("live_preview").children.map((option) => option.textContent), ["Off", "On"]);
  assert.deepEqual(control("retry_seed_mode").children.map((option) => option.textContent), ["Random Seed", "Original Seed", "Custom Seed"]);
  assert.deepEqual(native.map((widget) => widget.value), values);
  assert.equal(callbacks.length, 0);
  assert.equal(translationRequests, 1);
  setLocale("zh");
  assert.ok(labels().includes("生成段数"));
  assert.deepEqual(control("live_preview").children.map((option) => option.textContent), ["关", "开"]);
  assert.deepEqual(control("retry_seed_mode").children.map((option) => option.textContent), ["随机种子", "原种子", "自定义种子"]);
  director.remove();
  assert.equal(localeListeners.size, listeners - 1);
});
