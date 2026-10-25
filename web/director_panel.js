import { app } from "../../scripts/app.js";
import { createSamplingOverlay } from "./sampling_overlay.js";
import { t, onLocaleChange, loadPanelTranslations, panelInput } from "./i18n.js";

const panelHeight = 440;
const tabs = [
  ["generate", "生成设置", "Generation"], ["retry", "片段重抽", "Retry Segment"],
  ["preview", "接缝预览", "Seam Preview"], ["info", "生成信息", "Generation Info"],
];
const basicNames = new Set(["segments", "duration", "aspect_ratio", "megapixels", "seed",
  "control_after_generate", "steps", "sampler_name", "scheduler"]);
const retryNames = ["retry_prompt", "retry_seed_mode", "retry_seed"];
const previewNames = ["live_preview", "preview_fps", "preview_interval", "preview_max_resolution"];

function installStyle() {
  if (document.getElementById("h3-director-style")) return;
  const style = document.createElement("style");
  style.id = "h3-director-style";
  style.textContent = `
    .h3-director-panel { position:relative; box-sizing:border-box; height:440px; min-height:0; overflow:hidden;
      display:flex; flex-direction:column; color:var(--input-text,#ddd); background:var(--comfy-menu-bg,#202328);
      border:1px solid var(--border-color,#4b5358); border-radius:8px; padding:9px; font:12px/1.45 sans-serif; }
    .h3-director-panel * { box-sizing:border-box; }
    .h3-director-header { display:flex; justify-content:flex-end; margin-bottom:8px; }
    .h3-director-tabs { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:4px; margin-bottom:8px; }
    .h3-director-panel button { color:inherit; background:var(--comfy-input-bg,#30353b);
      border:1px solid var(--border-color,#545c63); border-radius:4px; padding:5px 6px; cursor:pointer; }
    .h3-director-tabs button[aria-selected="true"] { background:#365b61; border-color:#83bbc4; color:#fff; }
    .h3-director-panel button:disabled { opacity:.5; cursor:default; }
    .h3-director-body { min-height:0; flex:1; overflow:hidden; }
    .h3-director-section { height:100%; min-height:0; overflow:auto; overscroll-behavior:contain; padding-right:3px; }
    .h3-director-panel [hidden] { display:none !important; }
    .h3-director-field { display:grid; grid-template-columns:minmax(110px,.85fr) minmax(0,1.3fr); gap:7px;
      align-items:center; margin-bottom:5px; min-height:27px; }
    .h3-director-field label { overflow-wrap:anywhere; }
    .h3-director-panel input, .h3-director-panel select, .h3-director-panel textarea {
      width:100%; min-width:0; padding:5px 6px; color:inherit; background:var(--comfy-input-bg,#121519);
      border:1px solid var(--border-color,#505960); border-radius:4px; font:inherit; }
    .h3-director-panel input:invalid, .h3-director-panel textarea:invalid { border-color:#db8572; }
    .h3-director-panel input:disabled, .h3-director-panel select:disabled, .h3-director-panel textarea:disabled { opacity:.5; }
    .h3-director-field-wide { display:block; }
    .h3-director-field-wide label { display:block; margin-bottom:5px; }
    .h3-director-panel textarea { min-height:150px; resize:vertical; font:12px/1.5 monospace; }
    .h3-director-hint { grid-column:2; opacity:.7; font-size:11px; }
    .h3-director-panel details { margin-top:7px; }
    .h3-director-panel summary { cursor:pointer; padding:6px 0; opacity:.85; }
    .h3-director-summary { flex:none; border-top:1px solid var(--border-color,#414a50); padding-top:6px; margin-top:7px;
      min-height:24px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; opacity:.85; }
    .h3-sampling-overlay { position:absolute; inset:0; z-index:2; background:#090909; display:flex; flex-direction:column; }
    .h3-sampling-overlay img { width:100%; min-height:0; flex:1; object-fit:contain; }
    .h3-sampling-toolbar { display:flex; align-items:center; justify-content:space-between; gap:8px; padding:8px; }
    .h3-sampling-toolbar span { min-width:0; overflow-wrap:anywhere; }
    .h3-sampling-toolbar button { flex:none; }
  `;
  document.head.append(style);
}

export function createDirectorPanel(node, nodeData) {
  installStyle();
  const element = document.createElement("div");
  element.className = "h3-director-panel";
  const header = document.createElement("div");
  header.className = "h3-director-header";
  const badge = document.createElement("span");
  badge.textContent = t("分段生成 · 连续镜头", "Segmented generation · Continuous shots");
  badge.style.opacity = ".65";
  header.append(badge);
  const tabBar = document.createElement("div");
  tabBar.className = "h3-director-tabs";
  tabBar.setAttribute("role", "tablist");
  const body = document.createElement("div");
  body.className = "h3-director-body";
  const summary = document.createElement("div");
  summary.className = "h3-director-summary";
  summary.textContent = t("设置参数后运行；生成完成可选择片段重抽。", "Set parameters and run. Select a finished segment to retry.");
  summary.setAttribute("role", "status");
  const sections = {};
  const tabButtons = {};
  for (const [key, zh, en] of tabs) {
    const label = t(zh, en);
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = label;
    button.dataset.tab = key;
    button.setAttribute("role", "tab");
    button.addEventListener("click", () => setTab(key));
    tabBar.append(button);
    tabButtons[key] = button;
    const section = document.createElement("div");
    section.className = "h3-director-section";
    section.dataset.section = key;
    section.setAttribute("role", "tabpanel");
    section.setAttribute("aria-label", label);
    sections[key] = section;
    body.append(section);
  }
  element.append(header, tabBar, body, summary);
  const sampling = createSamplingOverlay(node, element);
  const advanced = document.createElement("details");
  const advancedTitle = document.createElement("summary");
  advancedTitle.textContent = t("高级设置与 LATENT 保存", "Advanced Settings & LATENT Saving");
  advanced.append(advancedTitle);
  const recipeDetails = document.createElement("details");
  const recipeTitle = document.createElement("summary");
  recipeTitle.textContent = t("片段记录路径", "Segment Record Path");
  recipeDetails.append(recipeTitle);

  const definitions = { ...nodeData.input?.required, ...nodeData.input?.optional };
  const seedControl = node.widgets.find((item) => item.name === "control_after_generate")
    ?? node.widgets.find((item) => item.name === definitions.seed?.[1]?.control_after_generate);
  const originals = [];
  const controls = [];
  const retryRows = new Map();
  const previewRows = new Map();
  for (const native of node.widgets ?? []) {
    const name = native === seedControl ? "control_after_generate" : native.name;
    if (native.name === "preview_frames") {
      originals.push([native, native.hidden]);
      native.hidden = true;
      continue;
    }
    const definition = definitions[name];
    if (!definition && name !== "control_after_generate") continue;
    const [type, options = {}] = definition ?? ["COMBO", {}];
    if (!["STRING", "INT", "FLOAT", "BOOLEAN", "COMBO"].includes(type) && !Array.isArray(type)) continue;
    const row = document.createElement("div");
    row.className = "h3-director-field";
    const label = document.createElement("label");
    label.textContent = native.label ?? options.display_name ?? (name === "control_after_generate" ? t("生成后种子", "Seed after generation") : name);
    row.title = options.tooltip ?? native.tooltip ?? "";
    const values = () => {
      const choices = native.options?.values;
      return typeof choices === "function" ? choices.call(native) : choices ?? (Array.isArray(type) ? type : []);
    };
    const boolean = type === "BOOLEAN";
    const combo = boolean || Array.isArray(type) || type === "COMBO" || native.type === "combo";
    const multiline = options.multiline || native.name === "retry_prompt";
    const input = document.createElement(combo ? "select" : multiline ? "textarea" : "input");
    input.dataset.widget = name;
    input.setAttribute("aria-label", label.textContent);
    if (!combo && !multiline) input.type = ["INT", "FLOAT"].includes(type) ? "number" : "text";
    if (type === "INT" || type === "FLOAT") {
      input.step = type === "INT" ? "1" : "any";
      if (options.min != null) input.min = String(options.min);
      if (options.max != null) input.max = String(options.max);
    }
    if (options.placeholder) input.placeholder = options.placeholder;
    if (multiline) row.classList.add("h3-director-field-wide");
    const hint = document.createElement("span");
    hint.className = "h3-director-hint";
    hint.textContent = t("由连线控制", "Controlled by a connection");
    hint.hidden = true;
    row.append(label, input, hint);
    let choices = [];
    function sync() {
      if (native.name === "retry_seed") row.hidden = node.widgets.find((item) => item.name === "retry_seed_mode")?.value !== "custom";
      const translated = panelInput(name);
      label.textContent = translated.name ?? native.label ?? options.display_name
        ?? (name === "control_after_generate" ? t("生成后种子", "Seed after generation") : name);
      row.title = translated.tooltip ?? native.tooltip ?? options.tooltip ?? "";
      hint.textContent = t("由连线控制", "Controlled by a connection");
      input.setAttribute("aria-label", label.textContent);
      const linked = node.inputs?.some((slot) => slot.link != null
        && (slot.name === native.name || slot.widget?.name === native.name));
      input.disabled = Boolean(linked);
      hint.hidden = !linked;
      if (combo) {
        const next = boolean ? [false, true] : values();
        if (next.length !== choices.length || next.some((value, index) => value !== choices[index])) {
          choices = [...next];
          input.replaceChildren();
          for (const [index, value] of choices.entries()) {
            const option = document.createElement("option");
            option.value = String(index);
            option.textContent = boolean ? (value ? t("开", "On") : t("关", "Off")) : String(value);
            input.append(option);
          }
        }
        for (const [index, option] of Array.from(input.children).entries()) {
          const value = choices[index];
          option.textContent = boolean ? (value ? t("开", "On") : t("关", "Off"))
            : translated.options?.[String(value)] ?? String(value);
        }
        input.value = String(choices.indexOf(native.value));
      } else if (document.activeElement !== input) input.value = String(native.value ?? "");
    }
    function commit(event) {
      if (input.disabled) return;
      input.setCustomValidity?.("");
      let value = input.value;
      if (combo) {
        const index = Number(input.value);
        if (!Number.isInteger(index) || index < 0 || index >= choices.length) return;
        value = choices[index];
      } else if (type === "INT" || type === "FLOAT") {
        value = Number(input.value);
        if (!input.value.trim() || !Number.isFinite(value) || (type === "INT" && !Number.isInteger(value))
          || (options.min != null && value < options.min) || (options.max != null && value > options.max)) {
          input.setCustomValidity(t("请输入范围内的有效数字", "Enter a valid number within the allowed range."));
          if (event.type === "change") input.reportValidity?.();
          return;
        }
      }
      if (native.value === value) return;
      node.graph?.beforeChange?.(node);
      const previous = native.value;
      if (native.setValue && !native.element && !native.component) {
        native.setValue(value, { e: event, node, canvas: app.canvas });
      } else {
        native.value = value;
        // DOMWidget's value setter already invokes its callback.
        if (!native.element && !native.component) native.callback?.(value, app.canvas, node, app.canvas.graph_mouse, event);
        node.onWidgetChanged?.(native.name, value, previous, native);
        node.graph?.incrementVersion?.();
      }
      node.graph?.afterChange?.(node);
      syncWidgets();
      node.setDirtyCanvas(true, true);
    }
    input.addEventListener("input", commit);
    input.addEventListener("change", commit);
    input.addEventListener("blur", sync);
    if (native.name === "retry_segment") recipeDetails.append(row);
    else if (retryNames.includes(native.name)) retryRows.set(native.name, row);
    else if (previewNames.includes(native.name)) previewRows.set(native.name, row);
    else (basicNames.has(name) ? sections.generate : advanced).append(row);
    originals.push([native, native.hidden]);
    native.hidden = true;
    controls.push(sync);
  }
  advanced.append(...previewNames.map((name) => previewRows.get(name)).filter(Boolean));
  sections.generate.append(advanced);
  sections.retry.append(...retryNames.map((name) => retryRows.get(name)).filter(Boolean));
  sections.retry.append(recipeDetails);
  const widget = node.addDOMWidget("h3_sgun_director", "h3_sgun_director", element, {
    serialize: false,
    hideOnZoom: false,
    getMinHeight: () => panelHeight,
    getMaxHeight: () => panelHeight,
  });
  loadPanelTranslations().then(syncWidgets);
  widget.computeSize = (width) => [Math.max(460, width), panelHeight];
  let removed = false;
  const unsubscribeLocale = onLocaleChange(() => {
    badge.textContent = t("分段生成 · 连续镜头", "Segmented generation · Continuous shots");
    advancedTitle.textContent = t("高级设置与 LATENT 保存", "Advanced Settings & LATENT Saving");
    recipeTitle.textContent = t("片段记录路径", "Segment Record Path");
    for (const [key, zh, en] of tabs) {
      tabButtons[key].textContent = t(zh, en);
      sections[key].setAttribute("aria-label", t(zh, en));
    }
    syncWidgets();
  });
  function syncWidgets() {
    if (removed) return;
    for (const sync of controls) sync();
    for (const [native] of originals) native.hidden = true;
  }
  function setTab(key) {
    if (!sections[key]) return;
    if (key !== "preview") for (const video of sections.preview.querySelectorAll("video")) video.pause();
    for (const [name, section] of Object.entries(sections)) {
      section.hidden = name !== key;
      tabButtons[name].setAttribute("aria-selected", String(name === key));
      tabButtons[name].tabIndex = name === key ? 0 : -1;
    }
    syncWidgets();
    node.setDirtyCanvas(true, true);
  }
  function resize() {
    if (removed) return;
    syncWidgets();
    const width = Math.max(460, node.size[0]);
    const size = node.computeSize([width, 0]);
    node.setSize([width, size[1]]);
    node.setDirtyCanvas(true, true);
  }
  setTab("generate");
  return {
    sections, seedControl, setTab, syncWidgets, resize,
    setSummary(text) { summary.textContent = text; summary.title = text; },
    remove() {
      if (removed) return;
      removed = true;
      unsubscribeLocale();
      sampling.remove();
      for (const [native, hidden] of originals) native.hidden = hidden;
      element.remove();
    },
  };
}
