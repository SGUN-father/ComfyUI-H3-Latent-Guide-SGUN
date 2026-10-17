// Run with: node --experimental-vm-modules tests/test_long_video_widgets.mjs
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import vm from "node:vm";
import { test } from "node:test";

const context = vm.createContext({});
let extension;
const appModule = new vm.SyntheticModule(["app"], function () {
  this.setExport("app", { registerExtension(value) { extension = value; } });
}, { context });
const source = await readFile(new URL("../web/long_video_widgets.js", import.meta.url), "utf8");
const module = new vm.SourceTextModule(source, { context });
await module.link((specifier) => {
  assert.equal(specifier, "../../scripts/app.js");
  return appModule;
});
await module.evaluate();
assert.equal(extension.name, "H3LatentGuideSGUN.LongVideoWidgets");

function createNode(type = "H3LongVideo") {
  class Node {
    configure(info, ...args) {
      this.received = info;
      this.extraArgs = args;
      return this;
    }
  }
  const original = Node.prototype.configure;
  extension.beforeRegisterNodeDef(Node, { name: type });
  return { node: new Node(), original, prototype: Node.prototype };
}

function oldValues(merge, save, dynamic, drift) {
  const values = [4, 10.0, "16:9 (Widescreen)", 1.0, 261662374822964, "fixed",
    8, "er_sde", "simple", 22, "increment", 1.0, "match", merge, save,
    "video/my_latents/custom_name", dynamic];
  if (drift !== undefined) values.push(drift);
  return values;
}

function plain(value) {
  return JSON.parse(JSON.stringify(value));
}

test("old 17/18 widget layouts preserve every non-merge value and original input", () => {
  for (const merge of [false, true]) {
    for (const save of [false, true]) {
      for (const dynamic of [false, true]) {
        for (const drift of [undefined, false, true]) {
          const values = oldValues(merge, save, dynamic, drift);
          const info = { id: 56, widgets_values: values, properties: { untouched: true } };
          const original = structuredClone(info);
          const { node } = createNode();
          const marker = {};
          assert.equal(node.configure(info, marker, 7), node);
          assert.deepEqual(node.extraArgs, [marker, 7]);
          assert.deepEqual(plain(node.received.widgets_values), [...values.slice(0, 13), ...values.slice(14)]);
          assert.equal(node.received.widgets_values[13], save);
          assert.equal(node.received.widgets_values[14], "video/my_latents/custom_name");
          assert.equal(node.received.widgets_values[15], dynamic);
          if (drift !== undefined) assert.equal(node.received.widgets_values[16], drift);
          assert.equal(node.received.properties, info.properties);
          assert.deepEqual(info, original);
        }
      }
    }
  }
});

test("new layouts and a migrated second load are not migrated twice", () => {
  for (const drift of [undefined, false, true]) {
    const { node } = createNode();
    node.configure({ widgets_values: oldValues(true, false, true, drift) });
    const migrated = node.received;
    const snapshot = plain(migrated);
    node.configure(migrated);
    assert.equal(node.received, migrated);
    assert.deepEqual(plain(node.received), snapshot);
    const { node: fresh } = createNode();
    fresh.configure(migrated);
    assert.equal(fresh.received, migrated);
  }
});

test("other node types keep their configure method and data", () => {
  const { node, original, prototype } = createNode("H3DigitalHuman");
  assert.equal(prototype.configure, original);
  const info = { widgets_values: oldValues(false, true, true, false) };
  node.configure(info);
  assert.equal(node.received, info);
});

test("named widget maps only lose merge_video and preserve all other settings", () => {
  const named = { merge_video: false, save_latents: true, filename_prefix: "custom/latent",
    dynamic_mask: true, drift_correction: false, seed: 123 };
  for (const key of ["widgets_values", "widgets_values_named"]) {
    const info = { [key]: named };
    const { node } = createNode();
    node.configure(info);
    assert.deepEqual(plain(node.received[key]), {
      save_latents: true, filename_prefix: "custom/latent", dynamic_mask: true, drift_correction: false, seed: 123,
    });
    assert.equal(named.merge_video, false);
  }
});

test("positional and named representations migrate together without touching properties", () => {
  const info = { widgets_values: oldValues(true, true, false, true),
    widgets_values_named: { merge_video: true, save_latents: true, filename_prefix: "video/keep", dynamic_mask: false, drift_correction: true },
    properties: { arbitrary: [1, 2] } };
  const { node } = createNode();
  node.configure(info);
  assert.equal(node.received.widgets_values.length, 17);
  assert.equal(node.received.widgets_values[13], true);
  assert.equal(node.received.widgets_values[16], true);
  assert.equal(Object.hasOwn(node.received.widgets_values_named, "merge_video"), false);
  assert.equal(node.received.widgets_values_named.filename_prefix, "video/keep");
  assert.equal(node.received.properties, info.properties);
  assert.equal(info.widgets_values.length, 18);
  assert.equal(info.widgets_values_named.merge_video, true);
});

test("unknown shapes, arrays without old prefix signature, and empty info stay unchanged", () => {
  for (const info of [undefined, {}, { widgets_values: [] }, { widgets_values: { length: 18, 13: true } },
    { widgets_values: oldValues(true, true, false).concat(true, false) },
    { widgets_values: [...oldValues(true, true, false).slice(0, 15), false, false] }]) {
    const { node } = createNode();
    node.configure(info);
    assert.equal(node.received, info);
  }
});
