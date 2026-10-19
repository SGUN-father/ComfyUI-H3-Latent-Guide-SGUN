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

test("the old panel's empty slot becomes an off preview switch without shifting parameters", () => {
  const prefix = [...oldValues(true, false, true, true).filter((_, index) => index !== 13),
    "saved/segment.json [temp]", "Prompt remains exact", "18446744073709551615"];
  assert.equal(prefix.length, 20);
  for (const tail of [[""], ["", 2, 12, 768, ""]]) {
    const { node } = createNode();
    node.widgets = [...prefix.map((_, index) => ({ name: `original_${index}` })), { name: "live_preview" }];
    const info = { widgets_values: [...prefix, ...tail] };
    const snapshot = structuredClone(info);
    node.configure(info);
    assert.deepEqual(plain(node.received.widgets_values), [...prefix, false, ...tail.slice(1)]);
    assert.deepEqual(info, snapshot);
    const migrated = node.received;
    node.configure(migrated);
    assert.equal(node.received, migrated);
  }
});

test("named preview placeholders are normalized while explicit true and false stay intact", () => {
  for (const key of ["widgets_values", "widgets_values_named"]) {
    const { node } = createNode();
    const info = { [key]: { live_preview: "", retry_seed: "18446744073709551615" } };
    node.configure(info);
    assert.deepEqual(plain(node.received[key]), { live_preview: false, retry_seed: "18446744073709551615" });
    assert.equal(info[key].live_preview, "");
    for (const value of [true, false]) {
      const explicit = { [key]: { live_preview: value } };
      node.configure(explicit);
      assert.equal(node.received, explicit);
    }
  }
});

test("the previous panel placeholder becomes 10fps without shifting existing preview settings", () => {
  const { node } = createNode();
  const prefix = [...oldValues(true, false, true, true).filter((_, index) => index !== 13),
    "saved/segment.json [temp]", "Prompt remains exact", "18446744073709551615", true, 4, 12, 1024];
  assert.equal(prefix.length, 24);
  node.widgets = [...prefix.map((_, index) => ({ name: `original_${index}` })), { name: "preview_fps" }];
  const info = { widgets_values: [...prefix, ""], widgets_values_named: { preview_fps: "", preview_frames: 12 } };
  node.configure(info);
  assert.deepEqual(plain(node.received.widgets_values), [...prefix, 10]);
  assert.deepEqual(plain(node.received.widgets_values_named), { preview_fps: 10, preview_frames: 12 });
  assert.equal(info.widgets_values[24], "");
  assert.equal(info.widgets_values_named.preview_fps, "");
  const migrated = node.received;
  node.configure(migrated);
  assert.equal(node.received, migrated);
});

test("an old panel placeholder becomes random retry mode without changing saved settings", () => {
  const { node } = createNode();
  const prefix = [...oldValues(true, false, true, true).filter((_, index) => index !== 13),
    "saved/segment.json [output]", "Prompt remains exact", "18446744073709551615", true, 4, 12, 1024, 8];
  assert.equal(prefix.length, 25);
  node.widgets = [...prefix.map((_, index) => ({ name: `original_${index}` })), { name: "retry_seed_mode" }];
  const info = { widgets_values: [...prefix, ""], widgets_values_named: { retry_seed_mode: "", preview_fps: 8 } };
  node.configure(info);
  assert.deepEqual(plain(node.received.widgets_values), [...prefix, "randomize"]);
  assert.deepEqual(plain(node.received.widgets_values_named), { retry_seed_mode: "randomize", preview_fps: 8 });
  assert.equal(info.widgets_values[25], "");
  assert.equal(info.widgets_values_named.retry_seed_mode, "");
  const migrated = node.received;
  node.configure(migrated);
  assert.equal(node.received, migrated);
  for (const mode of ["randomize", "original", "custom"]) {
    const explicit = { widgets_values_named: { retry_seed_mode: mode } };
    node.configure(explicit);
    assert.equal(node.received, explicit);
  }
});

function oldOutputs() {
  return [
    { name: "Combined Video", type: "VIDEO", links: [10] },
    { name: "Last Latent", type: "LATENT", links: [11] },
    { name: "Info", type: "STRING", links: [12] },
    { name: "Segment Videos", type: "VIDEO", links: [13, 14] },
    { name: "Latent Paths", type: "STRING", links: [15], slot_index: 4 },
  ];
}

function outputGraph(version = 0.4) {
  const links = [
    [10, 1, 0, 2, 0, "VIDEO"],
    [11, 1, 1, 3, 0, "LATENT"],
    [12, 1, 2, 4, 0, "STRING"],
    [13, 1, 3, 5, 0, "VIDEO"],
    [14, 1, 3, 6, 0, "VIDEO"],
    [15, 1, 4, 7, 0, "STRING"],
    [16, 8, 3, 9, 0, "VIDEO"],
  ];
  return {
    version,
    nodes: [
      { id: 1, type: "H3LongVideo", outputs: oldOutputs(), widgets_values: [3, 10] },
      ...links.map(([id, , , target, , type]) => ({ id: target, type: "Consumer", inputs: [{ type, link: id }] })),
      { id: 8, type: "H3DigitalHuman", outputs: oldOutputs() },
    ],
    links: version === 0.4 ? links : links.map(([id, origin_id, origin_slot, target_id, target_slot, type]) => (
      { id, origin_id, origin_slot, target_id, target_slot, type }
    )),
    extra: { unrelated: { preserve: true } },
  };
}

for (const version of [0.4, 1]) {
  test(`workflow ${version} removes segment video links and moves only latent path links`, () => {
    const graph = outputGraph(version);
    const original = structuredClone(graph);
    extension.beforeConfigureGraph(graph);
    const source = graph.nodes[0];
    assert.deepEqual(source.outputs.map((output) => output.name), ["Combined Video", "Last Latent", "Info", "Latent Paths"]);
    assert.equal(source.outputs[3].slot_index, 3);
    assert.deepEqual(source.outputs[3].links, [15]);
    for (const id of [5, 6]) assert.equal(graph.nodes.find((node) => node.id === id).inputs[0].link, null);
    const expected = original.links.filter((link) => ![13, 14].includes(version === 0.4 ? link[0] : link.id));
    const latentLink = expected.find((link) => (version === 0.4 ? link[0] : link.id) === 15);
    if (version === 0.4) latentLink[2] = 3;
    else latentLink.origin_slot = 3;
    assert.deepEqual(graph.links, expected);
    assert.deepEqual(source.widgets_values, original.nodes[0].widgets_values);
    assert.deepEqual(graph.nodes.find((node) => node.id === 8), original.nodes.find((node) => node.id === 8));
    assert.deepEqual(graph.extra, original.extra);
    const migrated = structuredClone(graph);
    extension.beforeConfigureGraph(graph);
    assert.deepEqual(graph, migrated);
  });
}

test("a current four-output node preserves slot 3 latent paths and every graph link", () => {
  const graph = outputGraph(1);
  graph.nodes[0].outputs.splice(3, 1);
  graph.nodes[0].outputs[3].slot_index = 3;
  graph.links = graph.links.filter((link) => ![13, 14].includes(link.id));
  graph.links.find((link) => link.id === 15).origin_slot = 3;
  for (const id of [5, 6]) graph.nodes.find((node) => node.id === id).inputs[0].link = null;
  const snapshot = structuredClone(graph);
  extension.beforeConfigureGraph(graph);
  assert.deepEqual(graph, snapshot);
});

test("v1 floating links and native reroute metadata follow removal and slot movement", () => {
  const graph = outputGraph(1);
  graph.nodes[0].id = "1";
  graph.floatingLinks = [
    { id: 30, origin_id: "1", origin_slot: 3, target_id: -1, target_slot: -1, type: "VIDEO", parentId: 20 },
    { id: 31, origin_id: "1", origin_slot: 4, target_id: -1, target_slot: -1, type: "STRING", parentId: 21 },
    { id: 32, origin_id: "other", origin_slot: 3, target_id: -1, target_slot: -1, type: "VIDEO" },
  ];
  graph.reroutes = [{ id: 20, linkIds: [13, 30] }, { id: 21, linkIds: [15, 31] }];
  graph.extra.reroutes = [{ id: 22, linkIds: [10, 14] }];
  graph.extra.linkExtensions = [{ id: 10, parentId: 22 }, { id: 14, parentId: 22 }];
  graph.extra.linkPresentation = { 10: { label: "keep" }, 13: { hidden: true }, 14: { label: "remove" } };
  extension.beforeConfigureGraph(graph);
  assert.deepEqual(plain(graph.floatingLinks), [
    { id: 31, origin_id: "1", origin_slot: 3, target_id: -1, target_slot: -1, type: "STRING", parentId: 21 },
    { id: 32, origin_id: "other", origin_slot: 3, target_id: -1, target_slot: -1, type: "VIDEO" },
  ]);
  assert.deepEqual(plain(graph.reroutes), [{ id: 20, linkIds: [] }, { id: 21, linkIds: [15, 31] }]);
  assert.deepEqual(plain(graph.extra.reroutes), [{ id: 22, linkIds: [10] }]);
  assert.deepEqual(plain(graph.extra.linkExtensions), [{ id: 10, parentId: 22 }]);
  assert.deepEqual(graph.extra.linkPresentation, { 10: { label: "keep" } });
});

test("native nested subgraph definitions migrate within their own graph scope", () => {
  const child = outputGraph(1);
  const grandchild = outputGraph(1);
  child.definitions = { subgraphs: [grandchild] };
  const root = {
    version: 1,
    nodes: [{ id: 1, type: "OtherNode", outputs: oldOutputs() }],
    links: [{ id: 13, origin_id: 1, origin_slot: 3, target_id: 2, target_slot: 0, type: "VIDEO" }],
    definitions: { subgraphs: [child] },
  };
  const untouched = structuredClone({ nodes: root.nodes, links: root.links });
  extension.beforeConfigureGraph(root);
  assert.deepEqual({ nodes: root.nodes, links: root.links }, untouched);
  for (const graph of [child, grandchild]) {
    assert.equal(graph.nodes[0].outputs.length, 4);
    assert.equal(graph.links.some((link) => [13, 14].includes(link.id)), false);
    assert.equal(graph.links.find((link) => link.id === 15).origin_slot, 3);
    assert.equal(graph.nodes.find((node) => node.id === 5).inputs[0].link, null);
  }
});

test("only official SaveVideo nodes disconnected from old segment output are muted", () => {
  for (const version of [0.4, 1]) {
    const graph = outputGraph(version);
    const segmentSaver = graph.nodes.find((node) => node.id === 5);
    const alreadyMuted = graph.nodes.find((node) => node.id === 6);
    const combinedSaver = graph.nodes.find((node) => node.id === 2);
    for (const node of [segmentSaver, alreadyMuted, combinedSaver]) {
      node.type = "SaveVideo";
      node.inputs[0].name = "video";
    }
    segmentSaver.mode = 0;
    alreadyMuted.mode = 2;
    combinedSaver.mode = 0;
    graph.nodes.push({ id: 20, type: "SaveVideo", mode: 0, inputs: [{ name: "video", link: null }] });
    extension.beforeConfigureGraph(graph);
    assert.equal(segmentSaver.mode, 2);
    assert.equal(segmentSaver.inputs[0].link, null);
    assert.equal(alreadyMuted.mode, 2);
    assert.equal(combinedSaver.mode, 0);
    assert.equal(combinedSaver.inputs[0].link, 10);
    assert.equal(graph.nodes.find((node) => node.id === 20).mode, 0);
  }
});

test("ordinary processors and non-video saver inputs do not acquire a muted mode", () => {
  const graph = outputGraph(1);
  const processor = graph.nodes.find((node) => node.id === 5);
  processor.inputs[0].name = "video";
  const saver = graph.nodes.find((node) => node.id === 6);
  saver.type = "SaveVideo";
  saver.mode = 0;
  saver.inputs[0].name = "other";
  extension.beforeConfigureGraph(graph);
  assert.equal(processor.inputs[0].link, null);
  assert.equal(Object.hasOwn(processor, "mode"), false);
  assert.equal(saver.inputs[0].link, null);
  assert.equal(saver.mode, 0);
});
