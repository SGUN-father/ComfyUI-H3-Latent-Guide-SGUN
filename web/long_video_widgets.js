import { app } from "../../scripts/app.js";

function migrateVideoOutputs(graph) {
  const oldNodes = new Set();
  const removedLinks = new Set();
  for (const node of graph.nodes ?? []) {
    if (node.type !== "H3LongVideo" || node.outputs?.length !== 5
      || node.outputs[3].type !== "VIDEO" || node.outputs[4].type !== "STRING") continue;
    oldNodes.add(String(node.id));
    for (const id of node.outputs[3].links ?? []) removedLinks.add(id);
    node.outputs.splice(3, 1);
    if (node.outputs[3].slot_index === 4) node.outputs[3].slot_index = 3;
  }
  if (oldNodes.size) {
    for (const key of ["links", "floatingLinks"]) {
      if (!Array.isArray(graph[key])) continue;
      graph[key] = graph[key].filter((link) => {
        const array = Array.isArray(link);
        if (!oldNodes.has(String(array ? link[1] : link.origin_id))) return true;
        const slot = array ? link[2] : link.origin_slot;
        if (slot === 3) {
          removedLinks.add(array ? link[0] : link.id);
          return false;
        }
        if (slot === 4) {
          if (array) link[2] = 3;
          else link.origin_slot = 3;
        }
        return true;
      });
    }
    for (const node of graph.nodes ?? []) {
      let disconnectedVideo = false;
      for (const input of node.inputs ?? []) {
        if (removedLinks.has(input.link)) {
          input.link = null;
          if (node.type === "SaveVideo" && input.name === "video") disconnectedVideo = true;
        }
      }
      if (disconnectedVideo && !node.inputs.some((input) => input.name === "video" && input.link != null)) node.mode = 2;
    }
    for (const reroute of [...(graph.reroutes ?? []), ...(graph.extra?.reroutes ?? [])]) {
      if (Array.isArray(reroute.linkIds)) reroute.linkIds = reroute.linkIds.filter((id) => !removedLinks.has(id));
    }
    if (Array.isArray(graph.extra?.linkExtensions)) {
      graph.extra.linkExtensions = graph.extra.linkExtensions.filter((link) => !removedLinks.has(link.id));
    }
    for (const id of removedLinks) {
      if (graph.extra?.linkPresentation) delete graph.extra.linkPresentation[id];
    }
  }
  for (const subgraph of graph.definitions?.subgraphs ?? []) migrateVideoOutputs(subgraph);
}

app.registerExtension({
  name: "H3LatentGuideSGUN.LongVideoWidgets",
  beforeConfigureGraph(graph) {
    migrateVideoOutputs(graph);
  },
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "H3LongVideo") return;
    const configure = nodeType.prototype.configure;
    nodeType.prototype.configure = function (info, ...args) {
      const values = info?.widgets_values;
      const oldArray = Array.isArray(values) && [17, 18].includes(values.length)
        && typeof values[13] === "boolean" && typeof values[14] === "boolean"
        && typeof values[15] === "string";
      const namedObject = values && !Array.isArray(values) && typeof values === "object"
        && !Object.hasOwn(values, "length") && Object.hasOwn(values, "merge_video");
      if (oldArray || namedObject || Object.hasOwn(info?.widgets_values_named ?? {}, "merge_video")) {
        info = { ...info };
        if (oldArray) {
          info.widgets_values = [...values];
          info.widgets_values.splice(13, 1);
        } else if (namedObject) {
          info.widgets_values = { ...values };
          delete info.widgets_values.merge_video;
        }
        if (info.widgets_values_named) {
          info.widgets_values_named = { ...info.widgets_values_named };
          delete info.widgets_values_named.merge_video;
        }
      }
      // The old panel's empty slot can occupy a newly appended preview input.
      for (const [name, fallback] of [["live_preview", false], ["preview_fps", 10], ["retry_seed_mode", "randomize"]]) {
        const index = this.widgets?.findIndex((widget) => widget.name === name) ?? -1;
        if (index >= 0 && Array.isArray(info?.widgets_values) && info.widgets_values[index] === "") {
          info = { ...info, widgets_values: [...info.widgets_values] };
          info.widgets_values[index] = fallback;
        }
        for (const key of ["widgets_values", "widgets_values_named"]) {
          if (!Array.isArray(info?.[key]) && info?.[key]?.[name] === "") {
            info = { ...info, [key]: { ...info[key], [name]: fallback } };
          }
        }
      }
      return configure.call(this, info, ...args);
    };
  },
});
