import { app } from "../../scripts/app.js";

app.registerExtension({
  name: "H3LatentGuideSGUN.LongVideoWidgets",
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
      return configure.call(this, info, ...args);
    };
  },
});
