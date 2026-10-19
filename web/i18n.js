import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

let panelInputs = {};
let translations;

export function t(zh, en) {
  const locale = app.ui.settings.getSettingValue("Comfy.Locale", "en");
  return String(locale).toLowerCase().startsWith("zh") ? zh : en;
}

export function onLocaleChange(callback) {
  const settings = app.ui.settings;
  settings.addEventListener("Comfy.Locale.change", callback);
  return () => settings.removeEventListener("Comfy.Locale.change", callback);
}

export function loadPanelTranslations() {
  return translations ??= api.getCustomNodesI18n().then((data) => {
    panelInputs = Object.fromEntries(["zh", "en"].map((locale) =>
      [locale, data[locale]?.nodeDefs?.H3LongVideo?.inputs ?? {}]));
  }).catch((error) => {
    console.warn("H3 panel translations could not be loaded", error);
  });
}

export function panelInput(name) {
  return panelInputs[t("zh", "en")]?.[name] ?? {};
}
