import ast
import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def schema_definitions():
    definitions = {}
    for source in ROOT.glob("*.py"):
        tree = ast.parse(source.read_text(encoding="utf-8"))
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            for method in node.body:
                if not isinstance(method, ast.FunctionDef) or method.name != "define_schema":
                    continue
                schema = next(item.value for item in method.body if isinstance(item, ast.Return))
                definitions[node.name] = {item.arg: item.value for item in schema.keywords}
    return definitions


def constant_keyword(call, key, default=None):
    return next((item.value.value for item in call.keywords if item.arg == key), default)


class TestNodeLocales(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalogs = {
            locale: json.loads((ROOT / "locales" / locale / "nodeDefs.json").read_text(encoding="utf-8"))
            for locale in ("en", "zh")
        }
        cls.schemas = schema_definitions()

    def test_all_registered_nodes_have_both_languages(self):
        tree = ast.parse((ROOT / "__init__.py").read_text(encoding="utf-8"))
        extension = next(item for item in tree.body if isinstance(item, ast.ClassDef)
                         and item.name == "H3LatentGuideExtension")
        method = next(item for item in extension.body if isinstance(item, ast.AsyncFunctionDef)
                      and item.name == "get_node_list")
        registered = {item.id for item in next(item.value for item in method.body
                                              if isinstance(item, ast.Return)).elts}
        for catalog in self.catalogs.values():
            self.assertEqual(set(catalog), registered)

    def test_inputs_outputs_and_tooltips_cover_the_live_schemas(self):
        for node_name, schema in self.schemas.items():
            with self.subTest(node=node_name):
                input_calls = schema["inputs"].elts
                input_names = {call.args[0].value for call in input_calls}
                if "seed" in input_names:
                    input_names.add("control_after_generate")
                for locale, catalog in self.catalogs.items():
                    entry = catalog[node_name]
                    self.assertEqual(set(entry["inputs"]), input_names)
                    self.assertEqual(set(entry["outputs"]),
                                     {str(i) for i in range(len(schema["outputs"].elts))})
                    self.assertTrue(entry["display_name"])
                    self.assertTrue(entry["description"])
                    for call in input_calls:
                        translated = entry["inputs"][call.args[0].value]
                        self.assertTrue(translated["name"])
                        tooltip = constant_keyword(call, "tooltip")
                        if tooltip:
                            self.assertTrue(translated["tooltip"])
                            if locale == "zh":
                                self.assertEqual(translated["tooltip"], tooltip)
                    for i, call in enumerate(schema["outputs"].elts):
                        translated = entry["outputs"][str(i)]
                        self.assertTrue(translated["name"])
                        if constant_keyword(call, "tooltip"):
                            self.assertTrue(translated["tooltip"])

    def test_english_has_no_untranslated_chinese_text(self):
        self.assertIsNone(re.search(r"[\u3400-\u9fff]", json.dumps(self.catalogs["en"], ensure_ascii=False)))

    def test_translated_combo_options_preserve_backend_values(self):
        for catalog in self.catalogs.values():
            inputs = catalog["H3LongVideo"]["inputs"]
            seed_values = {"fixed", "increment", "decrement", "randomize"}
            self.assertEqual(set(inputs["control_after_generate"]["options"]), seed_values)
            self.assertEqual(set(inputs["seed_mode"]["options"]), seed_values)
            self.assertEqual(set(inputs["retry_seed_mode"]["options"]), {"randomize", "original", "custom"})
            self.assertEqual(set(inputs["ref_image_size"]["options"]), {"match", "max"})


if __name__ == "__main__":
    unittest.main()
