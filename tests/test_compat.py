import importlib.util
from pathlib import Path
import sys
import unittest
from unittest import mock

COMFY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(COMFY_ROOT))
sys.argv = [sys.argv[0], "--cpu"]

spec = importlib.util.spec_from_file_location("h3_compat_test", Path(__file__).resolve().parents[1] / "compat.py")
compat = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compat)


class TestCompat(unittest.TestCase):
    def test_current_comfyui_meets_every_requirement(self):
        self.assertEqual(compat.missing_requirements(), [])
        compat.check_comfyui()

    def test_missing_feature_raises_readable_import_error(self):
        requirements = compat.REQUIREMENTS + (("测试功能 / test feature", "2099-01-01", lambda: False),)
        with mock.patch.object(compat, "REQUIREMENTS", requirements):
            with self.assertLogs(level="ERROR"):
                with self.assertRaises(ImportError) as raised:
                    compat.check_comfyui()
        message = str(raised.exception)
        self.assertIn(compat.MINIMUM, message)
        self.assertIn(compat.MINIMUM_RELEASE, message)
        self.assertIn("测试功能 / test feature", message)
        self.assertIn("2099-01-01", message)
        self.assertNotIn("H3 模型", message)

    def test_feature_probe_handles_missing_modules_and_attributes(self):
        self.assertTrue(compat._has("comfy.nested_tensor", "NestedTensor"))
        self.assertFalse(compat._has("comfy.nested_tensor", "NoSuchThing"))
        self.assertFalse(compat._has("comfy.no_such_module_h3_sgun", "anything"))
        self.assertTrue(compat._has("comfy_api.latest", "io.Autogrow.TemplatePrefix"))
        self.assertFalse(compat._has("comfy_api.latest", "io.NoSuchType.Input"))
        # Several dotted names in one module are each resolved from the module root.
        self.assertTrue(compat._has("comfy_api.latest", "io.Autogrow.TemplatePrefix", "io.RemoteOptions"))
        self.assertTrue(compat._has("comfy.nested_tensor", "NestedTensor", "NestedTensor"))
        self.assertFalse(compat._has("comfy_api.latest", "io.RemoteOptions", "io.NoSuchType"))

    def test_video_from_list_without_audio_parameter_is_rejected(self):
        self.assertTrue(compat._video_from_list_takes_audio())

        class OldVideoFromList:
            def __init__(self, videos, codec=None):
                pass

        with mock.patch("comfy_api.latest.InputImpl.VideoFromList", OldVideoFromList):
            self.assertFalse(compat._video_from_list_takes_audio())


if __name__ == "__main__":
    unittest.main()
