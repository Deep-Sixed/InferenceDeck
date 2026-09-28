from __future__ import annotations

import json
import unittest
from unittest import mock

from inferencedeck import server_manager
from inferencedeck.api_params import validate_overrides
from inferencedeck.config import AppConfig
from inferencedeck.llama_args import build_llama_server_args
from inferencedeck.profile_resolver import ResolvedProfile
from inferencedeck.sampling import SAMPLING_PRESETS, layer_sampling_preset, request_defaults
from inferencedeck.schema import Environment

CODING = SAMPLING_PRESETS["coding"]["params"]


class LayerPresetTests(unittest.TestCase):
    def test_profile_preset_sits_under_profile_values(self) -> None:
        merged, warnings = layer_sampling_preset({"sampling_preset": "coding", "temperature": 0.5})
        self.assertEqual(warnings, [])
        self.assertEqual(merged["temperature"], 0.5)  # the profile pinned it
        self.assertEqual(merged["top_k"], CODING["top_k"])
        self.assertEqual(merged["sampling_preset"], "coding")

    def test_override_preset_sits_over_profile_but_under_overrides(self) -> None:
        base = {"temperature": 0.8, "top_k": 10, "ctx_size": 4096}
        merged, _ = layer_sampling_preset(base, {"sampling_preset": "Coding", "top_k": 5})
        self.assertEqual(merged["temperature"], CODING["temperature"])  # preset beat the profile
        self.assertEqual(merged["top_k"], 5)  # explicit override beat the preset
        self.assertEqual(merged["ctx_size"], 4096)
        self.assertEqual(merged["sampling_preset"], "coding")

    def test_override_none_drops_profile_preset(self) -> None:
        merged, warnings = layer_sampling_preset({"sampling_preset": "creative", "top_p": 0.8}, {"sampling_preset": "none"})
        self.assertEqual(warnings, [])
        self.assertNotIn("sampling_preset", merged)
        self.assertNotIn("temperature", merged)
        self.assertEqual(merged["top_p"], 0.8)

    def test_unknown_preset_is_ignored_with_warning(self) -> None:
        merged, warnings = layer_sampling_preset({"sampling_preset": "spicy", "temperature": 0.4})
        self.assertEqual(merged, {"temperature": 0.4})
        self.assertEqual(len(warnings), 1)
        self.assertIn("spicy", warnings[0])

    def test_no_preset_is_a_plain_merge(self) -> None:
        merged, warnings = layer_sampling_preset({"temperature": 0.4}, {"temperature": 0.9})
        self.assertEqual((merged, warnings), ({"temperature": 0.9}, []))

    def test_request_defaults_summary(self) -> None:
        summary = request_defaults(
            {"sampling_preset": "coding", "temperature": 0.2, "n_predict": 512, "ctx_size": 8192,
             "top_k": None, "chat_template_kwargs": {"enable_thinking": False}}
        )
        self.assertEqual(summary["preset"], "coding")
        self.assertEqual(
            summary["values"],
            {"temperature": 0.2, "n_predict": 512, "chat_template_kwargs": {"enable_thinking": False}},
        )


class ChatTemplateKwargsTests(unittest.TestCase):
    def _args(self, **params) -> tuple[list[str], list[str]]:
        command = build_llama_server_args("/bin/llama-server", "/m.gguf", params)
        return command.argv, command.warnings

    def _flag(self, argv: list[str]) -> str | None:
        return argv[argv.index("--chat-template-kwargs") + 1] if "--chat-template-kwargs" in argv else None

    def test_dict_becomes_json_flag(self) -> None:
        argv, warnings = self._args(jinja=True, chat_template_kwargs={"reasoning_effort": "high", "enable_thinking": True})
        self.assertEqual(json.loads(self._flag(argv)), {"reasoning_effort": "high", "enable_thinking": True})
        self.assertEqual(warnings, [])

    def test_json_string_is_accepted(self) -> None:
        argv, _ = self._args(jinja=True, chat_template_kwargs='{"enable_thinking": false}')
        self.assertEqual(json.loads(self._flag(argv)), {"enable_thinking": False})

    def test_bad_values_are_dropped_with_warning(self) -> None:
        for bad in ("not json", "[1, 2]", ["enable_thinking"]):
            argv, warnings = self._args(jinja=True, chat_template_kwargs=bad)
            self.assertIsNone(self._flag(argv), bad)
            self.assertEqual(len(warnings), 1, bad)

    def test_empty_is_omitted_silently(self) -> None:
        for empty in (None, "", {}):
            argv, warnings = self._args(jinja=True, chat_template_kwargs=empty)
            self.assertIsNone(self._flag(argv))
            self.assertEqual(warnings, [])

    def test_warns_when_jinja_is_off(self) -> None:
        argv, warnings = self._args(chat_template_kwargs={"enable_thinking": False})
        self.assertIsNotNone(self._flag(argv))
        self.assertTrue(any("Jinja" in w for w in warnings))


class OverrideValidationTests(unittest.TestCase):
    def test_preset_override(self) -> None:
        self.assertEqual(validate_overrides({"sampling_preset": "coding"}), {"sampling_preset": "coding"})
        self.assertEqual(validate_overrides({"sampling_preset": "none"}), {"sampling_preset": "none"})
        for bad in ("spicy", 1, None, True):
            with self.assertRaises(ValueError):
                validate_overrides({"sampling_preset": bad})

    def test_chat_template_kwargs_stay_profile_only(self) -> None:
        with self.assertRaises(ValueError):
            validate_overrides({"chat_template_kwargs": {"enable_thinking": False}})


class PrepareLaunchTests(unittest.TestCase):
    def test_preset_reaches_launch_flags_and_defaults(self) -> None:
        profile = ResolvedProfile(
            mode="qwen",
            name="Qwen",
            description="",
            profile={},
            model={"path": "/models/qwen.gguf"},
            launchable=True,
            confidence=1.0,
            params={"sampling_preset": "coding", "temperature": 0.3, "jinja": True,
                    "chat_template_kwargs": {"enable_thinking": False}},
        )
        env = Environment(id="llama.cpp", kind="local", name="llama.cpp", available=True, binary_path="/bin/llama-server")
        with mock.patch.object(server_manager, "_profile_by_mode", return_value=profile), \
                mock.patch.object(server_manager, "detect_llama_cpp", return_value=env):
            prepared = server_manager.prepare_launch_command("qwen", project_root="/tmp", config=AppConfig())
        self.assertTrue(prepared["success"], prepared)
        argv = prepared["command"]["argv"]
        self.assertEqual(argv[argv.index("--temp") + 1], "0.3")
        self.assertEqual(argv[argv.index("--top-k") + 1], str(CODING["top_k"]))
        self.assertIn("--chat-template-kwargs", argv)
        defaults = prepared["request_defaults"]
        self.assertEqual(defaults["preset"], "coding")
        self.assertEqual(defaults["values"]["temperature"], 0.3)
        self.assertEqual(defaults["values"]["chat_template_kwargs"], {"enable_thinking": False})


if __name__ == "__main__":
    unittest.main()
