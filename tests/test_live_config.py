from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from inferencedeck import live_config, manifest, remotes
from inferencedeck.config import AppConfig, ConfigRejected
from inferencedeck.control import ControlPlane
from inferencedeck.live_config import LiveFile, Rejected, rejected_files


class _Dir(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.dir, True)
        live_config.reset()
        self.addCleanup(live_config.reset)

    def write(self, name: str, data) -> Path:
        path = self.dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(data if isinstance(data, str) else json.dumps(data), encoding="utf-8")
        return path


def _parse_number(data: bytes) -> dict:
    try:
        value = json.loads(data)
    except json.JSONDecodeError as exc:
        raise Rejected([f"not valid JSON: {exc.msg}"]) from None
    if not isinstance(value, dict) or not isinstance(value.get("n"), int):
        raise Rejected(["n must be an integer"])
    return value


class LiveFileTests(_Dir):
    def test_keeps_the_last_good_version(self) -> None:
        path = self.dir / "f.json"
        parse = mock.Mock(side_effect=_parse_number)
        live = LiveFile(path, parse, "f.json")
        self.assertIsNone(live.get())  # missing

        self.write("f.json", {"n": 1})
        self.assertEqual(live.get(), {"n": 1})
        self.assertIsNone(live.state()["rejected"])

        self.write("f.json", '{"n": ')  # caught mid-save
        self.assertEqual(live.get(), {"n": 1})
        state = live.state()
        self.assertEqual(state["using"], "previous version")
        self.assertIn("not valid JSON", state["rejected"]["errors"][0])

        calls = parse.call_count
        live.get()
        live.get()
        self.assertEqual(parse.call_count, calls)  # the same bad bytes aren't re-parsed

        self.write("f.json", {"n": "two"})  # valid JSON, invalid content
        self.assertEqual(live.get(), {"n": 1})
        self.assertEqual(live.state()["rejected"]["errors"], ["n must be an integer"])

        self.write("f.json", {"n": 2})
        self.assertEqual(live.get(), {"n": 2})
        self.assertIsNone(live.state()["rejected"])

    def test_reverting_clears_the_rejection(self) -> None:
        live = LiveFile(self.dir / "f.json", _parse_number, "f.json")
        self.write("f.json", {"n": 1})
        live.get()
        self.write("f.json", "{")
        live.get()
        self.write("f.json", {"n": 1})
        self.assertEqual(live.get(), {"n": 1})
        self.assertIsNone(live.state()["rejected"])

    def test_never_valid_and_deleted(self) -> None:
        live = LiveFile(self.dir / "f.json", _parse_number, "f.json")
        self.write("f.json", "nope")
        self.assertIsNone(live.get())
        self.assertEqual(live.state()["using"], "defaults")
        self.write("f.json", {"n": 3})
        self.assertEqual(live.get(), {"n": 3})
        (self.dir / "f.json").unlink()
        self.assertIsNone(live.get())  # deleting the file means defaults, not the old version
        self.assertIsNone(live.state()["rejected"])

    def test_returns_copies(self) -> None:
        live = LiveFile(self.dir / "f.json", _parse_number, "f.json")
        self.write("f.json", {"n": 1, "list": [1]})
        live.get()["list"].append(2)
        self.assertEqual(live.get()["list"], [1])


class AppConfigReloadTests(_Dir):
    def test_bad_edits_keep_the_previous_settings(self) -> None:
        path = self.write("config.json", {"default_port": 9000})
        self.assertEqual(AppConfig.load(path).default_port, 9000)

        self.write("config.json", '{"default_port": 9100,')
        self.assertEqual(AppConfig.load(path).default_port, 9000)
        rejected = rejected_files()
        self.assertEqual([(r["label"], r["using"]) for r in rejected], [("config.json", "previous version")])

        self.write("config.json", {"default_port": "9100"})  # wrong type
        self.assertEqual(AppConfig.load(path).default_port, 9000)
        self.assertIn("default_port: must be integer", rejected_files()[0]["rejected"]["errors"][0])

        self.write("config.json", {"default_port": 9100, "unknown_key": 1})  # unknown keys are only warnings
        self.assertEqual(AppConfig.load(path).default_port, 9100)
        self.assertEqual(rejected_files(), [])

    def test_never_valid_file_means_defaults(self) -> None:
        path = self.write("config.json", "[1, 2]")
        self.assertEqual(AppConfig.load(path), AppConfig())
        self.assertEqual(rejected_files()[0]["using"], "defaults")

    def test_update_refuses_while_an_edit_is_rejected(self) -> None:
        path = self.write("config.json", {"default_port": 9000})
        AppConfig.load(path)
        self.write("config.json", '{"default_port": 91')
        with self.assertRaises(ConfigRejected):
            AppConfig.update(lambda c: setattr(c, "llama_runtime", "cpu"), path)
        self.assertEqual(path.read_text(encoding="utf-8"), '{"default_port": 91')  # the edit is untouched

        self.write("config.json", {"default_port": 9100})
        updated = AppConfig.update(lambda c: setattr(c, "llama_runtime", "cpu"), path)
        self.assertEqual((updated.default_port, updated.llama_runtime), (9100, "cpu"))
        self.assertEqual(AppConfig.load(path).llama_runtime, "cpu")

    def test_status_reports_rejections(self) -> None:
        path = self.write("config.json", {"default_port": 9000})
        AppConfig.load(path)
        self.write("config.json", "{")
        AppConfig.load(path)
        with mock.patch("inferencedeck.control.list_servers", return_value=[]), \
                mock.patch("inferencedeck.control.active_endpoint", return_value=None):
            status = ControlPlane().status()
        self.assertEqual(status["config_rejected"][0]["label"], "config.json")
        json.dumps(status)


class ManifestReloadTests(_Dir):
    def test_broken_models_json_keeps_the_profiles(self) -> None:
        good = {"models": [{"mode": "qwen", "name": "Qwen", "recommended_params": {"ctx_size": 8192}}]}
        self.write("models.json", good)
        self.assertEqual([p.mode for p in manifest.load_profiles(self.dir)], ["qwen"])
        self.write("models.json", '{"models": [{"mode": "qwen"')
        self.assertEqual([p.mode for p in manifest.load_profiles(self.dir)], ["qwen"])
        self.write("models.json", {"models": {"mode": "qwen"}})
        self.assertEqual([p.mode for p in manifest.load_profiles(self.dir)], ["qwen"])
        self.assertEqual(rejected_files()[0]["rejected"]["errors"], ['"models" must be a list'])

    def test_first_version_broken_means_no_profiles_not_a_crash(self) -> None:
        self.write("models.json", "{")
        self.assertEqual(manifest.load_profiles(self.dir), [])
        self.assertEqual(rejected_files()[0]["label"], "models.json")


class EndpointReloadTests(_Dir):
    def test_active_endpoint_survives_a_half_saved_file(self) -> None:
        endpoint = {"provider": "llamacpp", "lane": "remote_host", "baseUrl": "http://10.0.0.2:8080/v1", "enabled": True}
        self.write("ep/box.json", endpoint)
        self.assertEqual(remotes.active_endpoint(self.dir / "ep").base_url, "http://10.0.0.2:8080/v1")
        self.write("ep/box.json", '{"provider": "llamacpp", "enabled": tr')
        self.assertIsNotNone(remotes.active_endpoint(self.dir / "ep"))
        self.assertEqual(rejected_files()[0]["label"], "remote endpoint")

    def test_never_valid_endpoint_is_listed_as_invalid(self) -> None:
        self.write("ep/bad.json", "{")
        listed = remotes.list_endpoints(self.dir / "ep")
        self.assertEqual(len(listed), 1)
        self.assertFalse(listed[0].valid)
        self.assertTrue(listed[0].error.startswith("bad JSON"))
        self.write("ep/list.json", "[]")
        self.assertIn("must be a JSON object", remotes._parse(self.dir / "ep" / "list.json").error)


if __name__ == "__main__":
    unittest.main()
