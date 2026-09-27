from __future__ import annotations

import contextlib
import io
import json
import re
import shutil
import tempfile
import threading
import unittest
import urllib.request
from dataclasses import asdict
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest import mock

from inferencedeck import cli, config_check, webui
from inferencedeck.config import AppConfig
from inferencedeck.config_check import (
    CONFIG_SCHEMA,
    ENDPOINT_SCHEMA,
    KNOWN_PARAMS,
    PROFILES_SCHEMA,
    SCHEMAS,
    check_all,
    check_config,
    check_endpoints,
    check_profiles,
    format_report,
    validate_schema,
)
from inferencedeck.control import ControlPlane
from inferencedeck.control_api import ControlRequestHandler

REPO = Path(__file__).resolve().parents[1]


class _Dir(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.dir, True)

    def write(self, name: str, data) -> Path:
        path = self.dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(data if isinstance(data, str) else json.dumps(data), encoding="utf-8")
        return path

    def paths(self, problems) -> dict[str, str]:
        return {p.path: p.severity for p in problems}


class ValidatorTests(unittest.TestCase):
    def test_types_ranges_enums_and_unknown_keys(self) -> None:
        problems = validate_schema(
            {"default_port": 70000, "idel_release_seconds": 5, "concurrent_vram_check": "nope",
             "model_dirs": ["ok", 3], "auto_scan_on_startup": "yes", "default_host": ""},
            CONFIG_SCHEMA,
        )
        found = {path: (severity, message) for severity, path, message in problems}
        self.assertEqual(found["default_port"][0], "error")
        self.assertIn("did you mean 'idle_release_seconds'", found["idel_release_seconds"][1])
        self.assertEqual(found["idel_release_seconds"][0], "warning")
        self.assertIn("'block'", found["concurrent_vram_check"][1])
        self.assertEqual(found["model_dirs[1]"][0], "error")
        self.assertIn("must be boolean", found["auto_scan_on_startup"][1])
        self.assertEqual(found["default_host"][1], "must not be empty")

    def test_any_of_and_required(self) -> None:
        params = PROFILES_SCHEMA["properties"]["models"]["items"]["properties"]["recommended_params"]
        self.assertEqual(validate_schema({"gpu_layers": "all"}, params), [])
        self.assertEqual(validate_schema({"gpu_layers": 40}, params), [])
        self.assertEqual(validate_schema({"gpu_layers": "most"}, params)[0][1], "gpu_layers")
        self.assertEqual(validate_schema({}, PROFILES_SCHEMA), [("error", "models", "is required")])

    def test_bools_are_not_numbers(self) -> None:
        self.assertEqual(validate_schema(True, {"type": "integer"})[0][0], "error")
        self.assertEqual(validate_schema(1.5, {"type": "integer"})[0][0], "error")
        self.assertEqual(validate_schema(1, {"type": "number"}), [])


class ConfigFileTests(_Dir):
    def test_missing_file_is_fine(self) -> None:
        self.assertEqual(check_config(self.dir / "config.json"), [])

    def test_defaults_are_valid(self) -> None:
        self.assertEqual(check_config(self.write("config.json", asdict(AppConfig()))), [])

    def test_schema_reference_is_allowed(self) -> None:
        self.assertEqual(check_config(self.write("config.json", {"$schema": "https://example.invalid/config.schema.json"})), [])
        self.assertEqual(AppConfig.load(self.dir / "config.json"), AppConfig())

    def test_bad_json_says_defaults_are_used(self) -> None:
        problems = check_config(self.write("config.json", '{"default_port": 8080,}'))
        self.assertEqual([p.severity for p in problems], ["error", "error"])
        self.assertIn("not valid JSON (line 1", problems[0].message)
        self.assertIn("defaults", problems[1].message)

    def test_paths_that_do_not_exist(self) -> None:
        folder = self.dir / "models"
        folder.mkdir()
        problems = check_config(self.write("config.json", {
            "llama_server_path": str(self.dir / "nope" / "llama-server"),
            "model_dirs": [str(folder), str(self.dir / "gone")],
        }))
        self.assertEqual(self.paths(problems), {"llama_server_path": "warning", "model_dirs[1]": "warning"})


class ProfilesFileTests(_Dir):
    def test_good_profile(self) -> None:
        self.write("models.json", {"models": [{"mode": "qwen", "name": "Qwen", "recommended_params": {
            "ctx_size": 32768, "gpu_layers": "all", "cache_type_k": "q8_0", "flash_attn": True,
            "sampling_preset": "coding", "chat_template_kwargs": {"enable_thinking": False},
            "capabilities": {"tools": True, "input": ["text", "image"]}, "vision": True, "runtime": "llama.cpp",
        }}]})
        self.assertEqual(check_profiles(self.dir), [])

    def test_problems(self) -> None:
        self.write("models.json", {"models": [
            {"mode": "a", "recommended_params": {"ctx-size": 8192, "flash_attn": "on", "runtime": "ollama"}},
            {"mode": "a", "recommended_params": {"chat_template_kwargs": "[1]", "sampling_preset": "spicy"}},
            {"name": "no mode"},
        ]})
        found = self.paths(check_profiles(self.dir))
        self.assertEqual(found, {
            "models[0].recommended_params.ctx-size": "warning",
            "models[0].recommended_params.flash_attn": "error",
            "models[0].recommended_params.runtime": "warning",
            "models[1].mode": "error",
            "models[1].recommended_params.chat_template_kwargs": "error",
            "models[1].recommended_params.sampling_preset": "error",
            "models[2].mode": "error",
        })
        message = next(p.message for p in check_profiles(self.dir) if p.path.endswith("ctx-size"))
        self.assertIn("did you mean 'ctx_size'", message)

    def test_bad_json_and_missing_file(self) -> None:
        self.assertEqual(check_profiles(self.dir), [])
        self.write("models.json", "{models: []}")
        self.assertEqual([p.severity for p in check_profiles(self.dir)], ["error", "error"])


class EndpointFileTests(_Dir):
    def test_repository_examples_are_valid(self) -> None:
        self.assertEqual(check_endpoints(REPO / "examples" / "remote_endpoints"), [])

    def test_gateway_routing_keys(self) -> None:
        self.write("ep/box.json", {"baseUrl": "http://10.0.0.2:8080/v1", "lane": "remote_host", "aliases": ["qwen"], "routable": True})
        self.assertEqual(check_endpoints(self.dir / "ep"), [])
        self.write("ep/box.json", {"baseUrl": "http://10.0.0.2:8080/v1", "lane": "remote_host", "aliases": "qwen"})
        self.assertEqual([(p.path, p.severity) for p in check_endpoints(self.dir / "ep")], [("aliases", "error")])

    def test_problems(self) -> None:
        self.write("ep/keyed.json", {"baseUrl": "https://x.invalid/v1", "apiKeyEnv": "K", "apiKey": "sk-secret"})
        self.write("ep/lane.json", {"baseUrl": "https://x.invalid/v1", "apiKeyEnv": "K", "lane": "moon", "colour": 1})
        self.write("ep/cloud.json", {"baseUrl": "https://x.invalid/v1", "lane": "true_cloud"})
        self.write("ep/broken.json", "{")
        problems = check_endpoints(self.dir / "ep")
        by_file = {(Path(p.file).name, p.path): p for p in problems}
        self.assertEqual(by_file[("keyed.json", "apiKey")].severity, "error")
        self.assertNotIn("sk-secret", json.dumps([p.to_dict() for p in problems]))
        self.assertEqual(by_file[("lane.json", "lane")].severity, "error")
        self.assertEqual(by_file[("lane.json", "colour")].severity, "warning")
        self.assertIn("apiKeyEnv is required", by_file[("cloud.json", "")].message)
        self.assertEqual(by_file[("broken.json", "")].severity, "error")


class ReportTests(_Dir):
    def test_check_all_and_format(self) -> None:
        config = self.write("config.json", {"concurrent_vram_check": "maybe"})
        self.write("models.json", {"models": [{"mode": "a", "recommended_params": {"ctxsize": 1}}]})
        report = check_all(project_root=self.dir, config_path=config, endpoints=self.dir / "none")
        self.assertFalse(report["ok"])
        self.assertEqual((len(report["errors"]), len(report["warnings"])), (1, 1))
        text = format_report(report)
        self.assertIn("ERROR   ", text)
        self.assertTrue(text.endswith("1 error(s), 1 warning(s)."))
        self.assertEqual(format_report({"errors": [], "warnings": []}), "Configuration OK.")


class DriftTests(unittest.TestCase):
    def test_config_schema_matches_app_config(self) -> None:
        self.assertEqual(set(CONFIG_SCHEMA["properties"]) - {"$schema"}, config_check.config_field_names())
        self.assertEqual(validate_schema(asdict(AppConfig()), CONFIG_SCHEMA), [])

    def test_every_param_the_launch_code_reads_is_known(self) -> None:
        pattern = re.compile(r"""params(?:\.get)?[\[(]["']([a-z_0-9]+)["']""")
        read: set[str] = set()
        for source in (REPO / "inferencedeck").glob("*.py"):
            read |= set(pattern.findall(source.read_text(encoding="utf-8")))
        self.assertEqual(sorted(read - KNOWN_PARAMS), [])

    def test_committed_schema_files_are_current(self) -> None:
        files = {"config": "config.schema.json", "profiles": "models.schema.json", "endpoint": "remote-endpoint.schema.json"}
        for key, name in files.items():
            committed = json.loads((REPO / "schemas" / name).read_text(encoding="utf-8"))
            self.assertEqual(committed, SCHEMAS[key], f"schemas/{name} is stale: regenerate it with `inferencedeck config schema {key}`")


class CommandLineTests(_Dir):
    def run_cli(self, *args: str) -> tuple[int, str]:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = cli.main(list(args))
        return code, out.getvalue()

    def test_validate_exit_codes(self) -> None:
        good = self.write("good.json", {"default_port": 8080})
        warn = self.write("warn.json", {"defualt_port": 8080})
        bad = self.write("bad.json", {"default_port": "8080"})
        common = ["--project-root", str(self.dir), "--endpoints", str(self.dir / "none")]
        self.assertEqual(self.run_cli("config", "validate", "--config", str(good), *common), (0, "Configuration OK.\n"))
        self.assertEqual(self.run_cli("config", "validate", "--config", str(warn), *common)[0], 0)
        self.assertEqual(self.run_cli("config", "validate", "--config", str(warn), "--strict", *common)[0], 1)
        code, out = self.run_cli("config", "validate", "--config", str(bad), "--json", *common)
        self.assertEqual(code, 1)
        self.assertEqual(json.loads(out)["errors"][0]["path"], "default_port")

    def test_schema_command(self) -> None:
        code, out = self.run_cli("config", "schema", "endpoint")
        self.assertEqual((code, json.loads(out)), (0, ENDPOINT_SCHEMA))

    def test_web_check_config_flag(self) -> None:
        report = {"ok": False, "errors": [{"severity": "error", "file": "c.json", "path": "", "message": "bad"}], "warnings": []}
        out = io.StringIO()
        with mock.patch.object(webui, "check_all", return_value=report), \
                mock.patch.object(webui, "serve") as serve, \
                mock.patch("sys.argv", ["inferencedeck-web", "--check-config"]), \
                contextlib.redirect_stdout(out):
            code = webui.main()
        self.assertEqual(code, 1)
        serve.assert_not_called()
        self.assertIn("c.json: bad", out.getvalue())

    def test_web_warns_on_start_but_still_serves(self) -> None:
        report = {"ok": True, "errors": [], "warnings": [{"severity": "warning", "file": "c.json", "path": "x", "message": "odd"}]}
        err = io.StringIO()
        with mock.patch.object(webui, "check_all", return_value=report), \
                mock.patch.object(webui, "serve") as serve, \
                mock.patch("sys.argv", ["inferencedeck-web"]), \
                contextlib.redirect_stderr(err):
            self.assertEqual(webui.main(), 0)
        serve.assert_called_once()
        self.assertIn("c.json x: odd", err.getvalue())


class ApiTests(_Dir):
    def test_config_check_endpoint(self) -> None:
        handler = type("H", (ControlRequestHandler,), {})
        self.write("models.json", {"models": [{"mode": "a", "recommended_params": {"ctx_size": "big"}}]})
        handler.control_plane = ControlPlane(project_root=self.dir)
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        self.addCleanup(httpd.server_close)
        self.addCleanup(httpd.shutdown)
        with mock.patch.object(config_check, "check_config", return_value=[]), \
                mock.patch.object(config_check, "check_endpoints", return_value=[]), \
                urllib.request.urlopen(f"http://127.0.0.1:{httpd.server_address[1]}/api/config/check", timeout=5) as response:
            report = json.loads(response.read())
        self.assertFalse(report["ok"])
        self.assertEqual(report["errors"][0]["path"], "models[0].recommended_params.ctx_size")


if __name__ == "__main__":
    unittest.main()
