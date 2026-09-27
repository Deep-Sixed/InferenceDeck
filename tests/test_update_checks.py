"""Update checks for vllm.cpp and MLC LLM: installed-version detection and release lookup."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from inferencedeck import backends, runtime_updates
from inferencedeck.config import AppConfig
from inferencedeck.paths import executable_names

OFFLINE = mock.patch.object(backends, "_request_json", return_value=(False, None, "offline"))


def _completed(stdout: str) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout, stderr="")


class VllmCppVersionTests(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.binary = self.root / "bin" / executable_names("vllm-server")[0]
        self.binary.parent.mkdir()
        self.binary.write_text("#!/bin/sh\n", encoding="utf-8")
        self.config = AppConfig(vllm_cpp_server_path=str(self.binary))

    def test_release_archive_version_file_needs_no_process(self) -> None:
        (self.root / "release-manifest.json").write_text("{}", encoding="utf-8")
        (self.root / "VERSION").write_text("0.0.2\n", encoding="utf-8")
        with OFFLINE, mock.patch.object(backends, "run_hidden") as run:
            env = backends.detect_vllm_cpp(self.config)
        run.assert_not_called()
        self.assertEqual(env.version, "0.0.2")
        self.assertEqual(env.details["version_source"], "archive")

    def test_version_flag_output_is_parsed(self) -> None:
        with OFFLINE, mock.patch.object(backends, "run_hidden", return_value=_completed("vllm.cpp 0.0.3+cuda c-abi=29\n")) as run:
            env = backends.detect_vllm_cpp(self.config)
        self.assertEqual(run.call_args.args[0], [str(self.binary), "--version"])
        self.assertEqual(env.version, "0.0.3+cuda")
        self.assertEqual(env.details["version_source"], "binary")

    def test_running_server_version_is_the_fallback(self) -> None:
        def fake_request(url: str, timeout: float = 0.8):
            if url.endswith("/v1/models"):
                return True, {"data": [{"id": "qwen"}]}, None
            if url.endswith("/version"):
                return True, {"version": "0.0.2"}, None
            return False, None, "unexpected"

        with mock.patch.object(backends, "_request_json", side_effect=fake_request), mock.patch.object(
            backends, "run_hidden", return_value=_completed("")
        ):
            env = backends.detect_vllm_cpp(self.config)
        self.assertEqual(env.version, "0.0.2")
        self.assertEqual(env.details["version_source"], "server")


class MlcLlmVersionTests(unittest.TestCase):
    def test_python_module_uses_this_interpreters_wheel(self) -> None:
        with mock.patch.object(backends, "_mlc_llm_package_version", return_value="0.20.0"):
            self.assertEqual(backends._mlc_llm_version([sys.executable, "-m", "mlc_llm"]), "0.20.0")

    def test_unsynced_source_build_version_is_unknown(self) -> None:
        with mock.patch.object(backends, "_mlc_llm_package_version", return_value="0.1.dev0"):
            self.assertIsNone(backends._mlc_llm_version([sys.executable, "-m", "mlc_llm"]))

    def test_console_script_in_another_env_asks_its_interpreter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            other_python = Path(tmp) / "python"
            other_python.write_text("", encoding="utf-8")
            script = Path(tmp) / "mlc_llm"
            script.write_text(f"#!{other_python}\nimport sys\n", encoding="utf-8")
            with mock.patch.object(backends, "run_hidden", return_value=_completed("0.26.dev94\n")) as run:
                version = backends._mlc_llm_version([str(script)])
        self.assertEqual(version, "0.26.dev94")
        self.assertEqual(run.call_args.args[0][:2], [str(other_python), "-c"])

    def test_script_without_readable_interpreter_has_no_version(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            launcher = Path(tmp) / "mlc_llm.exe"
            launcher.write_bytes(b"MZ\x90\x00")
            with mock.patch.object(backends, "run_hidden") as run:
                self.assertIsNone(backends._mlc_llm_version([str(launcher)]))
            run.assert_not_called()


class ReleaseLookupTests(unittest.TestCase):
    def test_new_runtimes_are_checked(self) -> None:
        self.assertEqual(runtime_updates.GITHUB_REPOS["vllm.cpp"], "mudler/vllm.cpp")
        self.assertEqual(runtime_updates.GITHUB_REPOS["mlc-llm"], "mlc-ai/mlc-llm")
        self.assertEqual(runtime_updates.runtime_label("mlc-llm"), "MLC LLM")

    def test_pep440_prereleases_are_recognized(self) -> None:
        for tag in ("v0.26.dev0", "v0.20.0rc1", "v0.20.0a1", "v1.2.3b2", "v0.0.2-alpha1-ci-test"):
            self.assertTrue(runtime_updates.is_prerelease_tag(tag), tag)
        for tag in ("v0.20.0", "b4500", "v0.0.2", "v0.0.1-m03-parity-27b-35b"):
            self.assertFalse(runtime_updates.is_prerelease_tag(tag), tag)

    def test_mlc_dev_and_vllm_cpp_local_versions_compare_sensibly(self) -> None:
        # A nightly past the latest release, and a dev build before it.
        self.assertEqual(runtime_updates.compare_versions("0.26.dev94", "v0.20.0"), 1)
        self.assertEqual(runtime_updates.compare_versions("0.20.dev5", "v0.20.0"), -1)
        # vllm.cpp's backend suffix doesn't affect the comparison.
        self.assertEqual(runtime_updates.compare_versions("0.0.2+cuda", "v0.0.2"), 0)

    def _tags(self, *names: str) -> list[dict]:
        return [{"name": name} for name in names]

    def test_falls_back_to_tags_when_repo_has_no_release(self) -> None:
        tags = self._tags("v0.26.dev0", "v0.20.0", "v0.19.0", "v0.20.dev0", "v0.1.dev0")

        def fake_request(url: str, timeout: float = 0):
            if "/releases" in url:
                return False, None, "HTTP Error 404: Not Found"
            return True, tags, None

        with mock.patch.object(runtime_updates, "_request_json", side_effect=fake_request):
            stable = runtime_updates.fetch_latest_release("mlc-ai/mlc-llm", "stable")
            pre = runtime_updates.fetch_latest_release("mlc-ai/mlc-llm", "prerelease")
        self.assertEqual((stable["ok"], stable["tag"], stable["source"]), (True, "v0.20.0", "tags"))
        self.assertEqual(stable["release_url"], "https://github.com/mlc-ai/mlc-llm/releases/tag/v0.20.0")
        self.assertEqual(pre["tag"], "v0.26.dev0")

    def test_empty_release_list_falls_back_to_tags(self) -> None:
        calls: list[str] = []

        def fake_request(url: str, timeout: float = 0):
            calls.append(url)
            if "/releases" in url:
                return True, [], None
            return True, self._tags("v0.0.2-alpha", "v0.0.3-rc1"), None

        with mock.patch.object(runtime_updates, "_request_json", side_effect=fake_request):
            result = runtime_updates.fetch_latest_release("mudler/vllm.cpp", "prerelease")
        self.assertEqual(result["tag"], "v0.0.3-rc1")
        self.assertTrue(calls[-1].endswith("/tags?per_page=100"))

    def test_published_release_does_not_touch_tags(self) -> None:
        calls: list[str] = []

        def fake_request(url: str, timeout: float = 0):
            calls.append(url)
            return True, {"tag_name": "v0.0.2"}, None

        with mock.patch.object(runtime_updates, "_request_json", side_effect=fake_request):
            result = runtime_updates.fetch_latest_release("mudler/vllm.cpp", "stable")
        self.assertEqual(result["tag"], "v0.0.2")
        self.assertEqual(len(calls), 1)

    def test_other_errors_are_reported_not_hidden_by_tags(self) -> None:
        with mock.patch.object(runtime_updates, "_request_json", return_value=(False, None, "HTTP Error 403: rate limited")) as req:
            result = runtime_updates.fetch_latest_release("mlc-ai/mlc-llm", "stable")
        self.assertFalse(result["ok"])
        self.assertIn("403", result["error"])
        self.assertEqual(req.call_count, 1)

    def test_check_reports_update_for_installed_mlc_and_vllm_cpp(self) -> None:
        latest = {"mudler/vllm.cpp": "v0.0.3", "mlc-ai/mlc-llm": "v0.20.0"}

        def fake_fetch(repo: str, channel: str, timeout: float = 0):
            return {"ok": True, "tag": latest[repo], "release_url": f"https://github.com/{repo}/releases", "error": None}

        environments = [
            {"id": "vllm.cpp", "version": "0.0.2+cuda"},
            {"id": "mlc-llm", "version": "0.26.dev94"},
        ]
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict("os.environ", {"LCC_CACHE_DIR": tmp}), mock.patch.object(
            runtime_updates, "fetch_latest_release", side_effect=fake_fetch
        ):
            result = runtime_updates.check_runtime_updates(environments, force_refresh=True)
        by_id = {item["runtime_id"]: item for item in result["updates"]}
        self.assertTrue(by_id["vllm.cpp"]["update_available"])
        self.assertFalse(by_id["mlc-llm"]["update_available"])
        self.assertEqual(by_id["mlc-llm"]["runtime_name"], "MLC LLM")


class UpdatesCommandTests(unittest.TestCase):
    def test_updates_command_checks_detected_runtimes(self) -> None:
        from inferencedeck import cli
        from inferencedeck.schema import Environment

        detected = [Environment(id="mlc-llm", kind="local_binary", name="MLC LLM", available=True, version="0.20.0")]
        with mock.patch.object(cli, "detect_all", return_value=detected), mock.patch.object(
            cli, "check_runtime_updates", return_value={"updates": []}
        ) as check, mock.patch("builtins.print") as printed:
            self.assertEqual(cli.main(["updates", "--channel", "prerelease", "--refresh"]), 0)
        environments = check.call_args.args[0]
        self.assertEqual(environments[0]["id"], "mlc-llm")
        self.assertEqual(check.call_args.kwargs, {"channel": "prerelease", "force_refresh": True})
        printed.assert_called_once()


if __name__ == "__main__":
    unittest.main()
