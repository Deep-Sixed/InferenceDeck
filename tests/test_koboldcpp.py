from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from inferencedeck import backends, fit, runtime_updates
from inferencedeck.config import AppConfig
from inferencedeck.koboldcpp_args import build_koboldcpp_args
from inferencedeck.launch_scripts import generate_launch_script
from inferencedeck.manifest import _parse_model_path
from inferencedeck.paths import executable_names, is_windows
from inferencedeck.profile_resolver import resolve_profiles
from inferencedeck.server_manager import READY_TIMEOUT_SECONDS, prepare_launch_command

OFFLINE = mock.patch.object(backends, "_request_json", return_value=(False, None, "offline"))
NO_KOBOLD_ENV = {"KOBOLDCPP_BIN": "", "KOBOLDCPP": "", "KOBOLDCPP_HOME": ""}


def _flag(argv: list[str], flag: str) -> str | None:
    return argv[argv.index(flag) + 1] if flag in argv else None


def _completed(stdout: str) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout, stderr="")


class KoboldCppArgsTests(unittest.TestCase):
    def test_translates_a_llama_style_profile(self) -> None:
        command = build_koboldcpp_args(
            ["/opt/kcpp/koboldcpp-linux-x64"],
            "/models/Qwen3-8B-Q4_K_M.gguf",
            {
                "host": "127.0.0.1",
                "port": 5002,
                "ctx_size": 32768,
                "threads": 8,
                "threads_batch": 8,
                "batch_size": 1024,
                "gpu_layers": 999,
                "cache_type_k": "q8_0",
                "cache_type_v": "q8_0",
                "flash_attn": False,
                "kv_offload": False,
                "mmap": True,
                "jinja": True,
                "reasoning": False,
                "draft_model": "/models/draft.gguf",
                "draft_max": 8,
                "tensor_overrides": ["blk\\.1=CPU", "blk\\.2=CPU"],
                "mmproj": "/models/mmproj.gguf",
                "n_predict": 512,
            },
            extra_args=["--quiet"],
        )
        argv = command.argv
        self.assertEqual(argv[:3], ["/opt/kcpp/koboldcpp-linux-x64", "--model", "/models/Qwen3-8B-Q4_K_M.gguf"])
        self.assertIn("--skiplauncher", argv)
        expected = {
            "--port": "5002",
            "--contextsize": "32768",
            "--threads": "8",
            "--blasthreads": "8",
            "--batchsize": "1024",
            "--gpulayers": "999",
            "--quantkv": "q8_0",
            "--jinjathink": "false",
            "--draftmodel": "/models/draft.gguf",
            "--draftamount": "8",
            "--overridetensors": "blk\\.1=CPU,blk\\.2=CPU",
            "--mmproj": "/models/mmproj.gguf",
            "--defaultgenamt": "512",
        }
        for flag, value in expected.items():
            self.assertEqual(_flag(argv, flag), value, flag)
        for flag in ("--noflashattention", "--lowvram", "--usemmap", "--jinja_tools"):
            self.assertIn(flag, argv)
        self.assertEqual(argv[-1], "--quiet")
        self.assertEqual(command.warnings, [])
        # No llama-server spellings leak through.
        for llama_flag in ("-m", "--ctx-size", "--gpu-layers", "--cache-type-k", "--flash-attn", "-ot"):
            self.assertNotIn(llama_flag, argv)

    def test_leaves_backend_threads_and_layers_to_koboldcpp_by_default(self) -> None:
        argv = build_koboldcpp_args(["koboldcpp"], "m.gguf", {"gpu_layers": "auto", "acceleration_backend": "auto", "device": "auto"}).argv
        for flag in ("--usecuda", "--usevulkan", "--usecpu", "--gpulayers", "--threads", "--noflashattention"):
            self.assertNotIn(flag, argv)

    def test_backend_and_device_selection(self) -> None:
        cuda = build_koboldcpp_args(["koboldcpp"], "m.gguf", {"acceleration_backend": "cuda", "device": "CUDA1"}).argv
        self.assertEqual(_flag(cuda, "--usecuda"), "1")
        vulkan = build_koboldcpp_args(["koboldcpp"], "m.gguf", {"acceleration_backend": "vulkan"}).argv
        self.assertIn("--usevulkan", vulkan)
        cpu = build_koboldcpp_args(["koboldcpp"], "m.gguf", {"acceleration_backend": "cpu", "gpu_layers": 999}).argv
        self.assertIn("--usecpu", cpu)
        self.assertEqual(_flag(cpu, "--gpulayers"), "0")

    def test_values_koboldcpp_would_reject_are_fixed_or_dropped_with_warnings(self) -> None:
        command = build_koboldcpp_args(
            ["koboldcpp"],
            "m.gguf",
            {
                "batch_size": 768,
                "cache_type_k": "q4_1",
                "n_predict": -1,
                "ubatch_size": 256,
                "op_offload": False,
                "temperature": 0.7,
            },
        )
        self.assertEqual(_flag(command.argv, "--batchsize"), "512")
        self.assertNotIn("--quantkv", command.argv)
        self.assertNotIn("--defaultgenamt", command.argv)
        text = " ".join(command.warnings)
        for fragment in ("using 512 for 768", "q4_1 was not applied", "--defaultgenamt", "ubatch_size", "op_offload", "temperature"):
            self.assertIn(fragment, text)

    def test_mismatched_kv_types_use_one_with_a_warning(self) -> None:
        command = build_koboldcpp_args(["koboldcpp"], "m.gguf", {"cache_type_k": "q8_0", "cache_type_v": "q4_0"})
        self.assertEqual(_flag(command.argv, "--quantkv"), "q8_0")
        self.assertIn("one KV cache type", " ".join(command.warnings))


class KoboldCppDetectionTests(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        backends._KOBOLDCPP_VERSIONS.clear()
        patches = [
            mock.patch.object(backends.shutil, "which", return_value=None),
            mock.patch.dict(os.environ, NO_KOBOLD_ENV),
            OFFLINE,
        ]
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)

    def _binary(self, name: str) -> Path:
        path = self.root / executable_names(name)[0]
        path.write_text("#!/bin/sh\n", encoding="utf-8")
        path.chmod(0o755)
        return path

    def test_release_binary_in_runtime_dirs_and_cached_version(self) -> None:
        binary = self._binary("koboldcpp-linux-x64")
        config = AppConfig(runtime_dirs=[str(self.root)])
        with mock.patch.object(backends, "run_hidden", return_value=_completed("1.122.1\n")) as run:
            env = backends.detect_koboldcpp(config)
            again = backends.detect_koboldcpp(config)
        self.assertEqual(env.binary_path, str(binary))
        self.assertEqual((env.version, again.version), ("1.122.1", "1.122.1"))
        self.assertEqual(env.details["version_source"], "binary")
        # A one-file build unpacks itself on every run: --version is asked once.
        run.assert_called_once()
        self.assertEqual(run.call_args.args[0], [str(binary), "--version"])

    def test_source_checkout_runs_under_python(self) -> None:
        script = self.root / "koboldcpp.py"
        script.write_text("print('x')\n", encoding="utf-8")
        with mock.patch.dict(os.environ, {"KOBOLDCPP_HOME": str(self.root)}):
            self.assertEqual(backends.koboldcpp_invocation(AppConfig()), [sys.executable, str(script)])

    def test_running_server_version_is_the_fallback(self) -> None:
        def fake_request(url: str, timeout: float = 0.8):
            if url.endswith("/v1/models"):
                return True, {"data": [{"id": "koboldcpp/qwen"}]}, None
            if url.endswith("/api/extra/version"):
                return True, {"result": "KoboldCpp", "version": "1.122.1"}, None
            return False, None, "unexpected"

        with mock.patch.object(backends, "_request_json", side_effect=fake_request):
            env = backends.detect_koboldcpp(AppConfig())
        self.assertTrue(env.available)
        self.assertEqual((env.version, env.details["version_source"]), ("1.122.1", "server"))

    def test_missing_koboldcpp_is_reported(self) -> None:
        env = backends.detect_koboldcpp(AppConfig())
        self.assertFalse(env.available)
        self.assertIn("KoboldCpp was not found", " ".join(env.warnings))

    def test_update_checks_cover_koboldcpp(self) -> None:
        self.assertEqual(runtime_updates.GITHUB_REPOS["koboldcpp"], "LostRuins/koboldcpp")
        self.assertEqual(runtime_updates.runtime_label("koboldcpp"), "KoboldCpp")
        # KoboldCpp tags both v1.122 and v1.122.1.
        self.assertEqual(runtime_updates.compare_versions("1.122", "v1.122.1"), -1)


class KoboldCppProfileTests(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.model_dir = self.root / "models"
        self.model_dir.mkdir()
        (self.model_dir / "Qwen3-8B-Q4_K_M.gguf").write_bytes(b"model")
        manifest = {
            "models": [
                {
                    "mode": "qwen-kobold",
                    "name": "Qwen3 8B on KoboldCpp",
                    "description": "KoboldCpp profile",
                    "recommended_params": {"runtime": "koboldcpp", "ctx_size": 16384},
                }
            ]
        }
        (self.root / "models.json").write_text(json.dumps(manifest), encoding="utf-8")
        self.binary = self.root / executable_names("koboldcpp")[0]
        self.binary.write_text("#!/bin/sh\n", encoding="utf-8")
        self.binary.chmod(0o755)
        backends._KOBOLDCPP_VERSIONS.clear()

    def test_profile_matches_gguf_and_leaves_tuning_to_koboldcpp(self) -> None:
        [profile] = resolve_profiles(project_root=self.root, model_dirs=[self.model_dir])
        self.assertTrue(profile.launchable, profile.missing)
        self.assertEqual(profile.model["name"], "Qwen3-8B-Q4_K_M")
        for key in ("threads_batch", "ubatch_size", "op_offload", "batch_size"):
            self.assertNotIn(key, profile.params)
        self.assertIn("reasoning", profile.params)

    def test_prepare_builds_koboldcpp_command(self) -> None:
        config = AppConfig(koboldcpp_path=str(self.binary), extra_koboldcpp_args=["--quiet"], extra_llama_args=["--no-webui"])
        with OFFLINE, mock.patch.object(backends, "run_hidden", return_value=_completed("1.122.1\n")):
            prepared = prepare_launch_command("qwen-kobold", self.root, [self.model_dir], config=config)
        self.assertTrue(prepared["success"], prepared)
        self.assertEqual(prepared["runtime"], "koboldcpp")
        argv = prepared["command"]["argv"]
        self.assertEqual(argv[:2], [str(self.binary), "--model"])
        self.assertTrue(argv[2].endswith("Qwen3-8B-Q4_K_M.gguf"))
        self.assertEqual(_flag(argv, "--contextsize"), "16384")
        self.assertIn("--quiet", argv)
        self.assertNotIn("--no-webui", argv)
        self.assertEqual(prepared["warnings"], [])
        self.assertGreater(READY_TIMEOUT_SECONDS["koboldcpp"], READY_TIMEOUT_SECONDS["llama.cpp"])

    def test_fit_refuses_koboldcpp_profiles(self) -> None:
        prepared = {"success": True, "runtime": "koboldcpp", "environment": {"details": {}}}
        with mock.patch.object(fit, "prepare_launch_command", return_value=prepared):
            result = fit.run_fit_test("qwen-kobold", self.root, [self.model_dir])
        self.assertFalse(result["success"])
        self.assertIn("koboldcpp", result["error"])


class KoboldCppLaunchScriptTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self._tmp, True)
        env = {
            "LCC_CONFIG_DIR": str(Path(self._tmp) / "config"),
            "LCC_CACHE_DIR": str(Path(self._tmp) / "cache"),
            "LCC_LAUNCH_SCRIPTS_DIR": str(Path(self._tmp) / "launch-scripts"),
        }
        patcher = mock.patch.dict(os.environ, env)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.project_root = Path(self._tmp) / "project"
        self.project_root.mkdir()
        self.model = self.project_root / "Qwen3-8B-Q4_K_M.gguf"
        self.model.write_bytes(b"model")

    def test_script_runs_the_release_binary(self) -> None:
        binary = self.project_root / executable_names("koboldcpp")[0]
        binary.write_bytes(b"binary")
        payload = generate_launch_script(
            mode="qwen-kobold",
            model_path=str(self.model),
            params={"runtime": "koboldcpp", "ctx_size": 16384},
            project_root=self.project_root,
            config=AppConfig(koboldcpp_path=str(binary)),
        )
        ps1 = Path(payload["ps1_path"]).read_text(encoding="utf-8")
        self.assertIn(f"& '{binary.as_posix()}' --model $model", ps1)
        self.assertIn("--contextsize 16384", ps1)
        self.assertIn("--skiplauncher", ps1)
        self.assertTrue(_parse_model_path(Path(payload["ps1_path"])).endswith("Qwen3-8B-Q4_K_M.gguf"))
        if not is_windows():
            sh = Path(payload["sh_path"]).read_text(encoding="utf-8")
            self.assertIn('--model "$model"', sh)
            self.assertIn('if [[ ! -f "$model" ]]', sh)

    def test_source_checkout_script_runs_under_python(self) -> None:
        script = self.project_root / "koboldcpp.py"
        script.write_text("", encoding="utf-8")
        payload = generate_launch_script(
            mode="qwen-kobold",
            model_path=str(self.model),
            params={"runtime": "koboldcpp"},
            project_root=self.project_root,
            config=AppConfig(koboldcpp_path=str(script)),
        )
        ps1 = Path(payload["ps1_path"]).read_text(encoding="utf-8")
        # The interpreter is written like every other script binary (forward slashes);
        # koboldcpp.py is an argument and keeps its native path.
        self.assertIn(f"& '{Path(sys.executable).as_posix()}' '{script}' --model $model", ps1)


if __name__ == "__main__":
    unittest.main()
