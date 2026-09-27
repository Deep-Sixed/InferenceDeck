from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from inferencedeck import backends, fit
from inferencedeck import launch_scripts as launch_scripts_module
from inferencedeck.config import AppConfig
from inferencedeck.launch_scripts import generate_all_launch_scripts, generate_launch_script
from inferencedeck.manifest import _parse_model_path
from inferencedeck.mlc_llm_args import build_mlc_llm_serve_args
from inferencedeck.paths import executable_names, is_windows
from inferencedeck.profile_resolver import resolve_profiles
from inferencedeck.server_manager import prepare_launch_command

OFFLINE = mock.patch.object(backends, "_request_json", return_value=(False, None, "offline"))
NO_MLC_ENV = {"MLC_LLM_BIN": ""}


def _flag(argv: list[str], flag: str) -> str | None:
    return argv[argv.index(flag) + 1] if flag in argv else None


class MlcLlmArgsTests(unittest.TestCase):
    def test_translates_profile_to_mlc_serve_flags(self) -> None:
        command = build_mlc_llm_serve_args(
            ["/usr/bin/mlc_llm"],
            "/models/Qwen3-8B-q4f16_1-MLC",
            {
                "host": "127.0.0.1",
                "port": 8082,
                "ctx_size": 16384,
                "max_num_seqs": 8,
                "gpu_memory_utilization": 0.85,
                "mlc_mode": "server",
                "device": "cuda:1",
                "model_lib": "/libs/qwen3-cuda.so",
                "enable_prefix_caching": False,
            },
            extra_args=["--enable-tracing"],
        )
        argv = command.argv
        self.assertEqual(argv[:3], ["/usr/bin/mlc_llm", "serve", "/models/Qwen3-8B-q4f16_1-MLC"])
        self.assertEqual(_flag(argv, "--port"), "8082")
        self.assertEqual(_flag(argv, "--mode"), "server")
        self.assertEqual(_flag(argv, "--device"), "cuda:1")
        self.assertEqual(_flag(argv, "--model-lib"), "/libs/qwen3-cuda.so")
        self.assertEqual(_flag(argv, "--prefix-cache-mode"), "disable")
        self.assertEqual(
            _flag(argv, "--overrides"),
            "context_window_size=16384;max_num_sequence=8;gpu_memory_utilization=0.85",
        )
        self.assertEqual(argv[-1], "--enable-tracing")
        self.assertEqual(command.warnings, [])

    def test_python_module_invocation_and_defaults(self) -> None:
        argv = build_mlc_llm_serve_args([sys.executable, "-m", "mlc_llm"], "HF://mlc-ai/Qwen3-8B-q4f16_1-MLC", {}).argv
        self.assertEqual(argv[:5], [sys.executable, "-m", "mlc_llm", "serve", "HF://mlc-ai/Qwen3-8B-q4f16_1-MLC"])
        for flag in ("--overrides", "--device", "--mode", "--prefix-cache-mode"):
            self.assertNotIn(flag, argv)

    def test_warns_about_llama_only_sampling_and_bad_mode(self) -> None:
        command = build_mlc_llm_serve_args(
            ["mlc_llm"], "m", {"gpu_layers": 999, "cache_type_k": "q8_0", "temperature": 0.7, "mlc_mode": "turbo"}
        )
        text = " ".join(command.warnings)
        self.assertIn("gpu_layers", text)
        self.assertIn("temperature", text)
        self.assertIn("turbo", text)


class MlcLlmProfileTests(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.model_dir = self.root / "models"
        self.model_dir.mkdir()
        # A GGUF with a similar name must NOT be matched for an MLC profile.
        (self.model_dir / "Qwen3-8B-Q4_K_M.gguf").write_bytes(b"model")
        self.mlc_folder = self.root / "mlc" / "Qwen3-8B-q4f16_1-MLC"
        self.mlc_folder.mkdir(parents=True)
        (self.mlc_folder / "mlc-chat-config.json").write_text("{}", encoding="utf-8")
        self.cli = self.root / "bin" / executable_names("mlc_llm")[0]
        self.cli.parent.mkdir()
        self.cli.write_text("#!/bin/sh\n", encoding="utf-8")
        self.cli.chmod(0o755)

    def _write_profiles(self, *params: dict) -> None:
        models = [
            {"mode": f"qwen-mlc-{i}", "name": "Qwen3 8B on MLC", "description": "MLC LLM profile", "recommended_params": p}
            for i, p in enumerate(params)
        ]
        (self.root / "models.json").write_text(json.dumps({"models": models}), encoding="utf-8")

    def _resolve(self):
        return resolve_profiles(project_root=self.root, model_dirs=[self.model_dir])

    def test_local_mlc_folder_resolves_without_gguf_matching(self) -> None:
        self._write_profiles({"runtime": "mlc-llm", "mlc_model": str(self.mlc_folder)})
        [profile] = self._resolve()
        self.assertTrue(profile.launchable, profile.missing)
        self.assertEqual(profile.model["path"], str(self.mlc_folder))
        self.assertEqual(profile.model["format"], "mlc")
        self.assertEqual(profile.confidence, 1.0)
        for key in ("batch_size", "flash_attn", "device", "reasoning"):
            self.assertNotIn(key, profile.params)

    def test_hf_id_is_launchable_and_bad_folder_is_not(self) -> None:
        self._write_profiles(
            {"runtime": "mlc-llm", "mlc_model": "HF://mlc-ai/Qwen3-8B-q4f16_1-MLC"},
            {"runtime": "mlc-llm", "mlc_model": str(self.model_dir)},
            {"runtime": "mlc-llm"},
        )
        hf, bad, missing = self._resolve()
        self.assertTrue(hf.launchable)
        self.assertEqual(hf.model["name"], "Qwen3-8B-q4f16_1-MLC")
        self.assertFalse(bad.launchable)
        self.assertIn("mlc-chat-config.json", " ".join(bad.warnings))
        self.assertFalse(missing.launchable)
        self.assertIn("mlc_model", " ".join(missing.warnings))

    def test_prepare_builds_mlc_serve_command(self) -> None:
        self._write_profiles({"runtime": "mlc-llm", "mlc_model": str(self.mlc_folder), "ctx_size": 8192})
        config = AppConfig(mlc_llm_path=str(self.cli), extra_mlc_llm_args=["--enable-tracing"])
        with OFFLINE:
            prepared = prepare_launch_command("qwen-mlc-0", self.root, [self.model_dir], config=config)
        self.assertTrue(prepared["success"], prepared)
        self.assertEqual(prepared["runtime"], "mlc-llm")
        argv = prepared["command"]["argv"]
        self.assertEqual(argv[:3], [str(self.cli), "serve", str(self.mlc_folder)])
        self.assertEqual(_flag(argv, "--overrides"), "context_window_size=8192")
        self.assertIn("--enable-tracing", argv)

    def test_falls_back_to_python_module_then_reports_missing(self) -> None:
        with mock.patch.object(backends.shutil, "which", return_value=None), mock.patch.dict(os.environ, NO_MLC_ENV):
            with mock.patch.object(backends.importlib.util, "find_spec", return_value=object()):
                self.assertEqual(backends.mlc_llm_invocation(AppConfig()), [sys.executable, "-m", "mlc_llm"])
            with mock.patch.object(backends.importlib.util, "find_spec", return_value=None), OFFLINE:
                env = backends.detect_mlc_llm(AppConfig())
        self.assertFalse(env.available)
        self.assertIn("MLC LLM was not found", " ".join(env.warnings))

    def test_fit_refuses_mlc_profiles(self) -> None:
        prepared = {"success": True, "runtime": "mlc-llm", "environment": {"details": {}}}
        with mock.patch.object(fit, "prepare_launch_command", return_value=prepared):
            result = fit.run_fit_test("qwen-mlc-0", self.root, [self.model_dir])
        self.assertFalse(result["success"])
        self.assertIn("mlc-llm", result["error"])


class MlcLlmLaunchScriptTests(unittest.TestCase):
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
        self.mlc_folder = self.project_root / "mlc" / "Qwen3-8B-q4f16_1-MLC"
        self.mlc_folder.mkdir(parents=True)
        (self.mlc_folder / "mlc-chat-config.json").write_text("{}", encoding="utf-8")
        self.cli = self.project_root / executable_names("mlc_llm")[0]
        self.cli.write_bytes(b"binary")
        self.config = AppConfig(mlc_llm_path=str(self.cli))

    def test_local_folder_script_serves_the_folder(self) -> None:
        payload = generate_launch_script(
            mode="qwen-mlc",
            model_path=str(self.mlc_folder),
            params={"runtime": "mlc-llm", "ctx_size": 8192},
            project_root=self.project_root,
            config=self.config,
        )
        ps1 = Path(payload["ps1_path"]).read_text(encoding="utf-8")
        self.assertIn(f"& '{self.cli.as_posix()}' serve $model", ps1)
        self.assertIn("Test-Path -LiteralPath $model", ps1)
        self.assertIn("--overrides 'context_window_size=8192'", ps1)
        self.assertTrue(_parse_model_path(Path(payload["ps1_path"])).endswith("Qwen3-8B-q4f16_1-MLC"))
        if not is_windows():
            sh = Path(payload["sh_path"]).read_text(encoding="utf-8")
            # A folder, so the guard checks existence rather than a regular file.
            self.assertIn('if [[ ! -e "$model" ]]', sh)
            self.assertIn('serve "$model"', sh)

    def test_hf_id_script_keeps_the_id_and_skips_the_guard(self) -> None:
        payload = generate_launch_script(
            mode="qwen-mlc-hf",
            model_path="HF://mlc-ai/Qwen3-8B-q4f16_1-MLC",
            params={"runtime": "mlc-llm"},
            project_root=self.project_root,
            config=self.config,
        )
        ps1 = Path(payload["ps1_path"]).read_text(encoding="utf-8")
        self.assertIn("$model = 'HF://mlc-ai/Qwen3-8B-q4f16_1-MLC'", ps1)
        self.assertNotIn("Test-Path", ps1)
        if not is_windows():
            self.assertNotIn("Missing model", Path(payload["sh_path"]).read_text(encoding="utf-8"))

    def test_python_module_fallback_renders_python_dash_m(self) -> None:
        with mock.patch.object(launch_scripts_module, "_mlc_llm_binary_for_generation", return_value=sys.executable):
            payload = generate_launch_script(
                mode="qwen-mlc",
                model_path=str(self.mlc_folder),
                params={"runtime": "mlc-llm"},
                project_root=self.project_root,
                config=AppConfig(),
            )
        ps1 = Path(payload["ps1_path"]).read_text(encoding="utf-8")
        self.assertIn(f"& '{Path(sys.executable).as_posix()}' -m mlc_llm serve $model", ps1)

    def test_scan_generates_script_for_mlc_profile(self) -> None:
        manifest = {
            "models": [
                {
                    "mode": "qwen-mlc",
                    "name": "Qwen on MLC",
                    "description": "MLC LLM profile",
                    "recommended_params": {"runtime": "mlc-llm", "mlc_model": str(self.mlc_folder)},
                }
            ]
        }
        (self.project_root / "models.json").write_text(json.dumps(manifest), encoding="utf-8")
        model_dir = self.project_root / "models"
        model_dir.mkdir()
        with OFFLINE:
            result = generate_all_launch_scripts(project_root=self.project_root, model_dirs=[model_dir], config=self.config)
        self.assertEqual(result.errors, [])
        [script] = result.generated
        self.assertEqual(script.mode, "qwen-mlc")
        self.assertIn(" serve $model", Path(script.ps1_path).read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
