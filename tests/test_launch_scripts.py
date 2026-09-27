from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from inferencedeck import launch_scripts as launch_scripts_module
from inferencedeck import paths as paths_module
from inferencedeck.config import AppConfig
from inferencedeck.launch_scripts import (
    delete_launch_script,
    generate_all_launch_scripts,
    generate_launch_script,
    launch_scripts_scan_summary,
    list_launch_scripts,
    startup_autoscan_if_enabled,
)
from inferencedeck.manifest import _parse_model_path


class _IsolatedDirs(unittest.TestCase):
    """Each test gets a fresh config/cache/launch-scripts dir + project root."""

    def setUp(self) -> None:
        self._tmp = tempfile.mkdtemp()
        self._orig_cwd = os.getcwd()
        self.config_dir = Path(self._tmp) / "config"
        self.cache_dir = Path(self._tmp) / "cache"
        self.scripts_dir = Path(self._tmp) / "launch-scripts"
        self.config_dir.mkdir(parents=True)
        self.cache_dir.mkdir(parents=True)
        self.scripts_dir.mkdir(parents=True)

        self._orig_env = {
            "LCC_CONFIG_DIR": os.environ.get("LCC_CONFIG_DIR"),
            "LCC_CACHE_DIR": os.environ.get("LCC_CACHE_DIR"),
            "LCC_LAUNCH_SCRIPTS_DIR": os.environ.get("LCC_LAUNCH_SCRIPTS_DIR"),
            "LLAMA_SERVER": os.environ.get("LLAMA_SERVER"),
            "LLAMA_CPP_HOME": os.environ.get("LLAMA_CPP_HOME"),
            "VLLM_CPP_SERVER": os.environ.get("VLLM_CPP_SERVER"),
            "VLLM_CPP_SERVER_URL": os.environ.get("VLLM_CPP_SERVER_URL"),
        }
        os.environ["LCC_CONFIG_DIR"] = str(self.config_dir)
        os.environ["LCC_CACHE_DIR"] = str(self.cache_dir)
        os.environ["LCC_LAUNCH_SCRIPTS_DIR"] = str(self.scripts_dir)

        self.project_root = Path(self._tmp) / "project"
        self.project_root.mkdir()
        (self.project_root / "models.json").write_text('{"models": []}', encoding="utf-8")
        self.model_dir = self.project_root / "models"
        self.model_dir.mkdir()
        self.server_bin = self.project_root / ("llama-server.exe" if paths_module.is_windows() else "llama-server")
        self.server_bin.write_bytes(b"binary")
        os.environ["LLAMA_SERVER"] = str(self.server_bin)
        os.environ["LLAMA_CPP_HOME"] = str(self.project_root)
        self.vllm_bin = self.project_root / ("vllm-server.exe" if paths_module.is_windows() else "vllm-server")
        self.vllm_bin.write_bytes(b"binary")
        os.environ["VLLM_CPP_SERVER"] = str(self.vllm_bin)
        # Point the vllm.cpp readiness probe at a closed port so detection never
        # talks to a real server on the default 127.0.0.1:8000.
        os.environ["VLLM_CPP_SERVER_URL"] = "http://127.0.0.1:9"

        # Run inside the temp project so any code path that falls back to
        # find_project_root() (e.g. startup autoscan without an explicit
        # project_root) stays sandboxed and cannot touch a real models.json.
        os.chdir(self.project_root)

    def tearDown(self) -> None:
        os.chdir(self._orig_cwd)
        shutil.rmtree(self._tmp, ignore_errors=True)
        for key, value in self._orig_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def _seed_model(self, name: str) -> Path:
        path = self.model_dir / name
        path.write_bytes(b"model-bytes")
        return path


class GenerateSingleLaunchScriptTests(_IsolatedDirs):
    def test_generate_writes_ps1_and_sh(self) -> None:
        model_path = self._seed_model("Tiny-1B-Q8_0.gguf")
        payload = generate_launch_script(
            mode="tiny",
            model_path=str(model_path),
            params={"ctx_size": 4096, "threads": 4, "gpu_layers": 999, "cache_type_k": "q8_0", "cache_type_v": "q8_0"},
            project_root=self.project_root,
        )
        self.assertTrue(Path(payload["ps1_path"]).is_file())
        ps1 = Path(payload["ps1_path"]).read_text(encoding="utf-8")
        self.assertIn("$model =", ps1)
        self.assertIn("& '", ps1)
        self.assertIn("--ctx-size 4096", ps1)
        if paths_module.is_windows():
            # The POSIX companion is intentionally skipped on Windows.
            self.assertEqual(payload["sh_path"], "")
        else:
            self.assertTrue(Path(payload["sh_path"]).is_file())
            sh = Path(payload["sh_path"]).read_text(encoding="utf-8")
            self.assertIn("#!/usr/bin/env bash", sh)
            self.assertIn('exec ', sh)

    def test_generated_ps1_is_readable_by_manifest_parser(self) -> None:
        model_path = self._seed_model("Tiny-1B-Q8_0.gguf")
        payload = generate_launch_script(
            mode="tiny",
            model_path=str(model_path),
            params={"ctx_size": 4096, "threads": 4, "gpu_layers": 999, "cache_type_k": "q8_0", "cache_type_v": "q8_0"},
            project_root=self.project_root,
        )
        parsed = _parse_model_path(Path(payload["ps1_path"]))
        self.assertIsNotNone(parsed)
        self.assertTrue(parsed.replace("\\", "/").endswith("models/Tiny-1B-Q8_0.gguf".replace("/", os.sep).replace("\\", "/")) or parsed.endswith("Tiny-1B-Q8_0.gguf"))

    def test_overwrite_false_skips_existing(self) -> None:
        model_path = self._seed_model("Tiny-1B-Q8_0.gguf")
        first = generate_launch_script(
            mode="tiny",
            model_path=str(model_path),
            params={"ctx_size": 4096},
            project_root=self.project_root,
        )
        self.assertFalse(first["skipped"])
        second = generate_launch_script(
            mode="tiny",
            model_path=str(model_path),
            params={"ctx_size": 8192},
            project_root=self.project_root,
            overwrite=False,
        )
        self.assertTrue(second["skipped"])


class ScanAllLaunchScriptsTests(_IsolatedDirs):
    def test_scan_creates_starter_script_for_newly_added_model(self) -> None:
        self._seed_model("Tiny-1B-Q8_0.gguf")
        first = generate_all_launch_scripts(project_root=self.project_root, model_dirs=[self.model_dir])
        self.assertEqual(len(first.generated), 1)
        self.assertEqual(first.scanned_model_count, 1)
        self.assertEqual(first.errors, [])

        # Drop a brand new model that has no manifest entry yet.
        self._seed_model("NewModel-7B-Q4_K_M.gguf")
        second = generate_all_launch_scripts(project_root=self.project_root, model_dirs=[self.model_dir])
        self.assertEqual(second.scanned_model_count, 2)
        new_modes = {item.mode for item in second.generated}
        self.assertIn("newmodel-7b-q4_k_m", new_modes)

    def test_scan_uses_manifest_params_for_known_profiles(self) -> None:
        manifest = {
            "models": [
                {
                    "mode": "tiny",
                    "name": "Tiny",
                    "description": "test",
                    "recommended_params": {
                        "ctx_size": 4096,
                        "threads": 4,
                        "gpu_layers": 999,
                        "cache_type_k": "q8_0",
                        "cache_type_v": "q8_0",
                    },
                }
            ]
        }
        (self.project_root / "models.json").write_text(json.dumps(manifest), encoding="utf-8")
        self._seed_model("Tiny-1B-Q8_0.gguf")
        result = generate_all_launch_scripts(project_root=self.project_root, model_dirs=[self.model_dir])
        self.assertEqual(len(result.generated), 1)
        self.assertEqual(result.generated[0].mode, "tiny")
        ps1 = Path(result.generated[0].ps1_path).read_text(encoding="utf-8")
        self.assertIn("--ctx-size 4096", ps1)
        self.assertIn("--cache-type-k q8_0", ps1)

    def test_overwrite_false_keeps_existing_unchanged(self) -> None:
        manifest = {
            "models": [
                {
                    "mode": "tiny",
                    "name": "Tiny",
                    "description": "test",
                    "recommended_params": {
                        "ctx_size": 4096,
                        "threads": 4,
                        "gpu_layers": 999,
                        "cache_type_k": "q8_0",
                        "cache_type_v": "q8_0",
                    },
                }
            ]
        }
        (self.project_root / "models.json").write_text(json.dumps(manifest), encoding="utf-8")
        self._seed_model("Tiny-1B-Q8_0.gguf")
        first = generate_all_launch_scripts(project_root=self.project_root, model_dirs=[self.model_dir])
        self.assertEqual(len(first.generated), 1)
        # Bump ctx_size; with overwrite=False the script must not change.
        manifest["models"][0]["recommended_params"]["ctx_size"] = 8192
        (self.project_root / "models.json").write_text(json.dumps(manifest), encoding="utf-8")
        result = generate_all_launch_scripts(project_root=self.project_root, model_dirs=[self.model_dir], overwrite=False)
        self.assertEqual(len(result.generated), 0)
        self.assertEqual(len(result.skipped), 1)
        # Confirm the script still has the original ctx-size.
        ps1 = Path(first.generated[0].ps1_path).read_text(encoding="utf-8")
        self.assertIn("--ctx-size 4096", ps1)


class ListAndDeleteTests(_IsolatedDirs):
    def test_list_and_delete_round_trip(self) -> None:
        self._seed_model("Tiny-1B-Q8_0.gguf")
        generate_launch_script(
            mode="tiny",
            model_path=str(self.model_dir / "Tiny-1B-Q8_0.gguf"),
            params={"ctx_size": 4096},
            project_root=self.project_root,
        )
        scripts = list_launch_scripts()
        self.assertEqual(len(scripts), 1)
        self.assertEqual(scripts[0]["mode"], "tiny")

        removed = delete_launch_script("tiny")
        self.assertTrue(removed)
        self.assertEqual(list_launch_scripts(), [])

    def test_summary_reflects_last_scan(self) -> None:
        self._seed_model("Tiny-1B-Q8_0.gguf")
        generate_all_launch_scripts(project_root=self.project_root, model_dirs=[self.model_dir])
        summary = launch_scripts_scan_summary()
        self.assertIsNotNone(summary["last_scan"])
        self.assertEqual(summary["last_scan_summary"]["scanned_model_count"], 1)
        self.assertEqual(len(summary["scripts"]), 1)


class StartupAutoScanTests(_IsolatedDirs):
    def test_startup_autoscan_disabled_when_config_says_so(self) -> None:
        config = AppConfig(auto_scan_on_startup=False, auto_generate_launch_scripts=True)
        self.assertIsNone(startup_autoscan_if_enabled(config))

    def test_startup_autoscan_disabled_when_generation_off(self) -> None:
        config = AppConfig(auto_scan_on_startup=True, auto_generate_launch_scripts=False)
        self.assertIsNone(startup_autoscan_if_enabled(config))

    def test_startup_autoscan_runs_when_both_enabled(self) -> None:
        self._seed_model("Tiny-1B-Q8_0.gguf")
        config = AppConfig(
            model_dirs=[str(self.model_dir)],
            auto_scan_on_startup=True,
            auto_generate_launch_scripts=True,
        )
        result = startup_autoscan_if_enabled(config)
        self.assertIsNotNone(result)
        self.assertEqual(result.scanned_model_count, 1)
        self.assertEqual(len(result.generated), 1)


VLLM_PARAMS = {"runtime": "vllm.cpp", "ctx_size": 8192, "max_num_seqs": 4, "enable_prefix_caching": True}


class VllmCppLaunchScriptTests(_IsolatedDirs):
    def test_vllm_cpp_profile_uses_vllm_server(self) -> None:
        model_path = self._seed_model("Qwen3-8B-Q4_K_M.gguf")
        payload = generate_launch_script(
            mode="qwen-vllm",
            model_path=str(model_path),
            params=dict(VLLM_PARAMS),
            project_root=self.project_root,
        )
        ps1 = Path(payload["ps1_path"]).read_text(encoding="utf-8")
        self.assertIn(f"& '{self.vllm_bin.as_posix()}' --model $model", ps1)
        self.assertNotIn("llama-server", ps1)
        self.assertNotIn(" -m ", ps1)
        self.assertEqual(ps1.count("--model"), 1)
        self.assertIn("--max-model-len 8192", ps1)
        self.assertIn("--num-blocks 256", ps1)
        self.assertIn("--max-num-seqs 4", ps1)
        self.assertIn("--enable-prefix-caching", ps1)
        self.assertIn("--served-model-name", ps1)
        self.assertNotIn("--ctx-size", ps1)
        self.assertIn("vllm-server", payload["command_line"])
        self.assertEqual(_parse_model_path(Path(payload["ps1_path"])), model_path.as_posix())
        if not paths_module.is_windows():
            sh = Path(payload["sh_path"]).read_text(encoding="utf-8")
            self.assertIn(f'exec {self.vllm_bin} --model "$model"', sh)
            self.assertIn("--max-model-len 8192", sh)
            self.assertNotIn("llama-server", sh)

    def test_vllm_cpp_profile_appends_extra_vllm_cpp_args_only(self) -> None:
        model_path = self._seed_model("Qwen3-8B-Q4_K_M.gguf")
        config = AppConfig(extra_llama_args=["--llama-only"], extra_vllm_cpp_args=["--vllm-extra", "1"])
        payload = generate_launch_script(
            mode="qwen-vllm",
            model_path=str(model_path),
            params=dict(VLLM_PARAMS),
            project_root=self.project_root,
            config=config,
        )
        ps1 = Path(payload["ps1_path"]).read_text(encoding="utf-8")
        self.assertIn("--vllm-extra 1", ps1)
        self.assertNotIn("--llama-only", ps1)

    def test_vllm_cpp_missing_binary_uses_placeholder_with_warning(self) -> None:
        model_path = self._seed_model("Qwen3-8B-Q4_K_M.gguf")
        missing = mock.Mock(binary_path=None)
        with mock.patch("inferencedeck.backends.detect_vllm_cpp", return_value=missing):
            payload = generate_launch_script(
                mode="qwen-vllm",
                model_path=str(model_path),
                params=dict(VLLM_PARAMS),
                project_root=self.project_root,
            )
        ps1 = Path(payload["ps1_path"]).read_text(encoding="utf-8")
        self.assertIn("& 'vllm-server' --model $model", ps1)
        self.assertTrue(any("vllm-server was not found" in w for w in payload["warnings"]))
        self.assertFalse(any("llama-server was not found" in w for w in payload["warnings"]))

    def test_vllm_cpp_speculative_config_json_survives_rendering(self) -> None:
        model_path = self._seed_model("Qwen3-8B-Q4_K_M.gguf")
        params = dict(VLLM_PARAMS, speculative_config={"method": "mtp", "num_speculative_tokens": 2})
        payload = generate_launch_script(
            mode="qwen-vllm",
            model_path=str(model_path),
            params=params,
            project_root=self.project_root,
        )
        spec = '{"method": "mtp", "num_speculative_tokens": 2}'
        ps1 = Path(payload["ps1_path"]).read_text(encoding="utf-8")
        self.assertIn(f"--speculative-config '{spec}'", ps1)
        if not paths_module.is_windows():
            sh = Path(payload["sh_path"]).read_text(encoding="utf-8")
            self.assertIn(f"--speculative-config '{spec}'", sh)

    def test_scan_generates_vllm_cpp_script_for_vllm_profile(self) -> None:
        manifest = {
            "models": [
                {"mode": "tiny", "name": "Tiny", "description": "test",
                 "recommended_params": {"ctx_size": 4096, "threads": 4, "gpu_layers": 999,
                                        "cache_type_k": "q8_0", "cache_type_v": "q8_0"}},
                {"mode": "qwen-vllm", "name": "Qwen (vllm.cpp)", "description": "test",
                 "recommended_params": dict(VLLM_PARAMS)},
            ]
        }
        (self.project_root / "models.json").write_text(json.dumps(manifest), encoding="utf-8")
        self._seed_model("Tiny-1B-Q8_0.gguf")
        self._seed_model("Qwen-8B-Q4_K_M.gguf")
        result = generate_all_launch_scripts(project_root=self.project_root, model_dirs=[self.model_dir])
        by_mode = {item.mode: item for item in result.generated}
        vllm_ps1 = Path(by_mode["qwen-vllm"].ps1_path).read_text(encoding="utf-8")
        self.assertIn(f"& '{self.vllm_bin.as_posix()}' --model $model", vllm_ps1)
        self.assertIn("--max-model-len 8192", vllm_ps1)
        tiny_ps1 = Path(by_mode["tiny"].ps1_path).read_text(encoding="utf-8")
        self.assertIn(f"& '{self.server_bin.as_posix()}' -m $model", tiny_ps1)

    def test_manifest_params_keep_vllm_cpp_keys(self) -> None:
        params = dict(VLLM_PARAMS, block_size=32, kv_cache_dtype="fp8", threads=4)
        kept = launch_scripts_module._manifest_params(params)
        for key in ("runtime", "ctx_size", "max_num_seqs", "enable_prefix_caching", "block_size", "kv_cache_dtype"):
            self.assertIn(key, kept)
        self.assertEqual(kept["runtime"], "vllm.cpp")


class ConfigFieldTests(_IsolatedDirs):
    def test_app_config_round_trips_new_fields(self) -> None:
        config = AppConfig(
            auto_generate_launch_scripts=False,
            auto_scan_on_startup=False,
        )
        path = self.config_dir / "config.json"
        config.save(path)
        loaded = AppConfig.load(path)
        self.assertFalse(loaded.auto_generate_launch_scripts)
        self.assertFalse(loaded.auto_scan_on_startup)


if __name__ == "__main__":
    unittest.main()
