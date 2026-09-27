from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from inferencedeck import backends, fit
from inferencedeck.config import AppConfig
from inferencedeck.paths import executable_names
from inferencedeck.profile_resolver import resolve_profiles
from inferencedeck.server_manager import prepare_launch_command
from inferencedeck.vllm_cpp_args import build_vllm_cpp_server_args

OFFLINE = mock.patch.object(backends, "_request_json", return_value=(False, None, "offline"))


def _flag(argv: list[str], flag: str) -> str | None:
    return argv[argv.index(flag) + 1] if flag in argv else None


class VllmCppArgsTests(unittest.TestCase):
    def test_translates_profile_to_vllm_server_flags(self) -> None:
        command = build_vllm_cpp_server_args(
            "/opt/vllm.cpp/bin/vllm-server",
            "/models/Qwen3-8B-Q4_K_M.gguf",
            {
                "host": "127.0.0.1",
                "port": 8081,
                "alias": "qwen",
                "ctx_size": 32768,
                "max_num_seqs": 8,
                "kv_cache_dtype": "fp8",
                "reasoning": False,
                "enable_prefix_caching": True,
                "speculative_config": {"method": "ngram", "num_speculative_tokens": 4},
            },
            extra_args=["--verbose"],
        )
        argv = command.argv
        self.assertEqual(argv[:3], ["/opt/vllm.cpp/bin/vllm-server", "--model", "/models/Qwen3-8B-Q4_K_M.gguf"])
        self.assertEqual(_flag(argv, "--port"), "8081")
        self.assertEqual(_flag(argv, "--served-model-name"), "qwen")
        self.assertEqual(_flag(argv, "--max-model-len"), "32768")
        # 32768 tokens / 32-token blocks, so the pool holds one full-length sequence.
        self.assertEqual(_flag(argv, "--num-blocks"), "1024")
        self.assertEqual(_flag(argv, "--max-num-seqs"), "8")
        self.assertEqual(_flag(argv, "--kv-cache-dtype"), "fp8")
        self.assertIn("--no-enable-thinking", argv)
        self.assertIn("--enable-prefix-caching", argv)
        self.assertEqual(json.loads(_flag(argv, "--speculative-config")), {"method": "ngram", "num_speculative_tokens": 4})
        self.assertEqual(argv[-1], "--verbose")
        self.assertEqual(command.cwd, str(Path("/opt/vllm.cpp/bin")))
        # No llama-server flags leak through.
        for llama_flag in ("-m", "--ctx-size", "--gpu-layers", "--flash-attn", "--reasoning"):
            self.assertNotIn(llama_flag, argv)

    def test_explicit_pool_sizing_wins_over_ctx_size(self) -> None:
        by_blocks = build_vllm_cpp_server_args("vllm-server", "m.gguf", {"ctx_size": 8192, "num_blocks": 4096}).argv
        self.assertEqual(_flag(by_blocks, "--num-blocks"), "4096")
        by_memory = build_vllm_cpp_server_args("vllm-server", "m.gguf", {"ctx_size": 8192, "kv_cache_memory_mib": 2048}).argv
        self.assertEqual(_flag(by_memory, "--kv-cache-memory"), str(2048 * 1024 * 1024))
        self.assertNotIn("--num-blocks", by_memory)
        custom_block = build_vllm_cpp_server_args("vllm-server", "m.gguf", {"ctx_size": 1000, "block_size": 16}).argv
        self.assertEqual(_flag(custom_block, "--num-blocks"), "63")

    def test_reasoning_unset_leaves_template_default(self) -> None:
        argv = build_vllm_cpp_server_args("vllm-server", "m.gguf", {"ctx_size": 4096}).argv
        self.assertNotIn("--enable-thinking", argv)
        self.assertNotIn("--no-enable-thinking", argv)

    def test_warns_about_llama_only_and_sampling_settings(self) -> None:
        command = build_vllm_cpp_server_args(
            "vllm-server",
            "m.gguf",
            {"ctx_size": 4096, "gpu_layers": 999, "cache_type_k": "q8_0", "device": "auto", "temperature": 0.7, "block_size": 24},
        )
        text = " ".join(command.warnings)
        self.assertIn("gpu_layers", text)
        self.assertIn("cache_type_k", text)
        self.assertNotIn("device", text)  # "auto" means nothing was chosen
        self.assertIn("temperature", text)
        self.assertIn("multiple of 16", text)


class VllmCppProfileTests(unittest.TestCase):
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
                    "mode": "qwen-vllm",
                    "name": "Qwen3 8B on vllm.cpp",
                    "description": "vllm.cpp profile",
                    "recommended_params": {"runtime": "vllm.cpp", "ctx_size": 16384, "max_num_seqs": 4},
                }
            ]
        }
        (self.root / "models.json").write_text(json.dumps(manifest), encoding="utf-8")
        bin_dir = self.root / "vllm.cpp" / "bin"
        bin_dir.mkdir(parents=True)
        self.binary = bin_dir / executable_names("vllm-server")[0]
        self.binary.write_text("#!/bin/sh\n", encoding="utf-8")
        self.binary.chmod(0o755)

    def test_profile_needs_only_ctx_size_and_gets_no_llama_defaults(self) -> None:
        [profile] = resolve_profiles(project_root=self.root, model_dirs=[self.model_dir])
        self.assertTrue(profile.launchable, profile.missing)
        for key in ("batch_size", "flash_attn", "kv_offload", "device", "reasoning"):
            self.assertNotIn(key, profile.params)

    def test_prepare_builds_vllm_server_command(self) -> None:
        config = AppConfig(vllm_cpp_server_path=str(self.binary), extra_vllm_cpp_args=["--verbose"], extra_llama_args=["--no-webui"])
        with OFFLINE:
            prepared = prepare_launch_command("qwen-vllm", self.root, [self.model_dir], config=config)
        self.assertTrue(prepared["success"], prepared)
        self.assertEqual(prepared["runtime"], "vllm.cpp")
        self.assertEqual(prepared["environment"]["id"], "vllm.cpp")
        argv = prepared["command"]["argv"]
        self.assertEqual(argv[0], str(self.binary))
        self.assertEqual(_flag(argv, "--max-model-len"), "16384")
        self.assertEqual(_flag(argv, "--max-num-seqs"), "4")
        self.assertIn("--verbose", argv)
        self.assertNotIn("--no-webui", argv)
        self.assertEqual(prepared["warnings"], [])

    def test_binary_found_under_runtime_dirs(self) -> None:
        config = AppConfig(runtime_dirs=[str(self.root / "vllm.cpp")])
        with OFFLINE, mock.patch.object(backends.shutil, "which", return_value=None), mock.patch.dict(
            "os.environ", {"VLLM_CPP_SERVER": "", "VLLM_CPP_SERVER_BIN": "", "VLLM_CPP_HOME": ""}
        ):
            env = backends.detect_vllm_cpp(config)
        self.assertEqual(env.binary_path, str(self.binary))
        self.assertTrue(env.available)

    def test_missing_binary_is_reported(self) -> None:
        with OFFLINE, mock.patch.object(backends.shutil, "which", return_value=None), mock.patch.dict(
            "os.environ", {"VLLM_CPP_SERVER": "", "VLLM_CPP_SERVER_BIN": "", "VLLM_CPP_HOME": ""}
        ):
            prepared = prepare_launch_command("qwen-vllm", self.root, [self.model_dir], config=AppConfig())
        self.assertFalse(prepared["success"])
        self.assertIn("vllm-server", prepared["error"])

    def test_fit_refuses_vllm_cpp_profiles(self) -> None:
        prepared = {"success": True, "runtime": "vllm.cpp", "environment": {"details": {}}}
        with mock.patch.object(fit, "prepare_launch_command", return_value=prepared):
            result = fit.run_fit_test("qwen-vllm", self.root, [self.model_dir])
        self.assertFalse(result["success"])
        self.assertIn("llama.cpp", result["error"])


if __name__ == "__main__":
    unittest.main()
