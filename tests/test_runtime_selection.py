from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from inferencedeck import backends, llama_runtimes
from inferencedeck.config import AppConfig
from inferencedeck.cpu_features import CpuFeatures, parse_cpuinfo_flags
from inferencedeck.llama_runtimes import resolve_llama_runtime
from inferencedeck.paths import executable_names

# Explicit tool paths on the test runner would bypass discovery.
CLEAR_TOOL_ENV = {name: "" for name in ("LLAMA_SERVER", "LLAMA_SERVER_BIN", "LLAMA_CLI", "LLAMA_CLI_BIN", "LLAMA_FIT_PARAMS", "LLAMA_FIT_PARAMS_BIN")}

# Thanatos / Lenovo P70 class: Haswell-or-later, AVX2 + FMA + F16C.
AVX2_BOX = CpuFeatures(x86=True, features=frozenset({"sse4_2", "avx", "avx2", "fma", "f16c", "bmi2"}), source="test")
# Friday class: Ivy Bridge-era Xeon, AVX + F16C but no AVX2/FMA.
AVX_ONLY_BOX = CpuFeatures(x86=True, features=frozenset({"sse4_2", "avx", "f16c"}), source="test")
# Sandy Bridge: AVX without F16C, so even the AVX1 build (F16C on) can't run.
SANDY_BRIDGE = CpuFeatures(x86=True, features=frozenset({"sse4_2", "avx"}), source="test")
APPLE_SILICON = CpuFeatures(x86=False, source="test")

FRIDAY_CMAKE = "\n".join(
    [
        "# This is the CMakeCache file.",
        "GGML_NATIVE:BOOL=OFF",
        "GGML_AVX:BOOL=ON",
        "GGML_AVX2:BOOL=OFF",
        "GGML_FMA:BOOL=OFF",
        "GGML_F16C:BOOL=ON",
        "GGML_CUDA:BOOL=ON",
    ]
)


def _binary(directory: Path, name: str = "llama-server") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / executable_names(name)[0]  # llama-server.exe on Windows
    path.write_text("#!/bin/sh\n", encoding="utf-8")
    path.chmod(0o755)
    return path


class RuntimeSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        # Keep llama-server binaries on this machine's PATH out of the results.
        patcher = mock.patch.object(llama_runtimes.shutil, "which", return_value=None)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _standard(self) -> Path:
        return _binary(self.root / "build" / "bin")

    def _cuda_avx1(self) -> Path:
        build = self.root / "build-cuda-avx1"
        build.mkdir(parents=True, exist_ok=True)
        (build / "CMakeCache.txt").write_text(FRIDAY_CMAKE, encoding="utf-8")
        return _binary(build / "bin")

    def _resolve(self, cpu: CpuFeatures, cuda: bool, requested: str = "auto", pinned=None) -> dict:
        return resolve_llama_runtime([self.root], requested, pinned, cpu=cpu, has_cuda=cuda)

    def test_avx2_machine_uses_standard_build(self) -> None:
        self._standard()
        self._cuda_avx1()
        result = self._resolve(AVX2_BOX, cuda=True)
        self.assertEqual(result["selected"]["variant"], "standard")
        self.assertIn("AVX2 supported", result["reason"])
        # The AVX1 build is compatible too, but standard wins.
        self.assertTrue(next(r for r in result["candidates"] if r["variant"] == "cuda-avx1")["compatible"])

    def test_avx_only_machine_falls_back_to_cuda_avx1(self) -> None:
        self._standard()
        cuda_avx1 = self._cuda_avx1()
        result = self._resolve(AVX_ONLY_BOX, cuda=True)
        self.assertEqual(result["selected"]["path"], str(cuda_avx1))
        self.assertEqual(result["selected"]["requires"], ["avx", "f16c"])
        self.assertIn("AVX2 not supported", result["reason"])
        self.assertIn("standard build needs AVX2", result["reason"])

    def test_avx1_cuda_build_needs_a_cuda_gpu(self) -> None:
        self._standard()
        self._cuda_avx1()
        result = self._resolve(AVX_ONLY_BOX, cuda=False)
        self.assertIsNone(result["selected"])
        self.assertIn("No compatible llama-server build", result["reason"])
        self.assertIn("needs an NVIDIA GPU with CUDA", result["reason"])

    def test_cpu_only_avx1_build_is_last_resort(self) -> None:
        self._standard()
        self._cuda_avx1()
        cpu_build = self.root / "build-cpu-avx1"
        cpu_build.mkdir()
        (cpu_build / "CMakeCache.txt").write_text(FRIDAY_CMAKE.replace("GGML_CUDA:BOOL=ON", "GGML_CUDA:BOOL=OFF"), encoding="utf-8")
        cpu_bin = _binary(cpu_build / "bin")
        self.assertEqual(self._resolve(AVX_ONLY_BOX, cuda=False)["selected"]["path"], str(cpu_bin))
        # With a GPU, CUDA AVX1 is preferred over CPU-only AVX1.
        self.assertEqual(self._resolve(AVX_ONLY_BOX, cuda=True)["selected"]["variant"], "cuda-avx1")

    def test_missing_f16c_rejects_avx1_build(self) -> None:
        self._cuda_avx1()
        result = self._resolve(SANDY_BRIDGE, cuda=True)
        self.assertIsNone(result["selected"])
        self.assertIn("needs F16C", result["reason"])

    def test_incompatible_pin_is_never_used(self) -> None:
        self._standard()
        self._cuda_avx1()
        result = self._resolve(AVX_ONLY_BOX, cuda=True, requested="standard")
        self.assertEqual(result["selected"]["variant"], "cuda-avx1")
        self.assertEqual(result["policy"], "auto")
        self.assertTrue(any("can't run here" in w for w in result["warnings"]))

    def test_incompatible_pinned_path_is_never_used(self) -> None:
        standard = self._standard()
        self._cuda_avx1()
        result = self._resolve(AVX_ONLY_BOX, cuda=True, pinned=[str(standard)])
        self.assertEqual(result["selected"]["variant"], "cuda-avx1")

    def test_compatible_pin_wins_over_automatic_choice(self) -> None:
        self._standard()
        self._cuda_avx1()
        result = self._resolve(AVX2_BOX, cuda=True, requested="cuda-avx1")
        self.assertEqual(result["selected"]["variant"], "cuda-avx1")
        self.assertEqual(result["policy"], "pinned")

    def test_unknown_pin_warns_and_selects_automatically(self) -> None:
        self._standard()
        result = self._resolve(AVX2_BOX, cuda=False, requested="nope")
        self.assertEqual(result["selected"]["variant"], "standard")
        self.assertTrue(any("not found" in w for w in result["warnings"]))

    def test_non_x86_skips_instruction_set_checks(self) -> None:
        self._standard()
        result = self._resolve(APPLE_SILICON, cuda=False)
        self.assertEqual(result["selected"]["variant"], "standard")

    def test_unreadable_cpu_features_fail_closed(self) -> None:
        self._standard()
        result = self._resolve(CpuFeatures(x86=True, source="unavailable"), cuda=True)
        self.assertIsNone(result["selected"])

    def test_sidecar_metadata_overrides_default(self) -> None:
        binary = _binary(self.root / "custom" / "bin")
        (binary.parent / "inferencedeck-runtime.json").write_text(
            json.dumps({"variant": "cuda-avx1", "label": "Friday build", "cpu": {"requires": ["avx"]}, "gpu": {"backend": "cuda"}}),
            encoding="utf-8",
        )
        result = resolve_llama_runtime([self.root / "custom"], cpu=AVX_ONLY_BOX, has_cuda=True)
        self.assertEqual(result["selected"]["label"], "Friday build")
        self.assertEqual(result["selected"]["source"], "inferencedeck-runtime.json")

    def test_portable_all_variants_build_runs_on_avx_only(self) -> None:
        build = self.root / "build"
        build.mkdir()
        (build / "CMakeCache.txt").write_text("GGML_BACKEND_DL:BOOL=ON\nGGML_CPU_ALL_VARIANTS:BOOL=ON\nGGML_CUDA:BOOL=ON\n", encoding="utf-8")
        _binary(build / "bin")
        self._cuda_avx1()
        result = self._resolve(AVX_ONLY_BOX, cuda=True)
        self.assertEqual(result["selected"]["variant"], "standard")
        self.assertEqual(result["selected"]["requires"], [])


# avx-llama-manager's build profiles (scripts/build.ps1).
AVX_LLAMA_MANAGER_AVX = "\n".join(
    [
        "GGML_SSE42:BOOL=ON", "GGML_AVX:BOOL=ON", "GGML_AVX2:BOOL=OFF", "GGML_FMA:BOOL=OFF",
        "GGML_F16C:BOOL=OFF", "GGML_BMI2:BOOL=OFF", "GGML_AVX_VNNI:BOOL=OFF", "GGML_AVX512:BOOL=OFF",
        "GGML_CUDA:BOOL=ON",
    ]
)
AVX_LLAMA_MANAGER_AVX2 = "\n".join(
    ["GGML_SSE42:BOOL=ON", "GGML_AVX:BOOL=ON", "GGML_AVX2:BOOL=ON", "GGML_FMA:BOOL=ON", "GGML_F16C:BOOL=ON", "GGML_BMI2:BOOL=ON"]
)


class OtherBuildLayoutTests(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        patcher = mock.patch.object(llama_runtimes.shutil, "which", return_value=None)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _msvc_build(self, cache: str) -> Path:
        # Visual Studio generators: llama.cpp/build/bin/Release/llama-server.exe
        build = self.root / "llama.cpp" / "build"
        build.mkdir(parents=True)
        (build / "CMakeCache.txt").write_text(cache, encoding="utf-8")
        return _binary(build / "bin" / "Release")

    def test_finds_visual_studio_release_build_and_its_cmake_cache(self) -> None:
        binary = self._msvc_build(AVX_LLAMA_MANAGER_AVX2)
        result = resolve_llama_runtime([self.root / "llama.cpp"], cpu=AVX2_BOX, has_cuda=False)
        self.assertEqual(result["selected"]["path"], str(binary))
        self.assertEqual(result["selected"]["source"], "CMakeCache.txt")
        self.assertIn("bmi2", result["selected"]["requires"])

    def test_avx_llama_manager_avx_profile_runs_without_f16c(self) -> None:
        # Unlike Friday's build it compiles F16C off, so Sandy Bridge can run it.
        self._msvc_build(AVX_LLAMA_MANAGER_AVX)
        result = resolve_llama_runtime([self.root / "llama.cpp"], cpu=SANDY_BRIDGE, has_cuda=True)
        self.assertEqual(result["selected"]["variant"], "cuda-avx1")
        self.assertEqual(result["selected"]["requires"], ["avx", "sse4_2"])

    def test_bmi2_build_is_refused_without_bmi2(self) -> None:
        self._msvc_build(AVX_LLAMA_MANAGER_AVX2)
        no_bmi2 = CpuFeatures(x86=True, features=frozenset({"sse4_2", "avx", "avx2", "fma", "f16c"}), source="test")
        result = resolve_llama_runtime([self.root / "llama.cpp"], cpu=no_bmi2, has_cuda=False)
        self.assertIsNone(result["selected"])
        self.assertIn("needs BMI2", result["reason"])

    def test_features_windows_cannot_report_are_refused_with_that_reason(self) -> None:
        self._msvc_build(AVX_LLAMA_MANAGER_AVX2 + "\nGGML_AVX_VNNI:BOOL=ON")
        windows_avx2 = CpuFeatures(
            x86=True,
            features=AVX2_BOX.features,
            source="IsProcessorFeaturePresent",
            unverifiable=frozenset({"avx_vnni"}),
        )
        result = resolve_llama_runtime([self.root / "llama.cpp"], cpu=windows_avx2, has_cuda=False)
        self.assertIsNone(result["selected"])
        self.assertIn("AVX-VNNI, which can't be verified on this OS", result["reason"])


class DetectLlamaCppTests(unittest.TestCase):
    def test_companion_tools_come_from_the_selected_build(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _binary(root / "build" / "bin")
            _binary(root / "build" / "bin", "llama-fit-params")
            build = root / "build-cuda-avx1"
            build.mkdir()
            (build / "CMakeCache.txt").write_text(FRIDAY_CMAKE, encoding="utf-8")
            server = _binary(build / "bin")
            fit = _binary(build / "bin", "llama-fit-params")
            with mock.patch.object(llama_runtimes, "detect_cpu_features", return_value=AVX_ONLY_BOX), \
                 mock.patch.object(llama_runtimes, "cuda_available", return_value=True), \
                 mock.patch.object(llama_runtimes.shutil, "which", return_value=None), \
                 mock.patch.object(backends, "candidate_llama_roots", return_value=[root]), \
                 mock.patch.object(backends, "_request_json", return_value=(False, None, "offline")), \
                 mock.patch.object(backends, "_binary_version", return_value=None), \
                 mock.patch.dict("os.environ", CLEAR_TOOL_ENV):
                env = backends.detect_llama_cpp(root, config=AppConfig())
        self.assertEqual(env.binary_path, str(server))
        self.assertEqual(env.details["llama_fit_params"], str(fit))
        self.assertEqual(env.details["runtime_selection"]["selected"]["variant"], "cuda-avx1")

    def test_no_compatible_build_reports_why(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _binary(root / "build" / "bin")
            with mock.patch.object(llama_runtimes, "detect_cpu_features", return_value=AVX_ONLY_BOX), \
                 mock.patch.object(llama_runtimes, "cuda_available", return_value=True), \
                 mock.patch.object(llama_runtimes.shutil, "which", return_value=None), \
                 mock.patch.object(backends, "candidate_llama_roots", return_value=[root]), \
                 mock.patch.object(backends, "_request_json", return_value=(False, None, "offline")), \
                 mock.patch.object(backends, "_binary_version", return_value=None), \
                 mock.patch.dict("os.environ", CLEAR_TOOL_ENV):
                env = backends.detect_llama_cpp(root, config=AppConfig())
        self.assertIsNone(env.binary_path)
        self.assertTrue(any("needs AVX2" in w for w in env.warnings))


class CpuFlagParsingTests(unittest.TestCase):
    def test_parses_proc_cpuinfo_flags(self) -> None:
        text = "processor\t: 0\nmodel name\t: Intel Xeon E5-2690 v2\nflags\t\t: fpu sse4_1 sse4_2 avx f16c aes\n\nprocessor\t: 1\n"
        self.assertEqual(parse_cpuinfo_flags(text), frozenset({"sse4_2", "avx", "f16c"}))


if __name__ == "__main__":
    unittest.main()


class SetRuntimeTests(unittest.TestCase):
    def _plane(self, candidates):
        from inferencedeck.control import ControlPlane

        plane = ControlPlane()
        plane.runtime = mock.Mock(return_value={"candidates": candidates})  # type: ignore[method-assign]
        return plane

    def test_rejects_incompatible_pin(self) -> None:
        plane = self._plane([{"id": "standard", "label": "Standard llama.cpp", "compatible": False, "incompatible_reason": "needs AVX2, which this CPU lacks"}])
        with self.assertRaisesRegex(ValueError, "needs AVX2"):
            plane.set_runtime("standard")

    def test_rejects_unknown_runtime(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown runtime"):
            self._plane([]).set_runtime("/usr/bin/anything")

    def test_saves_compatible_pin_and_auto(self) -> None:
        plane = self._plane([{"id": "cuda-avx1", "label": "x", "compatible": True, "incompatible_reason": None}])
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict("os.environ", {"LCC_CONFIG_DIR": tmp}):
            plane.set_runtime("cuda-avx1")
            self.assertEqual(AppConfig.load().llama_runtime, "cuda-avx1")
            plane.set_runtime("auto")
            self.assertEqual(AppConfig.load().llama_runtime, "auto")
