from __future__ import annotations

import subprocess
import unittest
from unittest import mock

from inferencedeck import hardware
from inferencedeck.estimates import estimate_memory_fit

GIB = 1024**3


def smi(index: int, name: str, bus: str | None, vram_gb: int = 24) -> dict:
    return {
        "index": index, "name": name, "vendor": "NVIDIA", "backend": "nvidia-smi",
        "pci_bus_id": hardware.normalize_pci_bus_id(bus), "integrated": False,
        "acceleration_backend": "cuda", "vram_total_bytes": vram_gb * GIB, "vram_free_bytes": vram_gb * GIB,
    }


def os_row(name: str, backend: str, bus: str | None = None, vendor: str = "NVIDIA") -> dict:
    return {
        "index": 0, "name": name, "vendor": vendor, "backend": backend,
        "pci_bus_id": hardware.normalize_pci_bus_id(bus), "integrated": False,
        "acceleration_backend": "cuda" if vendor == "NVIDIA" else "vulkan", "vram_total_bytes": None,
    }


class DedupeGpusTests(unittest.TestCase):
    def names(self, gpus):
        return [(g["index"], g["name"], g["backend"]) for g in hardware._dedupe_gpus(gpus)]

    def test_identical_cards_from_one_source_are_kept(self) -> None:
        gpus = [smi(0, "NVIDIA GeForce RTX 3090", "00000000:01:00.0"), smi(1, "NVIDIA GeForce RTX 3090", "00000000:02:00.0")]
        self.assertEqual(len(self.names(gpus)), 2)

    def test_lspci_rows_match_nvidia_smi_by_bus_address(self) -> None:
        gpus = [
            smi(0, "NVIDIA GeForce RTX 3090", "00000000:01:00.0"),
            smi(1, "NVIDIA GeForce RTX 3090", "00000000:02:00.0"),
            os_row("NVIDIA Corporation GA102 [GeForce RTX 3090]", "lspci", "01:00.0"),
            os_row("NVIDIA Corporation GA102 [GeForce RTX 3090]", "lspci", "02:00.0"),
            os_row("Intel Corporation Raptor Lake-S GT1 [UHD Graphics 770]", "lspci", "00:02.0", vendor="Intel"),
        ]
        self.assertEqual(self.names(gpus), [
            (0, "NVIDIA GeForce RTX 3090", "nvidia-smi"),
            (1, "NVIDIA GeForce RTX 3090", "nvidia-smi"),
            (2, "Intel Corporation Raptor Lake-S GT1 [UHD Graphics 770]", "lspci"),
        ])

    def test_card_the_driver_does_not_see_is_kept(self) -> None:
        # Same model name, different bus address: a real third card, not a duplicate.
        gpus = [
            smi(0, "NVIDIA GeForce RTX 3090", "00000000:01:00.0"),
            os_row("NVIDIA Corporation GA102 [GeForce RTX 3090]", "lspci", "01:00.0"),
            os_row("NVIDIA Corporation GA102 [GeForce RTX 3090]", "lspci", "03:00.0"),
        ]
        self.assertEqual(len(self.names(gpus)), 2)

    def test_windows_rows_match_by_name_one_for_one(self) -> None:
        gpus = [
            smi(0, "NVIDIA GeForce RTX 3090", "00000000:01:00.0"),
            smi(1, "NVIDIA GeForce RTX 3090", "00000000:02:00.0"),
            os_row("NVIDIA GeForce RTX 3090", "windows-cim"),
            os_row("NVIDIA GeForce RTX 3090", "windows-cim"),
            os_row("NVIDIA GeForce GT 710", "windows-cim"),
        ]
        self.assertEqual([n for _, n, _ in self.names(gpus)], [
            "NVIDIA GeForce RTX 3090", "NVIDIA GeForce RTX 3090", "NVIDIA GeForce GT 710",
        ])

    def test_similar_names_are_not_merged(self) -> None:
        gpus = [smi(0, "NVIDIA GeForce RTX 3090", None), os_row("NVIDIA GeForce RTX 3090 Ti", "windows-cim")]
        self.assertEqual(len(self.names(gpus)), 2)

    def test_detect_gpus_end_to_end_on_linux(self) -> None:
        smi_out = (
            "0, NVIDIA GeForce RTX 3090, 24576, 24000, 550.54, 405, 9751, 00000000:01:00.0\n"
            "1, NVIDIA GeForce RTX 3090, 24576, 23000, 550.54, 405, 9751, 00000000:02:00.0\n"
        )
        lspci_out = (
            "00:02.0 VGA compatible controller: Intel Corporation Raptor Lake-S GT1 [UHD Graphics 770] (rev 04)\n"
            "01:00.0 VGA compatible controller: NVIDIA Corporation GA102 [GeForce RTX 3090] (rev a1)\n"
            "02:00.0 VGA compatible controller: NVIDIA Corporation GA102 [GeForce RTX 3090] (rev a1)\n"
        )

        def fake_run(args, timeout=None):
            out = smi_out if "nvidia-smi" in args[0] else lspci_out
            return subprocess.CompletedProcess(args, 0, stdout=out, stderr="")

        with mock.patch.object(hardware.shutil, "which", side_effect=lambda name: f"/usr/bin/{name}"), \
                mock.patch.object(hardware, "_run", side_effect=fake_run), \
                mock.patch.object(hardware, "is_windows", return_value=False), \
                mock.patch.object(hardware.platform, "system", return_value="Linux"), \
                mock.patch.object(hardware, "_linux_lspci_bus_width", return_value=None):
            gpus = hardware.detect_gpus()
        self.assertEqual([g["name"] for g in gpus], [
            "NVIDIA GeForce RTX 3090", "NVIDIA GeForce RTX 3090", "Intel Corporation Raptor Lake-S GT1 [UHD Graphics 770]",
        ])
        self.assertEqual([g["pci_bus_id"] for g in gpus], ["0000:01:00.0", "0000:02:00.0", "0000:00:02.0"])
        self.assertEqual(gpus[1]["vram_free_bytes"], 23000 * 1024 * 1024)


class MultiGpuFitTests(unittest.TestCase):
    MODEL = {"name": "big-70B", "params_b": 70, "quant": "Q4_K_M", "size_bytes": 32 * GIB}

    def hw(self, *vram_gb: int) -> dict:
        gpus = [smi(i, "NVIDIA GeForce RTX 3090", None, v) for i, v in enumerate(vram_gb)]
        return {"gpus": gpus, "primary_gpu": gpus[0],
                "memory": {"total_bytes": 128 * GIB, "available_bytes": 100 * GIB}}

    def fit(self, hw, **params):
        return estimate_memory_fit({"ctx_size": 4096, "gpu_layers": "all", **params}, self.MODEL, hw)

    def test_default_split_sums_every_gpu(self) -> None:
        one = self.fit(self.hw(24))
        two = self.fit(self.hw(24, 24))
        self.assertEqual(one["status"], "near_limit")
        self.assertNotEqual(two["status"], "near_limit")
        self.assertEqual(two["estimated"]["accelerator_capacity_mib"], 48 * 1024)
        self.assertEqual(two["accelerator_name"], "2x NVIDIA GeForce RTX 3090")
        self.assertEqual(two["accelerator_count"], 2)
        # Per-GPU runtime overhead is charged once per card.
        self.assertGreater(two["estimated"]["accelerator_used_mib"], one["estimated"]["accelerator_used_mib"])

    def test_split_mode_none_and_device_limit_the_pool(self) -> None:
        self.assertEqual(self.fit(self.hw(24, 24), split_mode="none")["estimated"]["accelerator_capacity_mib"], 24 * 1024)
        self.assertEqual(self.fit(self.hw(24, 24, 24), device="CUDA0,CUDA2")["estimated"]["accelerator_capacity_mib"], 48 * 1024)

    def test_tensor_split_is_bounded_by_the_gpu_that_fills_first(self) -> None:
        # 3:1 over 24+24 GiB: the first card holds 75% and fills at 32 GiB total.
        fit = self.fit(self.hw(24, 24), tensor_split=[3, 1])
        self.assertEqual(fit["estimated"]["accelerator_capacity_mib"], 32 * 1024)
        self.assertEqual(self.fit(self.hw(24, 24), tensor_split="1,0")["accelerator_count"], 1)

    def test_single_gpu_result_is_unchanged(self) -> None:
        hw = self.hw(24)
        without_list = {k: v for k, v in hw.items() if k != "gpus"}
        self.assertEqual(self.fit(hw)["estimated"], self.fit(without_list)["estimated"])


if __name__ == "__main__":
    unittest.main()
