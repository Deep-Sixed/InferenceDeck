from __future__ import annotations

import json
import subprocess
import threading
import unittest
import urllib.request
from http.server import ThreadingHTTPServer
from unittest import mock

from inferencedeck import hf_download
from inferencedeck.control import ControlPlane
from inferencedeck.control_api import ControlRequestHandler

GIB = 1024**3

# Shaped like the HF tree API for a typical community GGUF repo: single-file
# quants, a two-shard quant in a subfolder, and two vision projectors.
REPO_FILES = [
    {"path": "README.md", "size": 2000},
    {"path": "gemma-3-4b-it-Q4_K_M.gguf", "size": 2 * GIB},
    {"path": "gemma-3-4b-it-UD-Q4_K_XL.gguf", "size": 2 * GIB},
    {"path": "gemma-3-4b-it-Q8_0.gguf", "size": 4 * GIB},
    {"path": "BF16/gemma-3-4b-it-BF16-00001-of-00002.gguf", "size": 5 * GIB},
    {"path": "BF16/gemma-3-4b-it-BF16-00002-of-00002.gguf", "size": 3 * GIB},
    {"path": "mmproj-BF16.gguf", "size": 800_000_000},
    {"path": "mmproj-F16.gguf", "size": 800_000_000},
]


class ResolveDownloadTests(unittest.TestCase):
    def resolve(self, **kwargs):
        return hf_download.resolve_download("unsloth/gemma-3-4b-it-GGUF", files=REPO_FILES, **kwargs)

    def test_groups_shards_and_separates_projectors(self) -> None:
        grouped = hf_download.gguf_variants(REPO_FILES)
        names = [v["name"] for v in grouped["variants"]]
        self.assertIn("BF16/gemma-3-4b-it-BF16", names)
        self.assertEqual(len(names), 4)  # mmproj files are not model variants
        bf16 = next(v for v in grouped["variants"] if v["quant"] == "BF16")
        self.assertEqual(len(bf16["files"]), 2)
        self.assertEqual(bf16["size_bytes"], 8 * GIB)
        self.assertEqual([m["path"] for m in grouped["mmproj"]], ["mmproj-BF16.gguf", "mmproj-F16.gguf"])

    def test_quant_is_case_insensitive_and_adds_preferred_mmproj(self) -> None:
        plan = self.resolve(quant="q4_k_m")
        self.assertTrue(plan["success"], plan)
        self.assertEqual(plan["files"], ["gemma-3-4b-it-Q4_K_M.gguf", "mmproj-F16.gguf"])
        self.assertEqual(plan["total_bytes"], 2 * GIB + 800_000_000)

    def test_matching_one_shard_pulls_every_shard(self) -> None:
        plan = self.resolve(pattern="*BF16-00001-of-00002.gguf", include_mmproj=False)
        self.assertTrue(plan["success"], plan)
        self.assertEqual(plan["files"], [
            "BF16/gemma-3-4b-it-BF16-00001-of-00002.gguf",
            "BF16/gemma-3-4b-it-BF16-00002-of-00002.gguf",
        ])
        self.assertIsNone(plan["mmproj"])

    def test_ambiguous_match_lists_candidates(self) -> None:
        plan = self.resolve(pattern="*Q4_K*")
        self.assertFalse(plan["success"])
        self.assertEqual(len(plan["matches"]), 2)
        self.assertIn("narrow it", plan["error"])

    def test_no_match_lists_available_variants(self) -> None:
        plan = self.resolve(quant="Q2_K")
        self.assertFalse(plan["success"])
        self.assertEqual(len(plan["available"]), 4)

    def test_missing_shard_is_flagged(self) -> None:
        files = [f for f in REPO_FILES if "00002-of" not in f["path"]]
        plan = hf_download.resolve_download("o/r", quant="BF16", files=files, include_mmproj=False)
        self.assertTrue(plan["success"])
        self.assertTrue(plan["warnings"])

    def test_rejects_malformed_repo_ids(self) -> None:
        for bad in ("", "noslash", "../etc/passwd", "a/b/c", "-x/y"):
            self.assertFalse(hf_download.resolve_download(bad, files=REPO_FILES)["success"], bad)


class ListRepoFilesTests(unittest.TestCase):
    def test_follows_link_pagination(self) -> None:
        pages = {
            "first": ([{"type": "file", "path": "a.gguf", "lfs": {"size": 10}}, {"type": "directory", "path": "BF16"}], "second"),
            "second": ([{"type": "file", "path": "BF16/b.gguf", "size": 5}], None),
        }
        calls: list[str] = []

        def fake_page(url: str):
            key = "second" if url == "second" else "first"
            calls.append(key)
            return pages[key]

        with mock.patch.object(hf_download, "_get_page", side_effect=fake_page):
            files = hf_download.list_repo_files("o/r")
        self.assertEqual(calls, ["first", "second"])
        self.assertEqual(files, [{"path": "a.gguf", "size": 10}, {"path": "BF16/b.gguf", "size": 5}])


class DownloadModelTests(unittest.TestCase):
    def setUp(self) -> None:
        patcher = mock.patch.object(hf_download, "list_repo_files", return_value=REPO_FILES)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_runs_cli_with_exact_files(self) -> None:
        done = subprocess.CompletedProcess([], 0, stdout="/cache/snapshots/abc\n", stderr="")
        with mock.patch.object(hf_download, "find_hf_cli", return_value="hf"), \
                mock.patch.object(hf_download, "run_hidden", return_value=done) as run:
            result = hf_download.download_model("unsloth/gemma-3-4b-it-GGUF", quant="Q8_0")
        self.assertTrue(result["success"], result)
        argv = run.call_args.args[0]
        self.assertEqual(argv[:3], ["hf", "download", "unsloth/gemma-3-4b-it-GGUF"])
        self.assertIn("gemma-3-4b-it-Q8_0.gguf", argv)
        self.assertIn("mmproj-F16.gguf", argv)
        self.assertNotIn("--local-dir", argv)
        self.assertEqual(result["location"], "/cache/snapshots/abc")

    def test_dry_run_does_not_download(self) -> None:
        with mock.patch.object(hf_download, "run_hidden") as run:
            result = hf_download.download_model("unsloth/gemma-3-4b-it-GGUF", quant="Q8_0", dry_run=True)
        self.assertTrue(result["success"])
        self.assertTrue(result["dry_run"])
        run.assert_not_called()

    def test_missing_cli_is_reported(self) -> None:
        with mock.patch.object(hf_download, "find_hf_cli", return_value=None):
            result = hf_download.download_model("unsloth/gemma-3-4b-it-GGUF", quant="Q8_0")
        self.assertFalse(result["success"])
        self.assertIn("pip install huggingface_hub", result["error"])

    def test_cli_failure_surfaces_last_error_line(self) -> None:
        failed = subprocess.CompletedProcess([], 1, stdout="", stderr="progress\n401 Unauthorized: gated repo\n")
        with mock.patch.object(hf_download, "find_hf_cli", return_value="hf"), \
                mock.patch.object(hf_download, "run_hidden", return_value=failed):
            result = hf_download.download_model("unsloth/gemma-3-4b-it-GGUF", quant="Q8_0", dest_dir="/models")
        self.assertFalse(result["success"])
        self.assertEqual(result["error"], "401 Unauthorized: gated repo")

    def test_draft_pull_uses_resolver_without_projector(self) -> None:
        from inferencedeck import draft_models

        with mock.patch.object(draft_models, "download_model", return_value={"success": True, "message": "ok"}) as call:
            self.assertTrue(draft_models.pull_draft_model("Qwen/Qwen2.5-1.5B-Instruct-GGUF")["success"])
        call.assert_called_once_with("Qwen/Qwen2.5-1.5B-Instruct-GGUF", quant="Q4_K_M", include_mmproj=False)


class HfDownloadApiTests(unittest.TestCase):
    def setUp(self) -> None:
        handler = type("TestHandler", (ControlRequestHandler,), {})
        handler.control_plane = mock.Mock(spec=ControlPlane)
        handler.control_plane.hf_files.return_value = {"success": True, "variants": []}
        handler.control_plane.hf_download.return_value = {"success": True, "files": ["m.gguf"]}
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"
        self.control = handler.control_plane

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def test_files_endpoint(self) -> None:
        with urllib.request.urlopen(self.base + "/api/hf/files?repo_id=o/r", timeout=2) as response:
            self.assertTrue(json.load(response)["success"])
        self.control.hf_files.assert_called_once_with("o/r")

    def test_download_endpoint_never_takes_a_destination(self) -> None:
        body = {"repo_id": "o/r", "quant": "Q4_K_M", "dest_dir": "/etc", "dry_run": True}
        req = urllib.request.Request(
            self.base + "/api/hf/download",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=2) as response:
            self.assertTrue(json.load(response)["success"])
        self.control.hf_download.assert_called_once_with(
            "o/r", quant="Q4_K_M", pattern=None, include_mmproj=True, dry_run=True
        )


if __name__ == "__main__":
    unittest.main()
