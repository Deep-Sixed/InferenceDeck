from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from inferencedeck import llama_flags
from inferencedeck.llama_args import build_llama_server_args

# Trimmed from real `llama-server --help` output of each generation.
OLD_HELP = """
-c,    --ctx-size N                     size of the prompt context
--mlock                                 force system to keep model in RAM
--mmap, --no-mmap                       whether to memory-map model
-md,   --model-draft FNAME              draft model for speculative decoding
--draft-max, --draft, --draft-n N       number of tokens to draft
--draft-min, --draft-n-min N            minimum number of draft tokens
--draft-p-min P                         minimum speculative decoding probability
--port PORT                             port to listen (default: 8080)
"""
NEW_HELP = """
-c,    --ctx-size N                     size of the prompt context
-lm,   --load-mode MODE                 model loading mode (default: auto)
--spec-draft-model, -md, --model-draft FNAME
--spec-draft-n-max N                    number of tokens to draft
--spec-draft-n-min N                    minimum number of draft tokens
--spec-draft-p-min, --draft-p-min P     minimum speculative decoding probability
--port PORT                             port to listen (default: 8080)
"""
# Flags every launch emits, present in both generations.
COMMON_HELP = """
--host HOST                             ip address to listen
-a,    --alias STRING                   model name alias
-fa,   --flash-attn [on|off|auto]       set Flash Attention use
--reasoning [on|off|auto]               use reasoning/thinking in the chat
-kvo,  --kv-offload, -nkvo, --no-kv-offload
--op-offload, --no-op-offload
--spec-type TYPE                        speculative decoding type
"""
OLD_HELP += COMMON_HELP
NEW_HELP += COMMON_HELP
OLD = llama_flags.parse_help_flags(OLD_HELP)
NEW = llama_flags.parse_help_flags(NEW_HELP)


def build(params, flags):
    return build_llama_server_args("llama-server", "m.gguf", params, flags=flags)


class LoadModeTests(unittest.TestCase):
    def test_parse_help_reads_long_flags_only(self) -> None:
        self.assertIn("--load-mode", NEW)
        self.assertIn("--no-mmap", OLD)
        self.assertNotIn("-lm", NEW)
        self.assertNotIn("--mmap", NEW)

    def test_default_emits_no_loading_flag_on_any_build(self) -> None:
        # Both generations default to mmap, so saying nothing works everywhere.
        for flags in (OLD, NEW, None):
            argv = build({}, flags).argv
            for flag in ("--mmap", "--no-mmap", "--mlock", "--load-mode"):
                self.assertNotIn(flag, argv)

    def test_mmap_and_mlock_booleans_per_build(self) -> None:
        cases = [
            ({"mmap": False}, ["--load-mode", "none"], ["--no-mmap"]),
            ({"mlock": True}, ["--load-mode", "mmap+mlock"], ["--mlock"]),
            ({"mmap": False, "mlock": True}, ["--load-mode", "mlock"], ["--no-mmap", "--mlock"]),
        ]
        for params, new_args, old_args in cases:
            with self.subTest(params=params):
                new = build(params, NEW)
                self.assertEqual(new.argv[-len(new_args):], new_args)
                self.assertEqual(new.warnings, [])
                old = build(params, OLD)
                self.assertEqual(old.argv[-len(old_args):], old_args)
                self.assertNotIn("--load-mode", old.argv)
                # Unknown build: the older spelling, as before.
                self.assertEqual(build(params, None).argv[-len(old_args):], old_args)

    def test_explicit_load_mode(self) -> None:
        self.assertEqual(build({"load_mode": "dio"}, NEW).argv[-2:], ["--load-mode", "dio"])
        self.assertEqual(build({"load_mode": "mlock", "mmap": True}, OLD).argv[-2:], ["--no-mmap", "--mlock"])
        dio_old = build({"load_mode": "dio"}, OLD)
        self.assertNotIn("--load-mode", dio_old.argv)
        self.assertTrue(any("dio" in w for w in dio_old.warnings))
        bad = build({"load_mode": "turbo"}, NEW)
        self.assertNotIn("--load-mode", bad.argv)
        self.assertTrue(bad.warnings)


class DraftFlagTests(unittest.TestCase):
    PARAMS = {"draft_model": "d.gguf", "draft_max": 8, "draft_min": 2, "draft_p_min": 0.6}

    def test_legacy_draft_keys_are_renamed_on_new_builds(self) -> None:
        cmd = build(self.PARAMS, NEW)
        argv = cmd.argv
        self.assertEqual(argv[argv.index("--spec-draft-n-max") + 1], "8")
        self.assertEqual(argv[argv.index("--spec-draft-n-min") + 1], "2")
        self.assertEqual(argv[argv.index("--spec-draft-p-min") + 1], "0.6")
        for removed in ("--draft-max", "--draft-min"):
            self.assertNotIn(removed, argv)
        self.assertEqual(cmd.warnings, [])

    def test_old_and_unknown_builds_keep_old_names(self) -> None:
        for flags in (OLD, None):
            argv = build(self.PARAMS, flags).argv
            self.assertIn("--draft-max", argv)
            self.assertIn("--draft-min", argv)
            self.assertIn("--draft-p-min", argv)


class UnsupportedFlagWarningTests(unittest.TestCase):
    def test_warns_about_flags_the_build_does_not_list(self) -> None:
        cmd = build({"spec_draft_n_max": 4, "draft_model": "d.gguf"}, OLD)
        self.assertTrue(any("--spec-draft-n-max" in w for w in cmd.warnings))

    def test_extra_args_are_not_second_guessed(self) -> None:
        cmd = build_llama_server_args("llama-server", "m.gguf", {}, extra_args=["--made-up"], flags=NEW)
        self.assertIn("--made-up", cmd.argv)
        self.assertFalse(any("--made-up" in w for w in cmd.warnings))


class SupportedFlagsCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.binary = self.tmp / "llama-server"
        self.binary.write_bytes(b"binary")
        patcher = mock.patch.object(llama_flags, "_cache_file", return_value=self.tmp / "flags.json")
        patcher.start()
        self.addCleanup(patcher.stop)
        llama_flags._memory.clear()
        self.addCleanup(llama_flags._memory.clear)

    def _help(self, text: str, code: int = 0):
        return subprocess.CompletedProcess([], code, stdout=text, stderr="")

    def test_probes_once_then_uses_disk_cache(self) -> None:
        with mock.patch.object(llama_flags, "run_hidden", return_value=self._help(NEW_HELP)) as run:
            self.assertIn("--load-mode", llama_flags.supported_flags(str(self.binary)))
            llama_flags._memory.clear()
            self.assertIn("--load-mode", llama_flags.supported_flags(str(self.binary)))
        run.assert_called_once()
        self.assertEqual(run.call_args.args[0], [str(self.binary), "--help"])

    def test_rebuilt_binary_is_probed_again(self) -> None:
        with mock.patch.object(llama_flags, "run_hidden", return_value=self._help(OLD_HELP)):
            self.assertIn("--no-mmap", llama_flags.supported_flags(str(self.binary)))
        self.binary.write_bytes(b"a newer, larger binary")
        stat = self.binary.stat()
        os.utime(self.binary, (stat.st_atime, stat.st_mtime + 10))
        with mock.patch.object(llama_flags, "run_hidden", return_value=self._help(NEW_HELP)):
            self.assertIn("--load-mode", llama_flags.supported_flags(str(self.binary)))

    def test_unusable_help_is_unknown_and_not_persisted(self) -> None:
        with mock.patch.object(llama_flags, "run_hidden", return_value=self._help("Segmentation fault", 139)):
            self.assertIsNone(llama_flags.supported_flags(str(self.binary)))
        self.assertFalse((self.tmp / "flags.json").exists())
        with mock.patch.object(llama_flags, "run_hidden", side_effect=OSError("exec format error")):
            llama_flags._memory.clear()
            self.assertIsNone(llama_flags.supported_flags(str(self.binary)))

    def test_missing_binary_is_unknown_without_running_anything(self) -> None:
        with mock.patch.object(llama_flags, "run_hidden") as run:
            self.assertIsNone(llama_flags.supported_flags(str(self.tmp / "nope")))
            self.assertIsNone(llama_flags.supported_flags(None))
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
