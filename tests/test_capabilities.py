from __future__ import annotations

import json
import shutil
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest import mock

from inferencedeck import capabilities, estimates, server_manager
from inferencedeck.api_params import validate_overrides
from inferencedeck.config import AppConfig
from inferencedeck.control import ControlPlane
from inferencedeck.control_api import ControlRequestHandler
from inferencedeck.llama_args import build_llama_server_args
from inferencedeck.profile_resolver import ResolvedProfile
from inferencedeck.schema import Environment

try:
    import gguf
except ImportError:  # pragma: no cover - gguf is a declared dependency
    gguf = None


def _write_gguf(path: Path, arch: str, **fields) -> str:
    writer = gguf.GGUFWriter(str(path), arch)
    if "context_length" in fields:
        writer.add_context_length(fields["context_length"])
    if "pooling" in fields:
        writer.add_pooling_type(fields["pooling"])
    if "template" in fields:
        writer.add_chat_template(fields["template"])
    if "vision" in fields:
        writer.add_clip_has_vision_encoder(fields["vision"])
    if "audio" in fields:
        writer.add_clip_has_audio_encoder(fields["audio"])
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return str(path)


class _MetaCache(unittest.TestCase):
    """Points the GGUF metadata cache at a temp dir and clears the in-memory cache."""

    def setUp(self) -> None:
        self.dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.dir, True)
        patcher = mock.patch.object(estimates, "_meta_cache_file", return_value=self.dir / "meta.json")
        patcher.start()
        self.addCleanup(patcher.stop)
        estimates._gguf_meta_mem.clear()
        self.addCleanup(estimates._gguf_meta_mem.clear)


@unittest.skipIf(gguf is None, "gguf is not installed")
class DiscoveryTests(_MetaCache):
    def test_chat_model_with_tools_and_projector(self) -> None:
        path = _write_gguf(self.dir / "qwen.gguf", "qwen3", context_length=131072, template="{% if tools %}<tool_call>{% endif %}")
        projector = _write_gguf(self.dir / "mmproj-qwen.gguf", "clip", vision=True, audio=True)
        model = {"path": path, "mmproj_path": projector}

        cold = capabilities.profile_capabilities(model, {}, probe=False)
        self.assertIsNone(cold["tools"])  # never opened the GGUF
        self.assertTrue(cold["vision_available"])

        caps = capabilities.profile_capabilities(model, {"vision": True}, probe=True)
        self.assertEqual(caps["input"], ["text", "image", "audio"])
        self.assertIs(caps["tools"], True)
        self.assertEqual(caps["context_max"], 131072)
        self.assertIs(caps["embedding"], False)
        self.assertFalse(caps["vision_available"])
        self.assertEqual(caps["sources"], {"input": "launch", "output": "default", "tools": "gguf",
                                           "reranker": "gguf", "embedding": "gguf", "context_max": "gguf"})

        estimates._gguf_meta_mem.clear()  # a fresh process reads the disk cache without parsing
        with mock.patch.object(estimates, "_parse_gguf_meta", side_effect=AssertionError("parsed")):
            warm = capabilities.profile_capabilities(model, {}, probe=False)
        self.assertEqual((warm["tools"], warm["context_max"], warm["input"]), (True, 131072, ["text"]))

    def test_embedding_and_reranker_models(self) -> None:
        embed = _write_gguf(self.dir / "nomic.gguf", "nomic-bert", context_length=8192)
        qwen_embed = _write_gguf(self.dir / "qwen-embed.gguf", "qwen3", pooling=gguf.PoolingType.LAST)
        rerank = _write_gguf(self.dir / "rerank.gguf", "qwen3", pooling=gguf.PoolingType.RANK)
        for path, embedding, reranker, output in (
            (embed, True, False, ["embedding"]),
            (qwen_embed, True, False, ["embedding"]),
            (rerank, False, True, ["score"]),
        ):
            caps = capabilities.profile_capabilities({"path": path}, {}, probe=True)
            self.assertEqual((caps["embedding"], caps["reranker"], caps["output"]), (embedding, reranker, output), path)
            self.assertTrue(capabilities.matches(caps, "reranker" if reranker else "embedding"))

    def test_vision_only_and_audio_only_projectors(self) -> None:
        path = _write_gguf(self.dir / "m.gguf", "gemma3")
        audio = _write_gguf(self.dir / "mmproj-audio.gguf", "clip", vision=False, audio=True)
        caps = capabilities.profile_capabilities({"path": path}, {"mmproj": audio}, probe=True)
        self.assertEqual(caps["input"], ["text", "audio"])


class ProjectorAndDeclaredTests(_MetaCache):
    def test_resolve_projector(self) -> None:
        model = {"path": "/m.gguf", "mmproj_path": "/mmproj.gguf"}
        self.assertEqual(capabilities.resolve_projector(model, {}), (None, []))
        self.assertEqual(capabilities.resolve_projector(model, {"vision": True}), ("/mmproj.gguf", []))
        self.assertEqual(capabilities.resolve_projector(model, {"mmproj": "/other.gguf", "vision": True}), ("/other.gguf", []))
        projector, warnings = capabilities.resolve_projector({"path": "/m.gguf"}, {"vision": True})
        self.assertIsNone(projector)
        self.assertEqual(len(warnings), 1)

    def test_projector_without_encoder_flags_counts_as_vision(self) -> None:
        caps = capabilities.profile_capabilities({"path": "/nope.gguf"}, {"mmproj": "/nope-mmproj.gguf"})
        self.assertEqual(caps["input"], ["text", "image"])

    def test_profile_declarations_win_and_bad_values_are_ignored(self) -> None:
        params = {"capabilities": {"tools": False, "input": ["Text", "Image"], "context_max": 32768,
                                   "reranker": "yes", "embedding": 1}}
        caps = capabilities.profile_capabilities(None, params)
        self.assertEqual((caps["tools"], caps["input"], caps["context_max"]), (False, ["text", "image"], 32768))
        self.assertIsNone(caps["reranker"])
        self.assertIsNone(caps["embedding"])
        self.assertEqual(caps["sources"]["tools"], "profile")

    def test_with_served(self) -> None:
        caps = capabilities.profile_capabilities(None, {"capabilities": {"tools": True}})
        served = {"input_modalities": ["text", "image"], "tools": False, "slot_ctx": 8192, "total_slots": 2}
        merged = capabilities.with_served(caps, served)
        self.assertEqual((merged["input"], merged["tools"], merged["slot_ctx"]), (["text", "image"], False, 8192))
        self.assertEqual(merged["sources"]["tools"], "server")
        self.assertIs(capabilities.with_served(caps, None), caps)
        self.assertEqual(capabilities.with_served(None, served)["input"], ["text", "image"])

    def test_filter_profiles(self) -> None:
        profiles = [
            {"mode": "chat", "capabilities": {"input": ["text", "image"], "tools": True}},
            {"mode": "embed", "capabilities": {"input": ["text"], "embedding": True}},
            {"mode": "unknown", "capabilities": None},
        ]
        self.assertEqual([p["mode"] for p in capabilities.filter_profiles(profiles, "tools")], ["chat"])
        self.assertEqual([p["mode"] for p in capabilities.filter_profiles(profiles, "image")], ["chat"])
        self.assertEqual([p["mode"] for p in capabilities.filter_profiles(profiles, "embedding")], ["embed"])
        with self.assertRaises(ValueError):
            capabilities.filter_profiles(profiles, "telepathy")


class LaunchTests(_MetaCache):
    def test_llama_flags(self) -> None:
        argv = build_llama_server_args("/bin/llama-server", "/m.gguf", {"mmproj": "/p.gguf", "reranking": True}).argv
        self.assertEqual(argv[argv.index("--mmproj") + 1], "/p.gguf")
        self.assertIn("--reranking", argv)
        plain = build_llama_server_args("/bin/llama-server", "/m.gguf", {}).argv
        self.assertNotIn("--mmproj", plain)
        self.assertNotIn("--reranking", plain)

    def test_vision_override_is_allowed(self) -> None:
        self.assertEqual(validate_overrides({"vision": True}), {"vision": True})
        with self.assertRaises(ValueError):
            validate_overrides({"mmproj": "/etc/passwd"})

    def test_prepare_passes_projector_and_reports_capabilities(self) -> None:
        profile = ResolvedProfile(
            mode="gemma", name="Gemma", description="", profile={},
            model={"path": "/models/gemma.gguf", "mmproj_path": "/models/mmproj-gemma.gguf"},
            launchable=True, confidence=1.0, params={},
        )
        env = Environment(id="llama.cpp", kind="local", name="llama.cpp", available=True, binary_path="/bin/llama-server")
        with mock.patch.object(server_manager, "_profile_by_mode", return_value=profile), \
                mock.patch.object(server_manager, "detect_llama_cpp", return_value=env):
            text_only = server_manager.prepare_launch_command("gemma", project_root="/tmp", config=AppConfig())
            vision = server_manager.prepare_launch_command("gemma", project_root="/tmp", overrides={"vision": True}, config=AppConfig())
        self.assertNotIn("--mmproj", text_only["command"]["argv"])
        self.assertTrue(text_only["capabilities"]["vision_available"])
        argv = vision["command"]["argv"]
        self.assertEqual(argv[argv.index("--mmproj") + 1], "/models/mmproj-gemma.gguf")
        self.assertEqual(vision["capabilities"]["input"], ["text", "image"])


class ProfilesApiTests(unittest.TestCase):
    def setUp(self) -> None:
        handler = type("H", (ControlRequestHandler,), {})
        handler.control_plane = mock.create_autospec(ControlPlane, instance=True)
        self.control = handler.control_plane
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.addCleanup(self.httpd.server_close)
        self.addCleanup(self.httpd.shutdown)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def _get(self, path: str) -> int:
        try:
            with urllib.request.urlopen(self.base + path, timeout=2) as response:
                return response.status
        except urllib.error.HTTPError as exc:
            return exc.code

    def test_capability_filter(self) -> None:
        self.control.profiles.return_value = []
        self.assertEqual(self._get("/api/profiles"), 200)
        self.control.profiles.assert_called_with()
        self.assertEqual(self._get("/api/profiles?capability=tools"), 200)
        self.control.profiles.assert_called_with("tools")
        self.control.profiles.side_effect = ValueError("capability must be one of ...")
        self.assertEqual(self._get("/api/profiles?capability=telepathy"), 400)

    def test_control_plane_adds_capabilities_and_filters(self) -> None:
        chat = ResolvedProfile(mode="chat", name="c", description="", profile={}, model=None, launchable=True,
                               confidence=1.0, params={"capabilities": {"tools": True}})
        embed = ResolvedProfile(mode="embed", name="e", description="", profile={}, model=None, launchable=True,
                                confidence=1.0, params={"capabilities": {"embedding": True}})
        with mock.patch("inferencedeck.control.resolve_profiles", return_value=[chat, embed]):
            plane = ControlPlane()
            everything = plane.profiles()
            tools = plane.profiles("tools")
        self.assertEqual([p["capabilities"]["tools"] for p in everything], [True, None])
        self.assertEqual([p["mode"] for p in tools], ["chat"])
        json.dumps(everything)  # the API returns it as JSON


if __name__ == "__main__":
    unittest.main()
