from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from inferencedeck import server_manager


class StateTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        patcher = mock.patch.object(server_manager, "cache_dir", return_value=Path(self._tmp.name))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self._tmp.cleanup)

    def test_concurrent_upserts_are_not_lost(self) -> None:
        # Read-modify-write without a lock lets one request overwrite another's
        # entry, leaving a live server untracked.
        barrier = threading.Barrier(16)

        def add(i: int) -> None:
            barrier.wait()
            server_manager._upsert_server({"id": f"s{i}", "mode": f"m{i}"})

        threads = [threading.Thread(target=add, args=(i,)) for i in range(16)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        ids = {s["id"] for s in server_manager.read_state()["servers"]}
        self.assertEqual(ids, {f"s{i}" for i in range(16)})

    def test_trim_never_drops_running_servers(self) -> None:
        procs = [
            subprocess.Popen([sys.executable, "-c", "import time;time.sleep(30)"], start_new_session=True)
            for _ in range(3)
        ]
        try:
            servers = [{"id": "old-a"}, {"id": "old-b"}]  # no pid: history-only records
            servers += [{"id": f"run{i}", "pid": p.pid} for i, p in enumerate(procs)]
            server_manager.write_state({"servers": servers})
            server_manager.trim_server_history(limit=2)
            ids = [s["id"] for s in server_manager.read_state()["servers"]]
            # Every running server survives, even past the limit, and the newest one is kept.
            self.assertEqual(ids, ["run0", "run1", "run2"])
        finally:
            for p in procs:
                p.kill()
                p.wait(timeout=5)

    def test_trim_keeps_newest_history_records(self) -> None:
        server_manager.write_state({"servers": [{"id": f"h{i}"} for i in range(6)]})
        server_manager.trim_server_history(limit=3)
        ids = [s["id"] for s in server_manager.read_state()["servers"]]
        self.assertEqual(ids, ["h3", "h4", "h5"])

    def test_write_state_leaves_no_temp_files(self) -> None:
        server_manager.write_state({"servers": []})
        leftovers = [p for p in os.listdir(self._tmp.name) if p.endswith(".tmp")]
        self.assertEqual(leftovers, [])


if __name__ == "__main__":
    unittest.main()
