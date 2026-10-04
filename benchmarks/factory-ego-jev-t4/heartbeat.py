#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path


class Heartbeat:
    """Single-file task heartbeat. Keeps only the latest state, not history."""

    def __init__(self, path: Path | str, task: str, interval_s: float = 30.0):
        self.path = Path(path)
        self.task = task
        self.interval_s = interval_s
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._state = {
            "task": task,
            "status": "starting",
            "phase": "startup",
            "pid": os.getpid(),
        }
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._write()
        self._thread.start()
        return self

    def update(self, **fields):
        with self._lock:
            self._state.update(fields)
        self._write()

    def finish(self, status: str = "completed", **fields):
        with self._lock:
            self._state.update(fields)
            self._state["status"] = status
        self._write()
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def _snapshot(self):
        with self._lock:
            state = dict(self._state)
        state["updated_at_unix"] = time.time()
        return state

    def _write(self):
        state = self._snapshot()
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n")
        tmp.replace(self.path)

    def _loop(self):
        while not self._stop.wait(self.interval_s):
            self._write()
