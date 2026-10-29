"""Benchmark progress, reported to GPUStack's ``PATCH /v2/benchmarks/{id}/state``.

Same contract as benchmark-runner's ``ServerBenchmarkerProgress``: the body is
``{"progress": <0-100>}`` with the worker token as a bearer credential, sent at
most once a second unless it moved by 2 points or reached 100. The ramp drives
it through ``run_index`` / ``run_total`` (one slice of the bar per measured
point) and ``_update_progress``; a point reports its own completed fraction
through ``report_fraction``.

Progress is telemetry. A failed update is logged and dropped: the measurement
is the report files, and an hour-long run must not die because one PATCH did.
"""

from __future__ import annotations

import asyncio
import json
import ssl
import threading
import time
import urllib.request
from typing import Optional
from urllib.parse import urlparse


class ServerProgress:
    def __init__(
        self,
        progress_url: str,
        progress_auth: Optional[str] = None,
        ca_cert: Optional[str] = None,
        insecure_skip_tls_verify: bool = False,
    ):
        self.progress_url = progress_url
        self.progress_auth = progress_auth
        self.ca_cert = ca_cert
        self.insecure_skip_tls_verify = insecure_skip_tls_verify
        self.run_index = 0
        self.run_total = 1
        self._last_update_ts = 0.0
        self._last_progress = -1.0
        self._lock = threading.Lock()

    def _context(self) -> Optional[ssl.SSLContext]:
        if urlparse(self.progress_url).scheme != "https":
            return None
        if self.insecure_skip_tls_verify:
            context = ssl.create_default_context()
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
            return context
        if self.ca_cert:
            return ssl.create_default_context(cafile=self.ca_cert)
        return None

    def send(self, progress: float) -> None:
        progress = min(100.0, max(0.0, progress))
        with self._lock:
            now = time.time()
            if not (
                now - self._last_update_ts >= 1.0
                or progress >= 100.0
                or progress - self._last_progress >= 2.0
            ):
                return
            self._last_update_ts = now
            try:
                request = urllib.request.Request(
                    self.progress_url,
                    data=json.dumps({"progress": progress}).encode(),
                    method="PATCH",
                    headers={"Content-Type": "application/json"},
                )
                if self.progress_auth:
                    request.add_header("Authorization", f"Bearer {self.progress_auth}")
                with urllib.request.urlopen(
                    request, timeout=30, context=self._context()
                ):
                    pass
                self._last_progress = progress
            except Exception as e:  # noqa: BLE001 - telemetry must not fail a run
                print(f"[WARN] Failed to update progress to server: {e}", flush=True)

    def report_fraction(self, fraction: float) -> None:
        """Progress within the current run slice (0..1)."""
        self.send((self.run_index + fraction) / max(1, self.run_total) * 100.0)

    async def _update_progress(self, progress: float) -> None:
        await asyncio.to_thread(self.send, progress)

    async def on_finalize(self) -> None:
        return None
