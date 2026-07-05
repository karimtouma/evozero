"""stdlib HTTP handler that serves the packaged dashboard and live metrics."""

from __future__ import annotations

import importlib.resources
import json
import threading
from http.server import BaseHTTPRequestHandler
from typing import Any

_STATE: dict[str, Any] = {"status": "iniciando"}
_LOCK = threading.Lock()


def set_state(state: dict[str, Any]) -> None:
    """Replace the live metrics served at ``/state``."""
    with _LOCK:
        _STATE.clear()
        _STATE.update(state)


def _index_html() -> bytes:
    # Works from a wheel, zip, or editable install; never uses __file__ paths.
    return (
        importlib.resources.files("evozero.dashboard").joinpath("_static/index.html").read_bytes()
    )


class DashboardHandler(BaseHTTPRequestHandler):
    """Serves ``/`` (HTML) and ``/state`` (JSON metrics)."""

    def _send(self, body: bytes, ctype: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        """Serve the dashboard HTML at ``/`` and the JSON metrics at ``/state``."""
        if self.path.startswith("/state"):
            with _LOCK:
                body = json.dumps(_STATE).encode()
            self._send(body, "application/json")
        else:
            self._send(_index_html(), "text/html; charset=utf-8")

    def log_message(self, *args: Any) -> None:
        """Silence the stdlib request logging."""
