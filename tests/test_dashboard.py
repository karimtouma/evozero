"""Tests for the stdlib dashboard server."""

from __future__ import annotations

import json
import urllib.request

from evozero import DashboardHandle, launch_dashboard


def _get(url: str) -> tuple[int, bytes]:
    with urllib.request.urlopen(url, timeout=5) as r:
        return r.status, r.read()


def test_launch_serve_and_stop() -> None:
    with launch_dashboard(host="127.0.0.1", port=8791) as dash:
        assert isinstance(dash, DashboardHandle)
        assert dash.url == "http://127.0.0.1:8791"

        status, body = _get(dash.url + "/")
        assert status == 200
        assert b"<canvas" in body

        dash.update({"gen": 42, "best": {"r2": 0.98}})
        status, sbody = _get(dash.url + "/state")
        assert status == 200
        state = json.loads(sbody)
        assert state["gen"] == 42
        assert state["best"]["r2"] == 0.98


def test_stop_closes_server() -> None:
    dash = launch_dashboard(host="127.0.0.1", port=8792)
    _get(dash.url + "/state")  # reachable
    dash.stop()
    try:
        _get(dash.url + "/state")
    except Exception:
        return  # expected: connection refused after stop
    raise AssertionError("server should be unreachable after stop()")
