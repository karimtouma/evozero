"""Live, dependency-free (stdlib) web dashboard for a search run."""

from __future__ import annotations

import threading
from http.server import ThreadingHTTPServer
from types import TracebackType
from typing import Any

from .server import DashboardHandler, set_state

__all__ = ["launch_dashboard", "DashboardHandle"]


class DashboardHandle:
    """Controls a running dashboard server.

    Update the live view with :meth:`update`; stop with :meth:`stop` or by
    leaving the ``with`` block.
    """

    def __init__(self, server: ThreadingHTTPServer, url: str) -> None:
        self._server = server
        self.url = url

    def update(self, state: dict[str, Any]) -> None:
        """Set the metrics dict served at ``/state`` (polled by the page)."""
        set_state(state)

    def stop(self) -> None:
        """Shut the server down."""
        self._server.shutdown()
        self._server.server_close()

    def __enter__(self) -> "DashboardHandle":
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None,
                 tb: TracebackType | None) -> None:
        self.stop()


def launch_dashboard(
    model: Any = None,
    *,
    host: str = "127.0.0.1",
    port: int = 8080,
    open_browser: bool = False,
    blocking: bool = False,
) -> DashboardHandle:
    """Start the dashboard server and return a :class:`DashboardHandle`.

    Parameters
    ----------
    model : object, optional
        Reserved for future auto-wiring; pass metrics via :meth:`DashboardHandle.update`.
    host, port : str, int
        Bind address.
    open_browser : bool, default=False
        Open the URL in the default browser.
    blocking : bool, default=False
        If ``True``, serve forever in the current thread (Ctrl-C to stop).
    """
    server = ThreadingHTTPServer((host, port), DashboardHandler)
    url = f"http://{host}:{port}"
    if open_browser:  # pragma: no cover
        import webbrowser

        webbrowser.open(url)
    if blocking:  # pragma: no cover
        try:
            server.serve_forever()
        finally:
            server.server_close()
        return DashboardHandle(server, url)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return DashboardHandle(server, url)
