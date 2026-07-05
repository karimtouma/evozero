"""``evozero`` command-line interface (stdlib argparse, no extra deps)."""

from __future__ import annotations

import argparse
import sys

from . import __version__


def _cmd_version(_: argparse.Namespace) -> int:
    print(f"evozero {__version__}")
    return 0


def _cmd_fit(args: argparse.Namespace) -> int:
    import numpy as np

    from . import SymbolicRegressor

    data = np.loadtxt(args.csv, delimiter=",", skiprows=1)
    X, y = data[:, :-1], data[:, -1]
    model = SymbolicRegressor(
        population_size=args.pop,
        generations=args.generations,
        max_time=args.max_time,
        device=args.device,
        verbose=1,
    )
    model.fit(X, y)
    print(f"R^2(test) = {model.score(X, y):.4f}")
    print(f"best: y = {model.best_equation_}")
    return 0


def _cmd_dashboard(args: argparse.Namespace) -> int:
    from .dashboard import launch_dashboard

    handle = launch_dashboard(host=args.host, port=args.port, blocking=True)
    print(f"dashboard at {handle.url}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``evozero`` console script."""
    parser = argparse.ArgumentParser(
        prog="evozero", description="GPU-native evolutionary computation."
    )
    parser.add_argument("--version", action="store_true", help="print version and exit")
    sub = parser.add_subparsers(dest="command")

    p_fit = sub.add_parser("fit", help="fit a SymbolicRegressor on a CSV (last column = target)")
    p_fit.add_argument("csv")
    p_fit.add_argument("--pop", type=int, default=3000)
    p_fit.add_argument("--generations", type=int, default=200)
    p_fit.add_argument("--max-time", type=float, default=None, dest="max_time")
    p_fit.add_argument("--device", default="auto")
    p_fit.set_defaults(func=_cmd_fit)

    p_dash = sub.add_parser("dashboard", help="serve the live dashboard")
    p_dash.add_argument("--host", default="127.0.0.1")
    p_dash.add_argument("--port", type=int, default=8080)
    p_dash.set_defaults(func=_cmd_dashboard)

    args = parser.parse_args(argv)
    if args.version:
        return _cmd_version(args)
    if not getattr(args, "command", None):
        parser.print_help()
        return 1
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
