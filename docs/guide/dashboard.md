# Live dashboard

`evozero` ships a **dependency-free** (Python standard library only) web dashboard that
visualizes a search in real time — the Pareto cloud, the convergence curve, and the
predicted-vs-actual fit of the selected solution, in a dark, Eureqa-style UI.

## Programmatic

```python
from evozero import launch_dashboard

with launch_dashboard(port=8080) as dash:
    print("open", dash.url)          # http://127.0.0.1:8080
    # push metrics from your training loop:
    dash.update({"gen": 10, "best": {"formula": "x0**2 + x1", "r2": 0.98, "complexity": 5}})
# the server stops when the with-block exits
```

`launch_dashboard(model=None, *, host="127.0.0.1", port=8080, open_browser=False, blocking=False)`
: Starts the HTTP server in a background thread and returns a {class}`~evozero.DashboardHandle`.
  Set `blocking=True` to serve in the foreground (Ctrl-C to stop).

`DashboardHandle`
: - `url` — the served URL.
  - `update(state)` — replace the JSON metrics served at `/state` (the page polls it).
  - `stop()` — shut the server down. Also a context manager.

## Command line

```bash
evozero dashboard --port 8080          # serves the dashboard, Ctrl-C to stop
```

## Serving over SSH (remote GPU)

If your search runs on a remote GPU box, run the dashboard there bound to localhost and
forward the port from your laptop:

```bash
# on the GPU server
evozero dashboard --host 127.0.0.1 --port 8770

# on your laptop
ssh -N -L 8770:127.0.0.1:8770 user@gpu-server
# then open http://localhost:8770
```

The dashboard is a single "vulcanized" `index.html` (all CSS/JS inline, no build step,
no CDN) packaged inside the wheel and served with `http.server` — so it adds **zero**
dependencies and works from any install.

```{note}
Live streaming *directly* from `SymbolicRegressor.fit` is on the roadmap. Today you drive
the dashboard by calling `handle.update(...)` from your own callback/loop, or point the
CLI at a run.
```
