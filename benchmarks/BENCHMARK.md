# Benchmark: evozero vs Operon / PySR / gplearn

**Honest, head-to-head symbolic regression benchmark. TL;DR: at the small `N` where SR
usually operates, the mature CPU tools (Operon, PySR) are excellent and evozero holds no
practical edge. At large `N`, case subsampling lets evozero match their R² (= 1.0) while
staying memory-flat (~1.4 GB) and wall-clock faster — but that edge comes from subsampling,
which the CPU tools could also adopt, not from a faster core search.** We publish this whether
or not it flatters evozero — a benchmark that only appears when you win is marketing, not
evidence.

## Methodology (fair by construction)

- **Same everything:** identical operators (`+ - * / sin cos`), identical `max_size = 30`,
  identical wall-clock budget, identical data, same train/test split.
- **CPU gets the hardware:** Operon/PySR/gplearn run with 32 threads on the H100 box's
  192-core CPU. evozero runs on one H100.
- **Native precision:** data is float64 (native for Operon/PySR — they degrade badly on
  float32). evozero casts to float32 internally by design — an honest, minor disadvantage.
- **Warmup excluded:** one discarded `fit()` per method pays Julia JIT / CUDA compile
  off-clock.
- **Metric:** best **test R²** reached within the budget; median over seeds.
- Problem: `y = x0² + x0·x1 + sin(x1)` (ground-truth, exactly recoverable), swept over
  dataset size `N`. Reproduce with `benchmarks/benchmark.py`.

## Results — test R² (median over seeds, fp64 data), wall time in parens

Small/medium `N` at a 30 s budget (2 seeds); the 10⁵/10⁶ rows are **re-measured** at a 60 s
budget (3 seeds) after case subsampling landed — that is where the picture changed.

| N (rows)  | evozero (GPU)             | Operon (CPU)   | PySR (CPU)    | gplearn (CPU) |
|-----------|:------------------------:|:--------------:|:-------------:|:-------------:|
| 1 000     | 1.000                    | 1.000          | 1.000         | 1.000         |
| 10 000    | 1.000                    | 1.000          | 1.000         | 1.000         |
| 100 000   | **1.000** (5–8 s)        | 0.99999 (60 s) | 1.000 (67 s)  | —             |
| 1 000 000 | **1.000** (7–8 s, 1.4 GB)| 0.99986 (60 s) | 1.000 (163 s) | —             |

The large-`N` rows changed markedly from the first run (evozero was ~0.995 at 10⁵ and **OOM**ed
at 10⁶). With **case subsampling** (`subsample_size = 2048`, auto-on above `N ≈ 83k`; identical
for tournament and DALex selection):

- **No OOM, flat memory:** evozero evaluates fitness on a 2048-row subsample and never
  materializes the `[P, N]` prediction tensor, so peak GPU memory is ~1.4 GB at *any* `N`.
- **N-independent cost → fast at large N:** per-generation cost no longer grows with `N`, so
  evozero reaches R² = 1.0 in **5–8 s at both 10⁵ and 10⁶**, while Operon/PySR pay a per-row
  cost that grows with `N` (PySR needs 163 s at 10⁶). **Honest caveat:** this speed edge is a
  property of *subsampling*, which the CPU tools could also adopt (PySR has a `batching`
  option) — it is **not** evidence that evozero's core search is faster per evaluation.

## Honest conclusions

1. **Small data (N ≤ 10⁴): everyone ties at R² = 1.0.** This is the regime where
   symbolic regression usually operates (PySR itself recommends < 10⁴ points). Here the
   optimized CPU tools are fully competitive — and **Operon reaches the exact solution in
   under a second** when allowed to stop early. The GPU provides no practical advantage.
2. **Medium data (N = 10⁵): now a tie.** With case subsampling evozero reaches R² = 1.0
   (it was ~0.995 and overshot the budget before), matching PySR and edging Operon (0.99999),
   in 5–8 s. The gap the first run exposed is closed on this target.
3. **Large data (N = 10⁶): evozero used to OOM — now it is the fastest to R² = 1.0.** It
   formerly materialized the whole population's predictions `[P, N]` (10⁴ × 10⁶ ≈ 40 GB) and
   crashed. **Case subsampling + a streamed fitness reduction** hold peak memory flat at
   ~1.4 GB and, because per-generation cost no longer scales with `N`, evozero solves in ~8 s
   versus Operon (60 s, 0.99986) and PySR (163 s). But see the honest caveat above: that speed
   comes from subsampling, a technique the CPU tools could also use.

**Bottom line:** at the small `N` where symbolic regression usually operates, the mature CPU
tools (Operon C++/SIMD, PySR Julia) are excellent and **evozero holds no practical edge** —
"GPU-native" is not intrinsically faster per evaluation. At large `N`, subsampling lets evozero
match their R² while staying memory-flat and wall-clock faster, but that is a property of
subsampling rather than a better core search. evozero's distinctive value remains being
PyTorch-native, the scikit-learn API, GPU-native lexicase (DALex), and the AutoML-Zero engine.

## What we changed after the first run

- **Gradient local constant optimization** (LBFGS through the differentiable interpreter),
  matching what Operon does — raised the first-run N = 10⁵ result from 0.99 to ~0.995.
- **GPU-native lexicase selection (DALex)** — near-free on GPU; improves fit on
  constants-heavy targets (see `CHANGELOG`).
- **Case subsampling + streamed fitness reduction** — removed the large-`N` output-tensor OOM
  entirely and made per-generation cost independent of `N`. This is what turned the 10⁵/10⁶
  rows from *"loses / OOM"* into *"ties, memory-flat, fast."*

## Reproduce

```bash
# on a machine with an NVIDIA GPU, after: pip install evozero pyoperon pysr gplearn
JULIA_NUM_THREADS=32 python benchmarks/benchmark.py --budget 30 --seeds 5
```
