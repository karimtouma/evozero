# Benchmark: evozero vs Operon / PySR / gplearn

**Honest, head-to-head symbolic regression benchmark. TL;DR: evozero does *not* beat
the best CPU tools. Operon is the leader.** We publish this because a benchmark that
only appears when you win is marketing, not evidence.

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

## Results (budget 30 s, median of 2 seeds, fp64)

| N (rows) | evozero (GPU) | Operon (CPU) | PySR (CPU) | gplearn (CPU) |
|----------|:-------------:|:------------:|:----------:|:-------------:|
| 1 000    | **1.000**     | 1.000        | 1.000      | 1.000         |
| 10 000   | **1.000**     | 1.000        | 1.000      | 1.000         |
| 100 000  | 0.995 → **0.99999**† | **1.000**    | **1.000**  | —             |
| 1 000 000| ~~OOM~~ → **no OOM**† | **0.9998**   | 0.984      | —             |

† **Update (case subsampling, post-table):** the large-`N` OOM is fixed. evozero now
evaluates fitness on a random case subsample above `N ≈ 83k` and never materializes the
`[P, N]` prediction tensor. Measured on the H100 (constants-heavy target, `subsample_size=2048`,
DALex selection): **N = 10⁵ → R² 0.99999** (was ~0.995) and **N = 10⁶ → R² 0.9995 at ~1 GB
peak** (was a ~40 GB OOM). This closes most of the medium-`N` gap and removes the large-`N`
crash; Operon is still marginally ahead and converges faster. Re-run `benchmark.py` to
reproduce on this exact target.

## Honest conclusions

1. **Small data (N ≤ 10⁴): everyone ties at R² = 1.0.** This is the regime where
   symbolic regression usually operates (PySR itself recommends < 10⁴ points). Here the
   optimized CPU tools are fully competitive — and **Operon reaches the exact solution in
   under a second** when allowed to stop early. The GPU provides no practical advantage.
2. **Medium data (N = 10⁵): Operon and PySR win.** They reach R² = 1.0 reliably; evozero
   reaches ~0.995 and even overshoots the time budget. **Operon's per-evaluation search is
   simply more sample-efficient** (mature GP + local constant optimization via
   SGD/Levenberg-Marquardt).
3. **Large data (N = 10⁶): evozero used to run out of GPU memory — now fixed.** It formerly
   materialized the whole population's predictions `[P, N]` at once (10⁴ × 10⁶ ≈ 40 GB).
   **Case subsampling + a streamed fitness reduction** (see the † note above) drop peak memory
   to ~1 GB and reach R² ≈ 0.9995; Operon still edges it and converges faster. In practice you
   would subsample anyway — evozero now does so automatically.

**Bottom line: "GPU-native" does not mean "faster than a well-optimized CPU
implementation" for symbolic regression.** Operon (C++/SIMD with local constant
optimization) is the tool to beat, and evozero does not beat it. evozero's value is *not*
raw SR speed — it is being PyTorch-native, the scikit-learn API, and the AutoML-Zero
engine (which has no direct competitor).

## What we changed after the first run (and it still wasn't enough)

- **Gradient local constant optimization** (LBFGS through the differentiable interpreter),
  matching what Operon does. This raised evozero's N = 10⁵ result from 0.99 to ~0.995 —
  a real improvement, but not enough to reach a reliable 1.0.
- **Memory-safe auto-chunking** of the interpreter stack (fixed one OOM source; the
  output-tensor OOM at N = 10⁶ remains).

## Reproduce

```bash
# on a machine with an NVIDIA GPU, after: pip install evozero pyoperon pysr gplearn
JULIA_NUM_THREADS=32 python benchmarks/benchmark.py --budget 30 --seeds 5
```
