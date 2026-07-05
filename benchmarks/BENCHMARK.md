# Benchmark: evozero vs Operon / PySR / gplearn

**Honest, head-to-head symbolic regression benchmark. TL;DR: on equal hardware the mature CPU
tools (Operon C++/SIMD, PySR Julia) clearly beat evozero. On CPU with full data evozero scores
R² 0.994 / 0.867 at N = 10⁴ / 10⁵ versus ~1.0 for Operon/PySR. evozero's apparent large-`N`
speed advantage is an artifact of *subsampling* (which the CPU tools can also do) plus the GPU —
not a better search.** We publish this whether or not it flatters evozero — a benchmark that only
appears when you win is marketing, not evidence.

## Methodology

- **Same everything:** identical operators (`+ - * / sin cos`), identical `max_size = 30`,
  identical wall-clock budget, identical data, same train/test split.
- **Native precision:** data is float64 (native for Operon/PySR — they degrade badly on
  float32). evozero casts to float32 internally by design — an honest, minor disadvantage.
- **Warmup excluded:** one discarded `fit()` per method pays Julia JIT / CUDA compile off-clock.
- **Metric:** best **test R²** reached within the budget; median over seeds.
- Problem: `y = x0² + x0·x1 + sin(x1)` (ground-truth, exactly recoverable), swept over `N`.
- Reproduce: `benchmarks/benchmark.py` (main sweep), `bench_large_n.py` (large-N + peak memory),
  `fair_compare.py` (the hardware/technique-controlled decomposition below).

## 1. Raw head-to-head (as first published)

evozero on **one H100**; Operon/PySR/gplearn on **32 CPU threads**. Test R² (median), fp64.
Small/medium `N` at 30 s; the 10⁵/10⁶ rows at 60 s after case subsampling landed.

| N (rows)  | evozero (GPU)             | Operon (CPU)   | PySR (CPU)    | gplearn (CPU) |
|-----------|:------------------------:|:--------------:|:-------------:|:-------------:|
| 1 000     | 1.000                    | 1.000          | 1.000         | 1.000         |
| 10 000    | 1.000                    | 1.000          | 1.000         | 1.000         |
| 100 000   | 1.000 (5–8 s)            | 0.99999 (60 s) | 1.000 (67 s)  | —             |
| 1 000 000 | 1.000 (7–8 s, ~1.4 GB)   | 0.99986 (60 s) | 1.000 (163 s) | —             |

At large `N` evozero looks fastest — **but this table mixes two confounds**: evozero runs on a
GPU *and* subsamples to 2048 rows, while the CPU tools run on full data. Section 2 removes both
confounds. It reverses the conclusion.

## 2. Fairness — decomposing the large-`N` result

No competitor has a GPU backend (Operon/PySR/gplearn are CPU-only), so a literal "GPU vs GPU" is
impossible. Instead we isolate the two advantages. Budget 30 s, 2 seeds, evozero `pop = 3000`.
Reproduce with `benchmarks/fair_compare.py`.

**A) CPU vs CPU — same hardware, same (full) data.** *Is evozero's search competitive?*

| N (rows)  | evozero (CPU, full) | Operon (CPU)   | PySR (CPU)   |
|-----------|:-------------------:|:--------------:|:------------:|
| 10 000    | **0.994** (21 s)    | 0.99998 (30 s) | 1.000 (31 s) |
| 100 000   | **0.867** (77 s)    | 0.99999 (30 s) | 1.000 (39 s) |

On equal hardware evozero **loses clearly** — and at 10⁵ it cannot even hold the time budget
(77 s), because evaluating a full-`N` population without the GPU is slow, so it completes very few
generations. Its GP search is also less sample-efficient than Operon's (mature GP + local
constant optimization). **No algorithmic advantage.**

**B) evozero GPU vs CPU (subsampling on) — what the GPU actually buys evozero.**

| N (rows)  | evozero (GPU, sub 2048) | evozero (CPU, sub 2048) |
|-----------|:-----------------------:|:-----------------------:|
| 100 000   | 1.000 (4 s, 0.8 GB)     | 0.996 (25 s)            |
| 1 000 000 | 0.99996 (17 s, 0.8 GB)  | 0.995 (22 s)            |

The GPU gives evozero ~5–6× throughput and better convergence on *its own* workload — but only
brings it **up to** competitive, not past the CPU tools' search.

**C) Subsampling isolation @ N = 10⁶ — give the CPU tools the SAME 2048 rows.**

| method                    | test R²   | sec  |
|---------------------------|:---------:|:----:|
| evozero (GPU, sub 2048)   | 0.99996   | 17 s |
| Operon (CPU, 2048 rows)   | **0.99999** | 30 s |
| PySR (CPU, 2048 rows)     | **1.000** | 30 s |

Given the same subsample, Operon and PySR **match or beat** evozero. This proves the caveat with
data: **the large-`N` speed advantage was subsampling, not the GPU and not a better search.**
Subsampling is a technique any tool can adopt (PySR ships a `batching` option).

## Honest conclusions

1. **On equal hardware, Operon and PySR beat evozero** (Section 2A): 0.994 / 0.867 vs ~1.0 at
   N = 10⁴ / 10⁵. Karim's original skepticism holds — "GPU-native" is not intrinsically better,
   and here the CPU tools' search is simply stronger.
2. **The GPU helps evozero, but only to reach parity** (2B): ~5–6× faster than evozero-on-CPU,
   yet still not better than the CPU competitors' algorithm.
3. **The large-`N` "win" was subsampling** (2C): with the same 2048 rows the CPU tools match or
   beat evozero. Case subsampling is real and useful (it removed the OOM and made evozero usable
   at any `N`), but it is not a moat.

**Bottom line: evozero does not beat the best CPU symbolic-regression tools — on equal terms it
loses.** Its value is *not* SR performance. It is being PyTorch-native (drop into a torch/CUDA
stack, autograd constant optimization), the scikit-learn API, GPU-native lexicase selection
(DALex, near-free on GPU), and the AutoML-Zero engine (no direct competitor). If you need the
best symbolic regression today, use **Operon** (or **PySR** for production ergonomics).

## What we changed across runs

- **Gradient local constant optimization** (LBFGS through the differentiable interpreter) —
  raised the first-run N = 10⁵ result from 0.99 to ~0.995.
- **GPU-native lexicase selection (DALex)** — near-free on GPU; helps on constants-heavy targets.
- **Case subsampling + streamed fitness reduction** — removed the large-`N` OOM and made
  per-generation cost independent of `N`. Genuinely useful, but Section 2 shows it is not a
  competitive advantage over CPU tools that subsample too.

## Reproduce

```bash
# on a machine with an NVIDIA GPU, after: pip install evozero pyoperon pysr gplearn
JULIA_NUM_THREADS=32 python benchmarks/benchmark.py --budget 30 --seeds 5   # main sweep
python benchmarks/bench_large_n.py                                          # large-N + peak mem
python benchmarks/fair_compare.py                                           # CPU-vs-CPU decomposition
```
