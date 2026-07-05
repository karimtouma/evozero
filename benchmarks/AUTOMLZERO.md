# Benchmark: AutoML-Zero search efficiency (evozero)

**Honest study of evozero's learning-algorithm search. TL;DR: at tractable single-machine
scale, evolution ties Random Search on this VM (the landscape is a sparse plateau), but the
GPU-oriented efficiency upgrade FEC is a measured win — it reaches higher fitness at fewer
than half the full evaluations.** We publish results whether or not they flatter the tool.

## What this measures — and why on the *training-steps* axis

Following Real et al. (AutoML-Zero, ICML 2020), the fair, hardware-agnostic budget is the
**total number of inner LEARN iterations executed**, *not* wall-clock (hardware-specific)
and *not* "number of algorithms evaluated" — once FEC skips full evaluations and Hurdles
truncate them, an "evaluation" is no longer a fixed unit of work. Every searcher is capped
at the same training-step budget; we report best-fitness-so-far at that cap.

Fitness = mean R² of an evolved `Setup/Predict/Learn` algorithm, trained from scratch and
evaluated on held-out data, averaged over a distribution of tasks (rewards *generalizing*
learning rules, i.e. meta-learning). The harness reuses evozero's AutoML-Zero VM primitives
and implements the search variants on top, so the shipped engine is untouched
(`benchmarks/automlzero_bench.py`; same pattern as `ab_selection.py`).

**Searchers** (identical VM search space + identical step budget):
- **random_search** — all-random algorithms, keep the best (no selection).
- **(1+1)-ES** — single incumbent, always-mutate, elitist accept (an *added* baseline, not
  in the paper — labeled as such).
- **aging_evolution** — regularized/aging evolution (evict the oldest, tournament `T=10`,
  mutation rate `U=0.9`); *vanilla* = FEC/Hurdles off.
- **FEC** (functional equivalence caching) and **Hurdles** (early-stop weak candidates),
  toggled one at a time. FEC charges its 10-step probe on **every** candidate (hit and
  miss); a hit then skips the ~150-step full evaluation.

## The landscape (why the headline result looks the way it does)

Sampling 500 random algorithms on a linear task: exactly **1** scores R² > 0; the median is
**0.0** (a "predict-zero" plateau). Mutating the single good one 300×: **0** children
improve. The AutoML-Zero VM's fitness surface at this scale is a flat plateau of
functionally-identical zero-predictors with rare, isolated spikes — **random-discovery
limited, not gradient-climbable.** This is precisely the regime in which the paper needed
massive compute (≈10⁴ CPU cores, billions of steps) for evolution's building-block
advantage to emerge.

## Results (budget 400,000 training steps, 3 seeds, 5 non-linear tasks, CPU)

Tiny per-op tensors (F=4, N=200) make this a CPU workload — GPU kernel-launch overhead
dominates and is *slower* here. Reference: a hand-coded linear-regression-by-GD algorithm
scores **0.6307** (the practical ceiling on these tasks).

| searcher            | best fit (median) | IQR              | full evals |
|---------------------|:-----------------:|:----------------:|:----------:|
| random_search       | **0.6028**        | [0.301, 0.603]   | ~2667      |
| (1+1)-ES *[added]*  | 0.0000            | [0.000, 0.301]   | ~2667      |
| aging (vanilla)     | **0.6028**        | [0.301, 0.603]   | 2667       |

### FEC / Hurdles ablation (aging evolution; one upgrade at a time)

| config              | best fit (median) | full evals | FEC hits |
|---------------------|:-----------------:|:----------:|:--------:|
| vanilla             | 0.6028            | 2667       | 0        |
| **+FEC**            | **0.6152**        | **1141**   | 21 744   |
| +Hurdles            | 0.6028            | 2795       | 0        |
| full (FEC+Hurdles)  | 0.6042            | **581**    | 35 554   |

## Honest conclusions

1. **Evolution ties Random Search here (both ≈0.603, near the 0.631 ceiling).** On this VM
   at tractable scale, evolution does **not** beat RS — the landscape is a needle-in-a-plateau
   and both are limited by random discovery. (1+1)-ES is worst: a single mutating lineage
   rarely lands on a spike. This matches the paper's finding that its evolution advantage
   requires far more compute than a single machine provides.
2. **FEC is a real, measured win.** At the *same* training-step budget it reaches **higher**
   fitness (0.603 → 0.615) with **fewer than half** the full evaluations (2667 → 1141),
   because it dedupes the ~22k functionally-equivalent candidates (mostly zero-predictors)
   at 10-step probe cost instead of 150-step full cost — so the search explores more
   candidates per unit of compute. With Hurdles added, the *full* config reaches equal
   quality at **4.6× fewer** full evaluations (581 vs 2667). This is exactly why FEC exists,
   and it is the honest, GPU/compute-oriented lever the AutoML-Zero engine benefits from.
3. **Hurdles are neutral on this landscape.** Almost every candidate is a zero-predictor, so
   the rolling 75th-percentile hurdle rarely changes which structures get discovered; it
   trims per-candidate cost without a fitness gain here (it would matter more on tasks with a
   graded difficulty distribution across the task set).

**Bottom line:** evozero's AutoML-Zero search is *sound* (it matches Random Search and
approaches the hand-coded reference), and its efficiency machinery (FEC especially) measurably
improves search-per-compute — but on this VM at single-machine scale it does not demonstrate
an evolution-beats-random advantage. That advantage is a large-compute phenomenon; claiming it
from a laptop-scale run would be dishonest.

## Reproduce

```bash
pip install evozero          # torch extra: pip install "evozero[torch]"
python benchmarks/automlzero_bench.py    # ~8 min on a CPU core; edit `budget` to sweep
```
