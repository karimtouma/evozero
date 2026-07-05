# AutoML-Zero

Most AutoML tunes the hyperparameters of a *fixed* model. `evozero` goes one level up: it
evolves the **learning algorithm itself**, from primitive operations — an idea introduced
by [Real et al. (2020)](https://arxiv.org/abs/2003.03384), adapted here to full-batch
training on the GPU.

## The idea

Each candidate is an *algorithm* with three functions over a typed-register virtual
machine (scalars, vectors of dimension `F`, and per-example buffers of dimension `N`):

```text
Setup()        initialize the weights / hyperparameters
Predict(X)     produce ŷ                           (may NOT read the labels)
Learn(X, y)    UPDATE the weights given the error  ← the adaptive rule
```

The evolved `Learn` function is what makes the model **adaptive** — it can rediscover
gradient descent, a one-shot least-squares rule, or something new. Fitness is the mean
`R²` across **several tasks**, so evolution is rewarded for algorithms that *generalize*
to tasks they were never trained on — this is meta-learning, "learning to learn".

## Usage

```python
import numpy as np
from evozero import LearnerSearch, Task

# Build a distribution of tasks (each is a (X, y) problem).
rng = np.random.default_rng(0)
tasks = []
for s in range(6):
    w = rng.normal(size=5)
    X = rng.normal(size=(256, 5)).astype("float32")
    y = X @ w
    tasks.append(Task(X, y, random_state=s))

search = LearnerSearch(
    population_size=200,
    n_meta_generations=400,
    max_time=120,          # give it a real budget; discovery is compute-heavy
    device="auto",
    random_state=0,
)
search.fit(tasks)

print("meta score (mean R²):", search.best_score_)
learner = search.export_learner()          # an EvolvedLearner
print(learner.to_python_source())          # the discovered algorithm as code
```

### Reusing the discovered learner

`EvolvedLearner` is itself a `fit`/`predict` estimator — it trains from scratch on new
data using the evolved `Learn` rule:

```python
learner.fit(X_train, y_train)
preds = learner.predict(X_test)
```

## API

`Task(X, y, *, val_fraction=0.3, random_state=0)`
: A meta-training task. The target is centered (the VM has no bias term) and split into
  train/validation internally.

`LearnerSearch(*, population_size=200, n_meta_generations=400, steps=30, max_time=None, device="auto", random_state=None, verbose=0)`
: The meta-search. `steps` is the number of inner training steps used to score an
  algorithm on each task.

  - `best_program_` : {class}`~evozero.EvolvedLearner` — the discovered learner.
  - `best_score_` : float — its mean validation `R²`.
  - `export_learner()` → {class}`~evozero.EvolvedLearner`.

`EvolvedLearner`
: `fit(X, y)` (train from scratch via the evolved rule), `predict(X)`,
  `to_python_source()` (the algorithm as readable pseudo-code).

## What to expect

On simple regression tasks, evolution reliably discovers a working learner and often
finds a **non-obvious** rule. For example, on standardized linear tasks it tends to find
the one-shot correlation rule `w = Xᵀy / N` (a Hebbian / normal-equations estimator that
converges in a single step) rather than iterative gradient descent.

```{admonition} Honest note on compute
:class: important
AutoML-Zero *from scratch* is genuinely compute-hungry — the original work used large
clusters. A short CPU run may not converge. Give `LearnerSearch` a real budget (GPU +
minutes to hours) and a modest task set to see it discover good learners. Nonlinear/MLP
tasks are the hard frontier and need the most compute.
```
