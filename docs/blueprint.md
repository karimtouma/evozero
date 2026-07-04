# BLUEPRINT DE PUBLICACIÓN — SDK `evozero`

**De cero a PyPI al más alto estándar de Python (julio 2026)**
Autor del blueprint: arquitecto. Destinatario: Karim. Todo lo de abajo está calibrado a herramientas verificadas como vigentes en jul-2026 y es copy-paste-able.

> **Nota de honestidad técnica up-front (lo único incierto):** el versionado dinámico por git-tag (requisito tuyo) **obliga** a elegir `hatchling`+`hatch-vcs` como backend, **no** `uv_build`. A jul-2026, `uv_build` nativo aún NO tiene plugin de versionado VCS compatible (ni `hatch-vcs` ni `uv-dynamic-versioning` lo soportan). Por eso este blueprint usa **hatchling** como backend de build y **uv** solo como gestor de entorno/lock. Ganas versionado por tag; pierdes la config "cero" de uv_build. Es el trade-off correcto para un SDK OSS con releases frecuentes.

---

## 1. Nombre recomendado

| Rol | Nombre | `import` | PyPI | GitHub | Evidencia |
|---|---|---|---|---|---|
| **RECOMENDADO** | **evozero** | `evozero` | LIBRE | 0 colisiones | `GET pypi.org/pypi/evozero/json` → 404; `gh search repos evozero` → 0 resultados (ni fuzzy) |
| Respaldo 1 | glyphene | `glyphene` | LIBRE | 0 colisiones | 404 en PyPI; 0 hits GitHub |
| Respaldo 2 | evoforge | `evoforge` | LIBRE | 5 hits low-signal | 404 en PyPI; colisión fuzzy menor (`slink/evoforge` inactivo) |

**Decisión: `evozero`** (nombre PyPI = `evozero`, import = `evozero`). Único candidato con **cero colisión** en ambos registros, identificador Python válido (minúsculas, sin guiones), corto y pronunciable, y el único que amarra los dos motores: `evo` = búsqueda evolutiva (regresión simbólica + islas), `zero` = AutoML-Zero (evolucionar el algoritmo de aprendizaje desde cero).

Tagline oficial: **"Evolve formulas and learning algorithms from zero — GPU-native evolutionary computation."**

**No usar `evogp`** pese a estar libre en PyPI: colisiona con `EMI-Group/evogp` (295★, activo, mismo nicho GPU+PyTorch+GP+regresión simbólica) → confusión de marca directa.

**Acción inmediata (hoy):** registra el nombre haciendo el primer `git push` del repo y publicando `0.0.1` a TestPyPI, o reserva vía primer release real. El "pending publisher" de PyPI **no reserva el nombre** hasta el primer publish exitoso — hay ventana de robo de nombre.

---

## 2. Licencia recomendada: **Apache-2.0**

**Razón (verificada, vigente 2026):**
1. **Cláusula de patentes explícita (Sección 3)** — relevante porque regresión simbólica evolutiva y AutoML-Zero son candidatos plausibles a reclamos de patente de ML; protege a Karim y a los adoptantes corporativos. MIT/BSD solo otorgan copyright, dejando ambigüedad de patentes.
2. **Estándar de facto del ecosistema ML/GPU**: PyTorch, JAX, cuML/RAPIDS, PySR, EvoTorch, vLLM usan Apache-2.0. Pasa revisión legal enterprise más fácil.
3. Aceptada por OSI, JOSS y pyOpenSci sin fricción.

> Contrapunto registrado: el ecosistema *científico* puro (numpy/scipy/sklearn/gplearn) usa BSD-3-Clause y algunos revisores lo prefieren por paridad. Ambas son válidas; **Apache-2.0 gana aquí por la carga algorítmica patentable + el hecho de que el core es GPU/PyTorch (ecosistema Apache).** Si en el futuro quieres que sklearn mismo dependa de ti, reconsidéralo — pero eso es improbable.

Archivos: `LICENSE` (texto Apache-2.0 completo) + `NOTICE` (si reusas fragmentos Apache de terceros). Metadata PEP 639: `license = "Apache-2.0"` + `license-files = ["LICENSE", "NOTICE"]`. **No** agregues clasificador Trove `License ::` redundante (backends nuevos lo rechazan junto al string SPDX).

---

## 3. Árbol del repositorio (src/ layout)

```
evozero/
├── src/
│   └── evozero/
│       ├── __init__.py            # PEP 562 __getattr__ lazy; re-exporta API pública; __version__
│       ├── py.typed               # PEP 561 marker (CRÍTICO)
│       ├── _import_utils.py       # HAS_TORCH via find_spec; is_torch_available() lru_cache
│       ├── _device.py             # resolve_device("auto") -> torch.device
│       ├── _compat.py             # shim opcional de sklearn (BaseEstimator o duck-typing propio)
│       ├── cli.py                 # entry point argparse (stdlib)
│       │
│       ├── core/                  # ── ÚNICO paquete que importa torch incondicionalmente ──
│       │   ├── __init__.py
│       │   ├── vm.py              # intérprete postfix tensorizado (torch)
│       │   ├── evolution.py       # bucle evolutivo genérico ask()/tell() (islas apiladas en tensor)
│       │   ├── islands.py         # modelo de islas + migración + restart (indexing GPU)
│       │   ├── constants.py       # optimización de constantes (LBFGS/Adam)
│       │   └── ops.py             # tabla de operadores primitivos tensorizados
│       │
│       ├── sr/                    # ── Motor 1: regresión simbólica (fachada sklearn-like) ──
│       │   ├── __init__.py        # __all__ = ["SymbolicRegressor", "Equation"]
│       │   ├── estimator.py       # SymbolicRegressor(fit/predict/score)
│       │   ├── pareto.py          # frente de Pareto (complejidad vs error)
│       │   ├── scaling.py         # linear scaling
│       │   ├── dimensional.py     # análisis dimensional opcional
│       │   └── export.py          # to_sympy / to_latex / to_numpy_func / to_torch_module
│       │
│       ├── automlzero/            # ── Motor 3: AutoML-Zero ──
│       │   ├── __init__.py        # __all__ = ["LearnerSearch", "EvolvedLearner", "Task"]
│       │   ├── vm.py              # VM de registros tipados: ops Setup/Predict/Learn
│       │   ├── search.py          # LearnerSearch: meta-búsqueda + meta-entrenamiento entre tareas
│       │   ├── learner.py         # EvolvedLearner (fit/predict + to_python_source)
│       │   └── ops.py             # instrucciones permitidas / mutación / crossover
│       │
│       └── dashboard/             # ── Motor 2: dashboard (SOLO stdlib) ──
│           ├── __init__.py        # __all__ = ["launch_dashboard", "DashboardHandle"]
│           ├── server.py          # subclase http.server.BaseHTTPRequestHandler
│           ├── app.py             # DashboardHandle, threading
│           └── _static/           # empaquetado como package-data
│               └── index.html     # HTML/CSS/JS "vulcanizado" (todo inline, sin build step)
│
├── tests/                         # FUERA de src/ — importa el paquete YA INSTALADO
│   ├── conftest.py                # registra/skip marker gpu automáticamente
│   ├── test_sr.py
│   ├── test_automlzero.py
│   ├── test_dashboard.py
│   ├── test_sklearn_compat.py     # check_estimator condicionado a sklearn instalado
│   └── property/
│       ├── test_roundtrip.py      # hypothesis: árbol -> latex/numpy -> re-evaluación
│       └── test_vm_invariants.py  # hypothesis: fuzzing VM postfix + CPU/GPU
│
├── examples/                      # scripts sphinx-gallery (# %% cells)
│   ├── plot_symbolic_regression.py
│   ├── plot_dashboard.py
│   └── plot_automl_zero.py
│
├── docs/                          # Sphinx
│   ├── conf.py
│   ├── index.md
│   └── api/
│
├── .github/
│   ├── workflows/
│   │   ├── ci.yml                 # test matrix CPU + lint + type
│   │   ├── release.yml            # build -> TestPyPI -> PyPI (OIDC)
│   │   └── nightly-gpu.yml        # self-hosted GPU, no bloqueante
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.yml
│   │   └── feature_request.yml
│   └── PULL_REQUEST_TEMPLATE.md
│
├── pyproject.toml
├── uv.lock                        # committeado
├── noxfile.py
├── .pre-commit-config.yaml
├── .readthedocs.yaml
├── codecov.yml
├── CHANGELOG.md                   # Keep a Changelog (towncrier compila fragments)
├── CITATION.cff
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md             # Contributor Covenant v2.1
├── SECURITY.md
├── LICENSE                        # Apache-2.0
├── NOTICE
└── README.md
```

**Reglas de aislamiento arquitectónico (verificadas contra PySR/EvoTorch/cuML):**
- Solo `core/` importa `torch` en top-level. `sr/`, `automlzero/`, `dashboard/`, `__init__.py` **nunca** importan torch en top-level.
- `dashboard/` no importa `core/` ni torch — solo lee un dict de métricas por generación (`run_details_`) que le pasa el estimador.
- `_static/index.html` se lee vía `importlib.resources.files("evozero.dashboard").joinpath("_static/index.html").read_bytes()`, **nunca** por rutas relativas a `__file__`.

---

## 4. `pyproject.toml` completo (PEP 621 + PEP 639)

Listo para pegar. Backend **hatchling + hatch-vcs** (versionado por git-tag). Ajusta la versión CUDA (`cu128`) a la vigente para tus H100.

```toml
[build-system]
requires = ["hatchling>=1.30", "hatch-vcs>=0.5"]
build-backend = "hatchling.build"

# ─────────────────────────── Metadata (PEP 621 + PEP 639) ───────────────────────────
[project]
name = "evozero"
dynamic = ["version"]                       # resuelto por hatch-vcs desde git tags
description = "GPU-native evolutionary computation: evolve formulas (symbolic regression) and learning algorithms (AutoML-Zero) from zero."
readme = "README.md"
requires-python = ">=3.11"
license = "Apache-2.0"                       # PEP 639 SPDX string
license-files = ["LICENSE", "NOTICE"]
authors = [{ name = "Karim Touma", email = "ktouma@deacero.com" }]
keywords = ["symbolic-regression", "genetic-programming", "automl", "evolutionary-computation", "pytorch", "gpu", "equation-discovery"]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Science/Research",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Typing :: Typed",
]

# torch ES el motor: dependencia base SIN pin de variante CUDA (default = wheel CPU de PyPI).
# La aceleración GPU se selecciona con el extra [cuda] + el índice de PyTorch (ver [tool.uv.*]).
dependencies = [
    "numpy>=1.24",
    "sympy>=1.12",
    "torch>=2.6",
]

[project.optional-dependencies]
cuda      = ["torch>=2.6"]                   # instala desde el índice cu128 vía tool.uv.sources
dashboard = ["pandas>=2.0"]                  # tablas ricas en el dashboard (el core sirve sin esto)
sklearn   = ["scikit-learn>=1.6"]            # compat Pipeline/GridSearchCV/check_estimator
all       = ["evozero[dashboard,sklearn]"]

[project.urls]
Homepage      = "https://github.com/karimtouma/evozero"
Documentation = "https://evozero.readthedocs.io"
Repository    = "https://github.com/karimtouma/evozero"
Changelog     = "https://github.com/karimtouma/evozero/blob/main/CHANGELOG.md"
Issues        = "https://github.com/karimtouma/evozero/issues"

[project.scripts]
evozero = "evozero.cli:main"                 # un solo comando raíz con subcomandos

# ─────────────────────────── Dependency groups (PEP 735, no user-facing) ───────────────────────────
[dependency-groups]
test = ["pytest>=9.1", "pytest-cov>=6.0", "hypothesis>=6.150", "coverage[toml]>=7.15"]
lint = ["ruff>=0.15"]
type = ["pyrefly>=1.1", "mypy>=2.1"]
docs = [
    "sphinx>=8.1",
    "sphinx-autodoc-typehints>=2.5",
    "numpydoc>=1.8",
    "sphinx-gallery>=0.18",
    "pydata-sphinx-theme>=0.16",
]
dev  = ["nox>=2026.4", "pre-commit>=4.6", "towncrier>=24.8", { include-group = "test" }, { include-group = "lint" }, { include-group = "type" }]

# ─────────────────────────── Versionado dinámico (git-tag) ───────────────────────────
[tool.hatch.version]
source = "vcs"                               # deriva la versión del tag vX.Y.Z

[tool.hatch.build.hooks.vcs]
version-file = "src/evozero/_version.py"     # opcional: escribe __version__ para runtime

[tool.hatch.build.targets.wheel]
packages = ["src/evozero"]

# Incluye los assets del dashboard en el wheel
[tool.hatch.build.targets.wheel.force-include]
"src/evozero/dashboard/_static" = "evozero/dashboard/_static"

[tool.hatch.build.targets.sdist]
include = ["src/evozero", "tests", "README.md", "LICENSE", "NOTICE", "CHANGELOG.md"]

# ─────────────────────────── uv: gestor de entorno + índice de torch ───────────────────────────
[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

# En macOS no hay CUDA -> forzar wheel CPU; en Linux/Windows con extra [cuda] -> cu128.
[tool.uv.sources]
torch = [
    { index = "pytorch-cpu",   marker = "sys_platform == 'darwin'" },
    { index = "pytorch-cu128", marker = "sys_platform != 'darwin' and extra == 'cuda'" },
]

# ─────────────────────────── Ruff (lint + format, reemplaza black/isort/flake8) ───────────────────────────
[tool.ruff]
line-length = 100
target-version = "py311"
src = ["src", "tests"]

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "C4", "SIM", "RET", "PTH", "NPY", "PERF", "RUF", "D"]
ignore = ["D105", "D107"]                    # docstrings en __magic__/__init__ opcionales

[tool.ruff.lint.pydocstyle]
convention = "numpy"                         # paridad con PyData/sklearn

[tool.ruff.lint.per-file-ignores]
"tests/**" = ["D", "ANN"]
"examples/**" = ["D", "B018"]

# ─────────────────────────── mypy (segunda opinión; Pyrefly es primario via nox/CI) ───────────────────────────
[tool.mypy]
python_version = "3.11"
strict = true
warn_unused_configs = true
files = ["src/evozero"]
# torch a veces necesita ignorar stubs incompletos:
[[tool.mypy.overrides]]
module = ["torch.*", "sympy.*"]
ignore_missing_imports = true

# ─────────────────────────── Pyrefly (type checker primario) ───────────────────────────
[tool.pyrefly]
project-includes = ["src/evozero"]
# Pyrefly lee esta sección; modo estricto por defecto en 1.x

# ─────────────────────────── pytest ───────────────────────────
[tool.pytest.ini_options]
minversion = "9.1"
testpaths = ["tests"]
addopts = "-ra --strict-markers --strict-config"
markers = [
    "gpu: requires a CUDA-capable device (skipped automatically if unavailable)",
    "slow: long-running (excluded from default CI run)",
]
filterwarnings = ["error"]

# ─────────────────────────── coverage.py ───────────────────────────
[tool.coverage.run]
branch = true
source = ["evozero"]
core = "sysmon"                              # acelera en Python 3.12+
omit = ["*/_version.py"]

[tool.coverage.report]
exclude_also = ["if TYPE_CHECKING:", "raise NotImplementedError", "@overload"]
skip_covered = false

# ─────────────────────────── towncrier (CHANGELOG) ───────────────────────────
[tool.towncrier]
package = "evozero"
directory = "changelog.d"
filename = "CHANGELOG.md"
start_string = "<!-- towncrier release notes start -->\n"
title_format = "## [{version}](https://github.com/karimtouma/evozero/tree/v{version}) - {project_date}"
type = [
    { name = "Added",      directory = "feature",    showcontent = true },
    { name = "Changed",    directory = "change",     showcontent = true },
    { name = "Deprecated", directory = "deprecated", showcontent = true },
    { name = "Removed",    directory = "removed",    showcontent = true },
    { name = "Fixed",      directory = "fix",        showcontent = true },
    { name = "Security",   directory = "security",   showcontent = true },
]
```

**Instalación resultante (documéntala en README):**
```bash
uv sync --extra cuda           # GPU H100 (cu128)
uv sync                        # CPU-only (macOS / CI)  -> wheel CPU de torch
pip install evozero            # usuario final CPU
pip install "evozero[cuda]" --index-url https://download.pytorch.org/whl/cu128  # usuario final GPU con pip
```

---

## 5. Diseño de API pública

### `__init__.py` (lazy, PEP 562 — `import evozero` nunca dispara `import torch`)

```python
# src/evozero/__init__.py
"""evozero: GPU-native evolutionary computation."""
from __future__ import annotations
import importlib
from typing import TYPE_CHECKING

try:
    from ._version import __version__
except ImportError:
    __version__ = "0.0.0+unknown"

__all__ = [
    "SymbolicRegressor", "Equation",
    "LearnerSearch", "EvolvedLearner", "Task",
    "launch_dashboard", "DashboardHandle",
    "__version__",
]

_LAZY = {
    "SymbolicRegressor": "evozero.sr",
    "Equation":          "evozero.sr",
    "LearnerSearch":     "evozero.automlzero",
    "EvolvedLearner":    "evozero.automlzero",
    "Task":              "evozero.automlzero",
    "launch_dashboard":  "evozero.dashboard",
    "DashboardHandle":   "evozero.dashboard",
}

def __getattr__(name: str):  # PEP 562
    if name in _LAZY:
        mod = importlib.import_module(_LAZY[name])
        return getattr(mod, name)
    raise AttributeError(f"module 'evozero' has no attribute {name!r}")

def __dir__() -> list[str]:
    return sorted(__all__)

if TYPE_CHECKING:  # que los type checkers vean los símbolos reales
    from .sr import Equation, SymbolicRegressor
    from .automlzero import EvolvedLearner, LearnerSearch, Task
    from .dashboard import DashboardHandle, launch_dashboard
```

### Superficie pública por motor

**Motor 1 — `evozero.SymbolicRegressor`** (contrato sklearn: `fit` retorna `self`, sin lógica en `__init__`, atributos aprendidos con sufijo `_`):

```python
class SymbolicRegressor:
    def __init__(self, *,
        population_size: int = 1000,
        n_islands: int = 8,
        migration_interval: int = 25,
        migration_fraction: float = 0.1,
        generations: int = 200,
        max_time: float | None = None,
        binary_operators: tuple[str, ...] = ("+", "-", "*", "/"),
        unary_operators: tuple[str, ...] = ("sin", "cos", "exp", "log"),
        max_size: int = 30,
        max_depth: int | None = None,
        parsimony_coefficient: float = 1e-3,
        constant_optimizer: str = "lbfgs",        # "lbfgs" | "adam" | "none"
        linear_scaling: bool = True,
        dimensional_constraints: dict | None = None,
        restart_patience: int | None = None,
        model_selection: str = "best",            # "best" | "accuracy" | "score"
        device: str = "auto",                     # "auto"|"cpu"|"cuda"|"cuda:0"|"mps"
        deterministic: bool = False,
        random_state: int | None = None,
        n_jobs: int = 1,
        verbose: int = 0,
    ) -> None: ...

    def fit(self, X, y, sample_weight=None) -> "SymbolicRegressor": ...
    def predict(self, X, index: int | None = None): ...
    def score(self, X, y, sample_weight=None) -> float: ...
    def get_params(self, deep: bool = True) -> dict: ...
    def set_params(self, **params) -> "SymbolicRegressor": ...

    # Atributos ajustados (trailing underscore)
    pareto_front_: "pandas.DataFrame"    # columns: complexity, loss, equation, sympy_expr
    equations_: "pandas.DataFrame"       # hall-of-fame / historial
    best_equation_: "Equation"
    run_details_: dict                   # métricas por generación -> alimenta el dashboard
    n_features_in_: int
    feature_names_in_: "np.ndarray"
    device_: "torch.device"

    # Exportadores
    def to_sympy(self, index: int | None = None): ...
    def to_latex(self, index: int | None = None, precision: int = 3) -> str: ...
    def to_numpy_func(self, index: int | None = None) -> "Callable": ...
    def to_torch_module(self, index: int | None = None) -> "torch.nn.Module": ...
```

**Motor 3 — `evozero.LearnerSearch` / `EvolvedLearner`:**

```python
class LearnerSearch:
    def __init__(self, *,
        population_size: int = 1000,
        n_islands: int = 10,
        tournament_size: int = 10,
        setup_ops: list[str] | None = None,
        predict_ops: list[str] | None = None,
        learn_ops: list[str] | None = None,
        max_program_size: int = 100,
        n_meta_generations: int = 1000,
        device: str = "auto",
        random_state: int | None = None,
        verbose: int = 0,
    ) -> None: ...
    def fit(self, tasks: "list[Task]") -> "LearnerSearch": ...
    best_program_: "EvolvedLearner"
    fitness_history_: "pandas.DataFrame"
    def export_learner(self, index: int | None = None) -> "EvolvedLearner": ...

class EvolvedLearner:                     # estimador reutilizable fit/predict
    def fit(self, X, y) -> "EvolvedLearner": ...
    def predict(self, X): ...
    def to_python_source(self) -> str: ...

class Task:                               # una tarea de meta-entrenamiento
    def __init__(self, X, y, *, loss: str = "mse") -> None: ...
```

**Motor 2 — dashboard:**

```python
def launch_dashboard(model, *, host: str = "127.0.0.1", port: int = 8080,
                     open_browser: bool = True, blocking: bool = False) -> "DashboardHandle": ...

class DashboardHandle:
    url: str
    def stop(self) -> None: ...
    def __enter__(self) -> "DashboardHandle": ...
    def __exit__(self, *exc) -> None: ...
```

### Ejemplos de uso reales

**(A) Regresión simbólica + export:**
```python
import numpy as np
from evozero import SymbolicRegressor

X = np.random.randn(500, 3)
y = X[:, 0] ** 2 + np.sin(X[:, 1]) - 0.5

model = SymbolicRegressor(
    n_islands=8, population_size=2000, generations=500,
    binary_operators=("+", "-", "*", "/"), unary_operators=("sin", "cos", "exp"),
    linear_scaling=True, device="auto", random_state=42, verbose=1,
)
model.fit(X, y)

print(model.best_equation_)          # ((x0^2) + sin(x1)) - 0.5
print(model.to_latex())              # x_{0}^{2} + \sin(x_{1}) - 0.5
f = model.to_numpy_func()            # callable NumPy puro (sin torch)
print(model.pareto_front_)           # complexity vs loss
```

**(B) Dashboard en vivo durante el fit:**
```python
from evozero import SymbolicRegressor
from evozero.dashboard import launch_dashboard

model = SymbolicRegressor(population_size=5000, generations=1000, device="auto")
with launch_dashboard(model, port=8080) as dash:   # abre http://127.0.0.1:8080
    print("Dashboard:", dash.url)
    model.fit(X, y)                                 # métricas se transmiten en vivo
# el dashboard se detiene al salir del with
```
CLI equivalente para revisar un run guardado:
```bash
evozero dashboard --run-dir ./runs/latest --port 8080
```

**(C) AutoML-Zero:**
```python
from evozero import LearnerSearch, Task

tasks = [Task(Xi, yi, loss="mse") for Xi, yi in my_meta_training_data]

search = LearnerSearch(population_size=1000, n_islands=10,
                       n_meta_generations=2000, device="auto", random_state=0)
search.fit(tasks)

learner = search.export_learner()      # EvolvedLearner descubierto
learner.fit(X_train, y_train)
preds = learner.predict(X_test)
print(learner.to_python_source())      # el algoritmo de aprendizaje como código Python
```

---

## 6. CI/CD

### `.github/workflows/ci.yml` (test matrix CPU + lint + type)

```yaml
name: CI
on:
  push: { branches: [main] }
  pull_request:
concurrency:
  group: ci-${{ github.ref }}
  cancel-in-progress: true

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
        with: { enable-cache: true }
      - run: uv run ruff check .
      - run: uv run ruff format --check .

  type:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
      - run: uv sync --group type
      - run: uv run pyrefly check          # primario
      - run: uv run mypy                   # segunda opinión (PEP 561)

  test:
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        python-version: ["3.11", "3.12", "3.13"]
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4
        with: { fetch-depth: 0 }           # hatch-vcs necesita el historial de tags
      - uses: astral-sh/setup-uv@v5
        with: { python-version: ${{ matrix.python-version }} }
      - run: uv sync --group test          # torch CPU (sin extra cuda)
      - run: uv run pytest -m "not gpu" --cov=evozero --cov-branch --cov-report=xml
      - uses: codecov/codecov-action@v5
        with: { files: ./coverage.xml }
        env: { CODECOV_TOKEN: ${{ secrets.CODECOV_TOKEN }} }
```

### `.github/workflows/nightly-gpu.yml` (self-hosted, no bloqueante)

```yaml
name: Nightly GPU
on:
  schedule: [{ cron: "0 6 * * *" }]
  workflow_dispatch:
jobs:
  gpu-tests:
    runs-on: [self-hosted, gpu]            # runner propio con H100
    steps:
      - uses: actions/checkout@v4
        with: { fetch-depth: 0 }
      - uses: astral-sh/setup-uv@v5
      - run: uv sync --extra cuda --group test
      - run: uv run pytest -m gpu          # solo la ruta CUDA real
```

**Cómo se prueba el código GPU sin GPU (clave del diseño):**
1. El intérprete tensorizado tiene un modo `device="cpu"` **numéricamente idéntico** (mismas ops torch, solo `.to("cpu")`). ~95% de los tests de correctitud (encuentra la ecuación, Pareto correcto, migración de islas, VM de AutoML-Zero) corren en CPU en runners estándar de GitHub.
2. Marker `gpu` registrado en `pyproject.toml` + skip automático en `conftest.py`:

```python
# tests/conftest.py
import pytest
def _cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False

def pytest_collection_modifyitems(config, items):
    if _cuda():
        return
    skip = pytest.mark.skip(reason="no CUDA device; run in nightly-gpu")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)
```
3. GitHub-hosted runners **no tienen GPU** (sin planes de cambio en 2026). El % de Codecov es de la ruta CPU — documéntalo en README para no engañar sobre cobertura de kernels GPU.

---

## 7. Runbook de release (paso a paso)

### Setup único (una sola vez)
1. **PyPI Trusted Publishers** (sin tokens): en `pypi.org` → *Publishing* → añade dos *pending publishers* al mismo repo `karimtouma/evozero`, workflow `release.yml`:
   - environment `pypi` (con required reviewers).
   - environment `testpypi` en `test.pypi.org`.
2. En GitHub: crea los *Environments* `pypi` (protection rule: required reviewer = tú) y `testpypi`.
3. Elimina cualquier secret `PYPI_API_TOKEN` que exista.

### `.github/workflows/release.yml`

```yaml
name: Release
on:
  push:
    tags: ["v*.*.*"]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with: { fetch-depth: 0 }           # tags completos para hatch-vcs
      - uses: astral-sh/setup-uv@v5
      - run: uv build                       # sdist + wheel en dist/
      - run: uvx twine check dist/*
      - uses: actions/upload-artifact@v4
        with: { name: dist, path: dist/ }

  publish-testpypi:
    needs: build
    runs-on: ubuntu-latest
    environment: testpypi
    permissions: { id-token: write }        # OIDC, solo aquí
    steps:
      - uses: actions/download-artifact@v4
        with: { name: dist, path: dist/ }
      - uses: pypa/gh-action-pypi-publish@release/v1
        with:
          repository-url: https://test.pypi.org/legacy/
          # attestations PEP 740 = ON por defecto (Sigstore)

  smoke-test:
    needs: publish-testpypi
    runs-on: ubuntu-latest
    steps:
      - uses: actions/setup-python@v5
        with: { python-version: "3.12" }
      - run: |
          sleep 30
          pip install --index-url https://test.pypi.org/simple/ \
            --extra-index-url https://pypi.org/simple/ evozero
          python -c "import evozero; print(evozero.__version__)"

  publish-pypi:
    needs: smoke-test
    runs-on: ubuntu-latest
    environment: pypi                       # gated: required reviewer
    permissions: { id-token: write }
    steps:
      - uses: actions/download-artifact@v4
        with: { name: dist, path: dist/ }
      - uses: pypa/gh-action-pypi-publish@release/v1
```

### Flujo de release manual (lo que ejecutas tú)
```bash
# 1. Asegura CI verde en main.
# 2. Compila el changelog desde los news fragments:
uv run towncrier build --version 0.2.0
git add CHANGELOG.md changelog.d/
git commit -m "chore: release 0.2.0"

# 3. Tag SemVer (MAJOR.MINOR.PATCH):
git tag -a v0.2.0 -m "evozero 0.2.0"
git push origin main --tags        # dispara release.yml

# 4. release.yml: build -> TestPyPI -> smoke-install -> (espera tu aprobación) -> PyPI
# 5. Aprueba el environment `pypi` en la pestaña Actions cuando smoke-test pase.
# 6. Crea el GitHub Release desde el tag con las notas del CHANGELOG.
```

**Reglas SemVer para SDK científico:** romper la API pública (`fit`/`predict`/nombres de params/atributos `_`) = **MAJOR**. Nueva funcionalidad retro-compatible = MINOR. Fix = PATCH. Investigadores fijan versiones — sé estricto.

**Changelog:** Conventional Commits para mensajes + towncrier (`changelog.d/123.feature.md`) para el CHANGELOG.md orientado a usuario (evita que refactors internos ensucien el changelog público). Formato Keep a Changelog.

---

## 8. Aislar la dependencia de torch

### 8a. Lazy import (patrón PySR + HuggingFace)

```python
# src/evozero/_import_utils.py
from __future__ import annotations
import importlib.util
from functools import lru_cache

HAS_TORCH = importlib.util.find_spec("torch") is not None   # costo cero, no importa nada

@lru_cache(maxsize=1)
def is_torch_available() -> bool:
    return HAS_TORCH

def require_torch() -> None:
    if not HAS_TORCH:
        raise ImportError(
            "This feature needs PyTorch. Install with:\n"
            "  pip install 'evozero[cuda]'  (GPU)  or  pip install torch  (CPU)"
        )
```

- `import evozero` → nunca importa torch (gracias al `__getattr__` de §5).
- Utilidades de export (LaTeX/NumPy) y carga de modelos funcionan **sin torch** en una máquina sin GPU.
- `torch` se importa por primera vez solo al construir un `SymbolicRegressor()` o llamar `.fit()`.

### 8b. Detección de device + fallback CPU

```python
# src/evozero/_device.py
from __future__ import annotations
import logging
from ._import_utils import require_torch

_logger = logging.getLogger("evozero")
_warned = False

def resolve_device(requested: str = "auto") -> "torch.device":
    require_torch()
    import torch
    if requested != "auto":
        return torch.device(requested)
    # API unificada torch.accelerator (2.6+); puede lanzar RuntimeError sin accelerator
    try:
        if torch.accelerator.is_available():          # cuda / xpu / mps
            return torch.device(torch.accelerator.current_accelerator())
    except (AttributeError, RuntimeError):
        pass
    if torch.cuda.is_available():
        return torch.device("cuda")
    global _warned
    if not _warned:
        _logger.warning("No GPU accelerator found; evozero falls back to CPU.")
        _warned = True                                # un solo warning, no por cada fit()
    return torch.device("cpu")
```
`device="auto"` como default en toda la API; se resuelve una vez en `fit()` y se expone como `device_` (nunca sobrescribe el parámetro de entrada). **No** uses `torch.set_default_device()` global (efectos colaterales entre instancias).

### 8c. Empaquetar el HTML del dashboard con `importlib.resources`

```python
# src/evozero/dashboard/server.py
from __future__ import annotations
import importlib.resources
from http.server import BaseHTTPRequestHandler

def _index_html() -> bytes:
    # Funciona igual en wheel, zip o editable-install. NUNCA rutas relativas a __file__.
    return (importlib.resources.files("evozero.dashboard")
            .joinpath("_static/index.html").read_bytes())

class DashboardHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            body = _index_html()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/api/metrics":
            ...  # sirve run_details_ como JSON (json de stdlib)
```
- Sirve un **único `index.html` "vulcanizado"** (JS/CSS inline, sin `node_modules`, sin build step) como TensorBoard → servible con `http.server` puro.
- Inclusión en el wheel garantizada por `[tool.hatch.build.targets.wheel.force-include]` (§4).
- El dashboard es **solo stdlib** (`http.server`, `threading`, `json`, `importlib.resources`) — cero deps nuevas.

---

## 9. Checklist final de "alto estándar" (2026)

**Empaquetado / tipos**
- [ ] `src/` layout + `py.typed` en `src/evozero/` (PEP 561)
- [ ] Classifier `"Typing :: Typed"` en pyproject.toml
- [ ] `license = "Apache-2.0"` + `license-files` (PEP 639 SPDX, **no** dict legacy, **no** clasificador Trove de licencia)
- [ ] Versionado dinámico por git-tag (hatch-vcs), `uv.lock` committeado
- [ ] `import evozero` funciona **sin torch instalado** (lazy `__getattr__`)
- [ ] torch aislado a `core/`; dashboard solo stdlib

**Calidad**
- [ ] Ruff (lint + format) — badge oficial vía shields.io endpoint
- [ ] Pyrefly `--strict` primario + `mypy --strict` segunda opinión en CI
- [ ] `pytest` + `pytest-cov` + `hypothesis` (property tests: round-trip árbol→LaTeX/NumPy, invariantes CPU/GPU del VM postfix, validez de mutaciones en la VM de AutoML-Zero)
- [ ] `check_estimator(SymbolicRegressor())` en tests, condicionado a sklearn instalado
- [ ] Codecov con `codecov.yml` de "caída máx 5%" (no umbral absoluto)
- [ ] pre-commit 4.6 (ver abajo)
- [ ] Nox con sesiones `lint`, `typecheck`, `tests` (parametrizada por Python), `tests_gpu` (skip graceful), `docs`

**Docs / comunidad**
- [ ] Docstrings estilo **NumPy** (numpydoc) con bloque `Examples` doctest-ejecutable en cada método público; validar con `numpydoc_validation` + `pytest --doctest-modules`
- [ ] **Sphinx** + autodoc + napoleon + sphinx-autodoc-typehints + intersphinx (numpy/torch/sklearn) + sphinx-gallery, publicado en **Read the Docs** (versionado nativo por tags). *No* MkDocs-Material (entró en modo mantenimiento nov-2025) ni Zensical (aún alpha).
- [ ] `examples/` estilo sphinx-gallery (uno por motor), 100% CPU/offline en el build de RTD; demos GPU pesadas con badge "Open in Colab"
- [ ] README con badges en tabla (PyPI version+downloads, CI, RTD, Codecov, ruff, Apache-2.0, Python 3.11–3.13, py.typed) + quickstart por motor + GIF del dashboard
- [ ] `CITATION.cff` desde el primer release; repo conectado a **Zenodo** antes de `v0.1.0` (concept DOI + DOI por versión)
- [ ] `CONTRIBUTING.md` (setup con GPU opcional, cómo correr tests sin GPU, pre-commit/ruff/pyrefly/nox)
- [ ] `CODE_OF_CONDUCT.md` (Contributor Covenant v2.1)
- [ ] `SECURITY.md` — tabla de versiones soportadas + reporte vía GitHub Security Advisories; **menciona explícitamente el modelo de amenaza de `eval` de expresiones simbólicas** (ejecutas código evolucionado)
- [ ] Issue templates YAML (`bug_report.yml`, `feature_request.yml`) + `PULL_REQUEST_TEMPLATE.md` con checklist
- [ ] (Opcional pero recomendado) someter a **JOSS** — requiere ≥6 meses de historial público y, criterio 2026, evidenciar decisiones de diseño humanas (islas+migración, linear scaling, VM tipada de AutoML-Zero) frente a código generado por IA

**Release**
- [ ] PyPI **Trusted Publishing (OIDC)**, jobs `build`/`publish` separados, `id-token: write` solo en publish
- [ ] `gh-action-pypi-publish@release/v1` (attestations PEP 740/Sigstore ON por defecto)
- [ ] Flujo tag → TestPyPI → smoke-install → PyPI (gated por reviewer)
- [ ] SemVer estricto + towncrier + GitHub Release

### `.pre-commit-config.yaml`
```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.15.20
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v5.0.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-toml
      - id: check-added-large-files          # evita checkpoints/datasets accidentales
# Nota: pyrefly/mypy NO como hook pre-commit (overhead alto); se corren en nox/CI.
```

### `CITATION.cff` (mínimo)
```yaml
cff-version: 1.2.0
message: "If you use evozero, please cite it as below."
title: "evozero: GPU-native evolutionary computation for symbolic regression and AutoML-Zero"
authors:
  - family-names: Touma
    given-names: Karim
version: 0.1.0
doi: 10.5281/zenodo.XXXXXXX        # placeholder hasta el primer archivado Zenodo
date-released: 2026-07-04
license: Apache-2.0
repository-code: "https://github.com/karimtouma/evozero"
```

---

## Resumen ejecutivo de decisiones (lo que puedes ejecutar hoy)

| Ítem | Decisión | Por qué |
|---|---|---|
| Nombre | **evozero** | Único con cero colisión PyPI+GitHub (verificado) |
| Licencia | **Apache-2.0** | Grant de patentes + estándar ML/GPU |
| Layout | src/ + py.typed | packaging.python.org, PEP 561 |
| Build backend | **hatchling + hatch-vcs** | Versionado por git-tag (uv_build no lo soporta aún) |
| Gestor/lock | **uv** + uv.lock | Ganador 2026; índice torch por variante CUDA |
| torch | base dep, lazy import, extra `[cuda]` cu128 | Es el motor; aislado a `core/`, fallback CPU |
| Lint/format | **Ruff** único | Reemplaza black/isort/flake8 |
| Types | **Pyrefly** primario + mypy | Pyrefly estable 1.x, rápido; mypy por ubicuidad |
| Tests | pytest + hypothesis + coverage | Property tests para invariantes CPU/GPU |
| Docs | **Sphinx + RTD** (numpydoc) | MkDocs-Material en modo mantenimiento |
| Release | Trusted Publishing OIDC + PEP 740 | Estándar PyPI 2026, sin tokens |
| CLI | argparse (stdlib) | Coherencia cero-deps con el dashboard |

**Atajo de bootstrap:** puedes generar todo el andamiaje (CI, docs, licencias, CITATION.cff, coverage, ruff/mypy/pre-commit) con `copier copy gh:scientific-python/cookie .` (plantilla activa, calibrada a estándares 2026) y luego injertar tus tres motores en `src/evozero/{core,sr,automlzero,dashboard}/`. Ahorra semanas. Único ajuste manual: cambiar el backend a hatchling+hatch-vcs y añadir el bloque `[tool.uv.index]`/`[tool.uv.sources]` de torch.