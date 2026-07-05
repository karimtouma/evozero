"""Sphinx configuration for the evozero documentation."""

from __future__ import annotations

import os
import sys

# Import evozero from ./src without installing it (torch is mocked below).
sys.path.insert(0, os.path.abspath("../src"))

# ---------------------------------------------------------------------------
# Project information
# ---------------------------------------------------------------------------
project = "evozero"
author = "Karim Touma"
copyright = "2026, Karim Touma"  # noqa: A001

try:
    from importlib.metadata import version as _version

    release = _version("evozero")
except Exception:  # pragma: no cover - source checkout without install
    release = "0.1.0"
version = ".".join(release.split(".")[:2])

# ---------------------------------------------------------------------------
# General configuration
# ---------------------------------------------------------------------------
extensions = [
    "myst_parser",
    "sphinx_design",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
]

# torch is heavy and GPU-oriented; mock it so the docs build anywhere.
autodoc_mock_imports = ["torch"]
autodoc_typehints = "description"
autodoc_member_order = "bysource"
napoleon_numpy_docstring = True
napoleon_google_docstring = False
always_document_param_types = True

myst_enable_extensions = ["colon_fence", "deflist", "fieldlist"]
myst_heading_anchors = 3

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "blueprint.md"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
}

# ---------------------------------------------------------------------------
# HTML output
# ---------------------------------------------------------------------------
html_theme = "pydata_sphinx_theme"
html_title = f"evozero {version}"
html_theme_options = {
    "github_url": "https://github.com/karimtouma/evozero",
    "icon_links": [
        {"name": "PyPI", "url": "https://pypi.org/project/evozero/", "icon": "fa-solid fa-box"},
    ],
    "use_edit_page_button": False,
    "navbar_align": "left",
    "show_toc_level": 2,
}
html_context = {"default_mode": "auto"}
