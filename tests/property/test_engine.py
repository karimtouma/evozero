"""Property-based tests for the core engine invariants (hypothesis)."""

from __future__ import annotations

import numpy as np
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from evozero.core import _sr_engine as E

_CPU = torch.device("cpu")


def _random_tree(seed: int) -> tuple[E.PrimSet, list, list]:
    rng = np.random.default_rng(seed)
    ps = E.PrimSet(3, E.UNARY_ALL, E.BINARY_ALL, named=["pi", "e"])
    code, const = E.gen_tree(ps, int(rng.integers(2, 6)), "grow", rng)
    return ps, code, const


@settings(max_examples=60, deadline=None)
@given(seed=st.integers(min_value=0, max_value=100_000))
def test_prefix_postfix_preserves_length_and_multiset(seed: int) -> None:
    ps, code, const = _random_tree(seed)
    pc, _pk = E.prefix_to_postfix(code, const, ps.arity)
    assert len(pc) == len(code)  # same number of nodes
    assert sorted(pc) == sorted(code)  # same multiset of tokens


@settings(max_examples=50, deadline=None)
@given(seed=st.integers(min_value=0, max_value=100_000))
def test_subtree_span_is_whole_tree_at_root(seed: int) -> None:
    ps, code, _ = _random_tree(seed)
    assert E.subtree_end(code, 0, ps.arity) == len(code)  # root subtree spans everything


@settings(max_examples=50, deadline=None)
@given(seed=st.integers(min_value=0, max_value=100_000))
def test_simplify_preserves_semantics(seed: int) -> None:
    """Algebraic simplification must not change what a program computes."""
    ps, code, const = _random_tree(seed)
    rng = np.random.default_rng(seed + 1)
    X = torch.from_numpy(rng.uniform(-3, 3, size=(3, 64)).astype(np.float32))

    sc, sk = E.simplify_prefix(code, const, ps)
    co, ko, _ = E.batch_postfix([(code, const)], ps)
    cs, ks, _ = E.batch_postfix([(sc, sk)], ps)
    y0 = E.run_population(co, ko, X, ps, _CPU).numpy()[0]
    y1 = E.run_population(cs, ks, X, ps, _CPU).numpy()[0]

    rel = np.abs(y0 - y1) / np.maximum(1.0, np.abs(y0))
    assert float(rel.max()) < 1e-2
