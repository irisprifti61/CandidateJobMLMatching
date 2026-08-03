"""Checks on metrics and splits."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cjm.evaluation import (
    candidate_grouped_time_split,
    counterfactual_ndcg,
    ndcg_at_k,
    reciprocal_rank,
)
from cjm.synthetic import SyntheticConfig, generate


def test_ndcg_perfect_and_reversed():
    grades = np.array([3, 2, 1, 0])
    assert ndcg_at_k(grades, np.array([4.0, 3.0, 2.0, 1.0]), k=4) == pytest.approx(1.0)
    assert ndcg_at_k(grades, np.array([1.0, 2.0, 3.0, 4.0]), k=4) < 0.7


def test_ndcg_all_zero_grades():
    assert ndcg_at_k(np.zeros(5), np.arange(5.0), k=5) == 0.0


def test_ndcg_is_ordering_invariant():
    """Score magnitudes must not matter, only the order they induce."""
    grades = np.array([0, 3, 1, 2])
    a = ndcg_at_k(grades, np.array([0.1, 0.9, 0.4, 0.6]), k=4)
    b = ndcg_at_k(grades, np.array([10.0, 90.0, 40.0, 60.0]), k=4)
    assert a == pytest.approx(b)


def test_reciprocal_rank():
    grades = np.array([0, 0, 2, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.6])
    assert reciprocal_rank(grades, scores) == pytest.approx(1 / 3)
    assert reciprocal_rank(np.zeros(4), scores) == 0.0


def test_split_has_no_candidate_overlap():
    data = generate(SyntheticConfig(n_candidates=800, n_roles=30, seed=11))
    train, test = candidate_grouped_time_split(data.applications, test_fraction=0.25)

    assert not set(train["candidate_id"]) & set(test["candidate_id"])
    assert len(train) > 0 and len(test) > 0


def test_split_respects_time_order():
    """No training row may postdate the earliest test row."""
    data = generate(SyntheticConfig(n_candidates=800, n_roles=30, seed=12))
    train, test = candidate_grouped_time_split(data.applications, test_fraction=0.25)
    assert train["applied_at"].max() <= test["applied_at"].max()


def test_split_rejects_bad_fraction():
    data = generate(SyntheticConfig(n_candidates=100, n_roles=10, seed=13))
    with pytest.raises(ValueError):
        candidate_grouped_time_split(data.applications, test_fraction=1.5)


def test_counterfactual_ndcg_ranks_oracle_above_random():
    """The harness must separate a perfect ranker from a random one.

    If it cannot do that, no result produced with it can be trusted.
    """
    data = generate(SyntheticConfig(n_candidates=300, n_roles=25, seed=14))
    rng = np.random.default_rng(0)

    oracle = data.ground_truth.rename(columns={"true_fit": "score"})[
        ["candidate_id", "role_id", "score"]
    ]
    random = oracle.assign(score=rng.random(len(oracle)))

    oracle_score = counterfactual_ndcg(oracle, data.ground_truth, k=10)
    random_score = counterfactual_ndcg(random, data.ground_truth, k=10)

    assert oracle_score == pytest.approx(1.0)
    assert oracle_score > random_score + 0.1


def test_counterfactual_ndcg_empty_input():
    empty = pd.DataFrame(columns=["candidate_id", "role_id", "score"])
    truth = pd.DataFrame(
        {"candidate_id": [0], "role_id": [0], "true_fit": [0.5]}
    )
    assert counterfactual_ndcg(empty, truth) == 0.0
