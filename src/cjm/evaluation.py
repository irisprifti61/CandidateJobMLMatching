"""Ranking metrics and leakage-safe splits.

Two things here are easy to get wrong and fatal if you do.

Splitting. A candidate who appears in both train and test lets the model
memorise the person rather than learn what predicts fit, and the resulting
score is meaningless. Splits are therefore grouped by candidate. They are also
ordered in time, because the system would be used to predict future outcomes
from past ones, and a random split quietly grants it knowledge of the future.

What to measure against. Ranking quality on *observed* applications tells you
how well the current process is reproduced, blind spots included — which is
not the objective. `counterfactual_ndcg` scores the ranking against true fit
over all roles, including those never applied to. That is only computable on
synthetic data, and it is the reason synthetic data is worth generating.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "dcg_at_k",
    "ndcg_at_k",
    "reciprocal_rank",
    "candidate_grouped_time_split",
    "counterfactual_ndcg",
]


def dcg_at_k(grades: np.ndarray, k: int) -> float:
    """Discounted cumulative gain of an already-ranked grade sequence.

    Uses the exponential gain 2**g - 1, which is standard for graded relevance
    and makes the jump from "reached interview" to "received offer" count for
    considerably more than the jump from 0 to 1.

    Args:
        grades: Relevance grades in predicted rank order.
        k: Cutoff.

    Returns:
        DCG@k.
    """
    g = np.asarray(grades, dtype=float)[:k]
    if g.size == 0:
        return 0.0
    discounts = np.log2(np.arange(2, g.size + 2))
    return float(((2.0**g - 1.0) / discounts).sum())


def ndcg_at_k(grades: np.ndarray, scores: np.ndarray, k: int = 10) -> float:
    """Normalised DCG for one ranked list.

    Args:
        grades: True relevance grade per item.
        scores: Predicted score per item; higher ranks first.
        k: Cutoff.

    Returns:
        nDCG@k in [0, 1], or 0.0 if no item carries positive relevance.
    """
    grades = np.asarray(grades, dtype=float)
    scores = np.asarray(scores, dtype=float)
    if grades.size == 0:
        return 0.0

    order = np.argsort(-scores, kind="stable")
    ideal = np.sort(grades)[::-1]

    best = dcg_at_k(ideal, k)
    if best == 0.0:
        return 0.0
    return dcg_at_k(grades[order], k) / best


def reciprocal_rank(grades: np.ndarray, scores: np.ndarray, threshold: float = 1.0) -> float:
    """Reciprocal rank of the first item at or above `threshold` relevance.

    Answers "how far down the list before something worth acting on?", which is
    closer to how a recruiter actually consumes a recommendation than nDCG.

    Returns:
        1/rank of the first relevant item, or 0.0 if there is none.
    """
    grades = np.asarray(grades, dtype=float)
    scores = np.asarray(scores, dtype=float)
    order = np.argsort(-scores, kind="stable")
    hits = np.flatnonzero(grades[order] >= threshold)
    return float(1.0 / (hits[0] + 1)) if hits.size else 0.0


def candidate_grouped_time_split(
    applications: pd.DataFrame,
    test_fraction: float = 0.25,
    time_column: str = "applied_at",
    group_column: str = "candidate_id",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split applications by time, without letting a candidate cross the split.

    A cutoff is placed on the time axis, then any candidate with activity on
    both sides is assigned wholly to test and removed from train. Dropping them
    from train rather than test is the conservative direction: it shrinks the
    training set instead of inflating the score.

    Args:
        applications: Table with a time column and a candidate column.
        test_fraction: Approximate share of rows after the cutoff.
        time_column: Name of the timestamp column.
        group_column: Name of the grouping column.

    Returns:
        `(train, test)`, both preserving the original columns.

    Raises:
        ValueError: If `test_fraction` is not strictly between 0 and 1.
    """
    if not 0.0 < test_fraction < 1.0:
        raise ValueError(f"test_fraction must be in (0, 1), got {test_fraction}")

    df = applications.sort_values(time_column, kind="stable")
    cutoff_idx = int(len(df) * (1.0 - test_fraction))
    cutoff_time = df.iloc[cutoff_idx][time_column]

    is_test = df[time_column] >= cutoff_time
    test_groups = set(df.loc[is_test, group_column])

    train = df[~df[group_column].isin(test_groups)].reset_index(drop=True)
    test = df[df[group_column].isin(test_groups)].reset_index(drop=True)
    return train, test


def counterfactual_ndcg(
    scores: pd.DataFrame,
    ground_truth: pd.DataFrame,
    k: int = 10,
    n_grades: int = 4,
) -> float:
    """Mean nDCG@k of predicted role rankings against true fit.

    Scores every role for every candidate, not only the roles they applied to.
    This is the question the project actually asks — would we have surfaced the
    right role for someone who never applied to it — and it is unanswerable on
    real data, where the counterfactual is unobserved.

    True fit is bucketed into integer grades so the metric is comparable with
    nDCG computed over observed funnel depth.

    Args:
        scores: Columns `candidate_id`, `role_id`, `score`.
        ground_truth: Columns `candidate_id`, `role_id`, `true_fit`.
        k: Cutoff.
        n_grades: Number of grade levels, matching the funnel stage count.

    Returns:
        Mean nDCG@k across candidates.
    """
    merged = scores.merge(ground_truth, on=["candidate_id", "role_id"], how="inner")
    if merged.empty:
        return 0.0

    # Equal-width buckets over [0, 1]; grade 0 means "would not clear a stage".
    merged["grade"] = np.floor(merged["true_fit"] * n_grades).clip(0, n_grades - 1)

    per_candidate = []
    for _, group in merged.groupby("candidate_id", sort=False):
        grades = group["grade"].to_numpy()
        # A candidate suited to no open role has an undefined ranking problem,
        # not a failed one. Scoring these zero would penalise the model for
        # cases where every possible ordering is equally correct, and would
        # make the metric track how many such candidates exist rather than how
        # well the model ranks.
        if grades.max() <= 0:
            continue
        per_candidate.append(ndcg_at_k(grades, group["score"].to_numpy(), k))

    return float(np.mean(per_candidate)) if per_candidate else 0.0
