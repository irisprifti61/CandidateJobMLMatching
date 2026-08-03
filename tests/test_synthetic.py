"""Checks on the generative model.

These assert the properties the rest of the project depends on: that the funnel
attrites steeply, that observation is sparse, and — most importantly — that the
observed sample really is biased, since a generator that failed to produce bias
would make every later debiasing result vacuous.
"""

from __future__ import annotations

import numpy as np

from cjm.synthetic import SyntheticConfig, generate


def test_shapes_and_ids():
    data = generate(SyntheticConfig(n_candidates=200, n_roles=20, seed=1))

    assert len(data.candidates) == 200
    assert len(data.roles) == 20
    assert len(data.ground_truth) == 200 * 20

    assert data.candidates["candidate_id"].is_unique
    assert data.roles["role_id"].is_unique
    assert len(data.capability_columns) == data.config.n_dims
    assert len(data.requirement_columns) == data.config.n_dims


def test_depth_within_bounds():
    cfg = SyntheticConfig(n_candidates=500, n_roles=20, seed=2)
    data = generate(cfg)

    assert data.applications["depth"].min() >= 0
    assert data.applications["depth"].max() <= cfg.n_funnel_stages


def test_funnel_attrites():
    """Each successive stage should retain a minority of the previous one."""
    data = generate(SyntheticConfig(n_candidates=3000, n_roles=40, seed=3))
    counts = data.applications["depth"].value_counts().sort_index()

    reaching = [(counts[counts.index >= d]).sum() for d in range(5)]
    for shallower, deeper in zip(reaching, reaching[1:]):
        assert deeper < shallower, f"no attrition: {reaching}"

    assert reaching[4] / reaching[0] < 0.15, "final stage is implausibly common"


def test_observation_is_sparse():
    """Only a small share of possible pairs is ever observed."""
    data = generate(SyntheticConfig(n_candidates=1000, n_roles=50, seed=4))
    assert data.observed_rate() < 0.1


def test_applications_are_biased_toward_visible_similarity():
    """The observed sample must be unrepresentative of the population.

    This is the property that makes the dataset worth generating. Without it
    there is no selection bias to correct, and any later comparison between a
    naive and a debiased model would be measuring noise.

    Two things are checked: that applications concentrate on particular roles,
    and — the one that actually matters — that observed pairs have higher true
    fit than the population, so a model trained on them sees a distorted world.
    """
    data = generate(SyntheticConfig(n_candidates=1500, n_roles=40, seed=5))

    truth = data.ground_truth.set_index(["candidate_id", "role_id"])
    applied = data.applications.set_index(["candidate_id", "role_id"])
    selected = truth.loc[applied.index]

    uniform = 1.0 / data.config.n_roles
    assert selected["propensity"].mean() > 3 * uniform, (
        f"applications look near-random: {selected['propensity'].mean():.4f} "
        f"vs uniform {uniform:.4f}"
    )

    fit_shift = selected["true_fit"].mean() - truth["true_fit"].mean()
    assert fit_shift > 0.05, f"observed sample barely differs from population: {fit_shift:+.3f}"


def test_unbiased_control_condition():
    """With sharpness 0, applications are uniform and the sample is unbiased.

    Gives a control condition: any gap between a naive and a debiased model
    should vanish here. If it does not, the gap measured elsewhere was an
    artefact of something other than selection bias.
    """
    cfg = SyntheticConfig(
        n_candidates=1500, n_roles=40, application_sharpness=0.0, seed=6
    )
    data = generate(cfg)

    truth = data.ground_truth.set_index(["candidate_id", "role_id"])
    applied = data.applications.set_index(["candidate_id", "role_id"])
    selected = truth.loc[applied.index]

    assert np.isclose(selected["propensity"].mean(), 1.0 / cfg.n_roles, rtol=0.05)

    fit_shift = selected["true_fit"].mean() - truth["true_fit"].mean()
    assert abs(fit_shift) < 0.03, f"control condition is biased: {fit_shift:+.3f}"


def test_reproducible():
    a = generate(SyntheticConfig(n_candidates=100, n_roles=10, seed=7))
    b = generate(SyntheticConfig(n_candidates=100, n_roles=10, seed=7))
    assert a.applications.equals(b.applications)
