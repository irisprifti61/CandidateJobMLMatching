"""Synthetic recruiting data with a known ground truth.

Real data cannot validate a debiasing method. The counterfactual outcome — how
a candidate would have fared in a role they never applied to — is never
observed, so there is nothing to check a correction against. This module
generates data where that counterfactual is known by construction.

The generative story, and the reason it is set up this way:

Candidates hold latent capability vectors; roles hold requirement vectors.
True fit penalises *shortfalls* against requirements and is largely indifferent
to surplus, because lacking a required skill is disqualifying while holding an
unrequested one is merely harmless. Seniority is treated separately, since
both under- and over-qualification are mismatches.

Candidates do not apply according to true fit. They apply according to
*conventional* signals — the visible part of a profile that makes someone look
like they belong to a field. True fit depends on the whole vector.

That gap is the entire problem. A model fit naively to observed applications
learns the conventional signal and reproduces the blind spots of the process
that produced the data. A candidate whose hidden dimensions suit a role they
would never think to apply for is invisible in the observed sample, and is
exactly who the system is meant to surface.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

__all__ = ["SyntheticConfig", "SyntheticData", "generate"]


@dataclass(frozen=True)
class SyntheticConfig:
    """Parameters of the generative model.

    Attributes:
        n_candidates: Number of candidates.
        n_roles: Number of open roles. Defaults to a realistically small set.
        n_dims: Dimensionality of the capability/requirement space.
        n_required_dims: How many dimensions each role actually demands. Roles
            require a handful of things, not everything; if every dimension
            counted, shortfalls would accumulate across all of them and almost
            no pair would ever clear a stage.
        n_visible_dims: How many leading dimensions drive application choice.
            The remaining dimensions affect fit but not who applies, which is
            what creates the selection bias.
        n_funnel_stages: Maximum funnel depth, so grades run 0..n_funnel_stages.
        stage_decay: Multiplicative difficulty applied per stage, making
            later rounds harder and keeping offers rare.
        mean_applications: Expected applications per candidate.
        shortfall_weight: Penalty per unit of unmet requirement.
        seniority_weight: Penalty per unit of seniority mismatch, applied in
            both directions.
        fit_intercept: Baseline log-odds of passing a single stage.
        application_sharpness: How strongly candidates concentrate on roles
            that look conventional for them. Zero gives uniform (unbiased)
            application, which is useful as a control condition.
        horizon_days: Window over which applications are spread.
        seed: Random seed.
    """

    n_candidates: int = 2000
    n_roles: int = 50
    n_dims: int = 12
    n_required_dims: int = 4
    n_visible_dims: int = 4
    n_funnel_stages: int = 4
    mean_applications: float = 1.4
    stage_decay: float = 0.55
    shortfall_weight: float = 0.7
    seniority_weight: float = 0.6
    fit_intercept: float = 2.6
    application_sharpness: float = 6.0
    horizon_days: int = 540
    seed: int = 0

    #: Requirement level on dimensions a role actually demands.
    requirement_mean: float = 0.5
    requirement_std: float = 0.8
    #: Requirement on irrelevant dimensions, low enough to never bind.
    irrelevant_requirement: float = -5.0


@dataclass
class SyntheticData:
    """A generated dataset.

    Attributes:
        candidates: One row per candidate, with capability features.
        roles: One row per role, with requirement features.
        applications: Observed applications and their funnel outcomes. This is
            the only table a model is allowed to train on.
        ground_truth: True fit for *every* candidate-role pair, including pairs
            never applied to. Evaluation only. Supplying this to a model
            defeats the purpose of the exercise.
        config: The configuration used.
    """

    candidates: pd.DataFrame
    roles: pd.DataFrame
    applications: pd.DataFrame
    ground_truth: pd.DataFrame
    config: SyntheticConfig = field(repr=False)

    @property
    def capability_columns(self) -> list[str]:
        return [c for c in self.candidates.columns if c.startswith("cap_")]

    @property
    def requirement_columns(self) -> list[str]:
        return [c for c in self.roles.columns if c.startswith("req_")]

    def observed_rate(self) -> float:
        """Fraction of all candidate-role pairs that were ever applied to."""
        total = len(self.candidates) * len(self.roles)
        return len(self.applications) / total


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _true_fit(
    capabilities: np.ndarray,
    seniority: np.ndarray,
    requirements: np.ndarray,
    role_seniority: np.ndarray,
    cfg: SyntheticConfig,
) -> np.ndarray:
    """Per-stage pass probability for every candidate-role pair.

    Asymmetric by construction: only shortfalls against requirements are
    penalised, so exceeding a requirement neither helps nor hurts. Seniority is
    penalised in both directions.

    Returns:
        Array of shape (n_candidates, n_roles) with values in (0, 1).
    """
    # (n_candidates, 1, n_dims) against (1, n_roles, n_dims)
    gap = requirements[None, :, :] - capabilities[:, None, :]
    shortfall = np.clip(gap, 0.0, None).sum(axis=2)

    seniority_gap = np.abs(seniority[:, None] - role_seniority[None, :])

    logit = (
        cfg.fit_intercept
        - cfg.shortfall_weight * shortfall
        - cfg.seniority_weight * seniority_gap
    )
    return _sigmoid(logit)


def _application_propensity(
    capabilities: np.ndarray,
    demand_mask: np.ndarray,
    cfg: SyntheticConfig,
) -> np.ndarray:
    """Probability of a candidate choosing each role, given they apply.

    Computed from the *visible* dimensions only: a candidate reads which skills
    a posting asks for and gravitates toward roles demanding things they are
    visibly strong in. Fit, by contrast, depends on all dimensions.

    That asymmetry is the mechanism producing selection bias. A candidate whose
    strength lies in the hidden dimensions of a role never thinks to apply, so
    the pair is absent from the observed sample — and it is exactly the kind of
    pair the system is supposed to surface.

    Returns:
        Row-stochastic array of shape (n_candidates, n_roles).
    """
    v = cfg.n_visible_dims
    affinity = capabilities[:, :v] @ demand_mask[:, :v].T / np.sqrt(v)

    logits = cfg.application_sharpness * affinity
    logits -= logits.max(axis=1, keepdims=True)  # stabilise before exponentiating
    weights = np.exp(logits)
    return weights / weights.sum(axis=1, keepdims=True)


def _sample_depth(
    fit: np.ndarray, n_stages: int, decay: float, rng: np.random.Generator
) -> np.ndarray:
    """Draw funnel depth as a sequence of increasingly demanding stages.

    Stage s is cleared with probability `fit * decay**s`. The decay matters: if
    every stage were equally easy, everyone who cleared the first would clear
    them all, and the final bucket would accumulate a large share of applicants
    instead of the few percent a real funnel yields. Later rounds are harder,
    so attrition compounds and offers stay rare.
    """
    stage_scale = decay ** np.arange(n_stages)
    thresholds = fit[:, None] * stage_scale[None, :]
    passes = rng.random((fit.shape[0], n_stages)) < thresholds
    # depth = length of the leading run of successes
    return np.cumprod(passes, axis=1).sum(axis=1).astype(int)


def generate(config: SyntheticConfig | None = None) -> SyntheticData:
    """Generate a synthetic recruiting dataset.

    Args:
        config: Generative parameters. Defaults to `SyntheticConfig()`.

    Returns:
        A `SyntheticData` bundle. Train only on `applications`; keep
        `ground_truth` for evaluation.
    """
    cfg = config or SyntheticConfig()
    rng = np.random.default_rng(cfg.seed)

    n_c, n_r, n_d = cfg.n_candidates, cfg.n_roles, cfg.n_dims

    capabilities = rng.normal(0.0, 1.0, size=(n_c, n_d))
    seniority = rng.normal(0.0, 1.0, size=n_c)

    # Each role demands only a few dimensions. Dimensions it does not demand
    # get a requirement far below any plausible capability, so they never
    # contribute a shortfall. Concentrating requirements this way keeps fit
    # well spread instead of collapsing toward zero, and matches how job
    # specifications actually read.
    requirements = np.full((n_r, n_d), cfg.irrelevant_requirement)
    demand_mask = np.zeros((n_r, n_d))
    for r in range(n_r):
        demanded = rng.choice(n_d, size=cfg.n_required_dims, replace=False)
        requirements[r, demanded] = rng.normal(
            cfg.requirement_mean, cfg.requirement_std, size=cfg.n_required_dims
        )
        demand_mask[r, demanded] = 1.0
    role_seniority = rng.normal(0.0, 0.8, size=n_r)

    fit = _true_fit(capabilities, seniority, requirements, role_seniority, cfg)
    propensity = _application_propensity(capabilities, demand_mask, cfg)

    # How many roles each candidate applies to.
    n_apps = rng.poisson(cfg.mean_applications, size=n_c)
    n_apps = np.clip(n_apps, 0, n_r)

    cand_idx: list[int] = []
    role_idx: list[int] = []
    for c in range(n_c):
        if n_apps[c] == 0:
            continue
        chosen = rng.choice(n_r, size=n_apps[c], replace=False, p=propensity[c])
        cand_idx.extend([c] * n_apps[c])
        role_idx.extend(chosen.tolist())

    cand_idx_arr = np.asarray(cand_idx, dtype=int)
    role_idx_arr = np.asarray(role_idx, dtype=int)

    pair_fit = fit[cand_idx_arr, role_idx_arr]
    depth = _sample_depth(pair_fit, cfg.n_funnel_stages, cfg.stage_decay, rng)

    day = rng.integers(0, cfg.horizon_days, size=len(cand_idx_arr))
    applied_at = pd.Timestamp("2024-01-01") + pd.to_timedelta(day, unit="D")

    candidates = pd.DataFrame(
        capabilities, columns=[f"cap_{i}" for i in range(n_d)]
    )
    candidates.insert(0, "candidate_id", np.arange(n_c))
    candidates["seniority"] = seniority

    roles = pd.DataFrame(requirements, columns=[f"req_{i}" for i in range(n_d)])
    roles.insert(0, "role_id", np.arange(n_r))
    roles["seniority"] = role_seniority

    applications = pd.DataFrame(
        {
            "candidate_id": cand_idx_arr,
            "role_id": role_idx_arr,
            "applied_at": applied_at,
            "depth": depth,
        }
    ).sort_values("applied_at", ignore_index=True)

    ground_truth = pd.DataFrame(
        {
            "candidate_id": np.repeat(np.arange(n_c), n_r),
            "role_id": np.tile(np.arange(n_r), n_c),
            "true_fit": fit.ravel(),
            "propensity": propensity.ravel(),
        }
    )

    return SyntheticData(
        candidates=candidates,
        roles=roles,
        applications=applications,
        ground_truth=ground_truth,
        config=cfg,
    )
