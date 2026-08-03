# CandidateJobMLMatching

Scoring how well a candidate fits a role — including roles they never applied to.

A learning system rather than a static predictor: recommendations that are acted on generate outcomes for pairs that would otherwise never have been observed, and those outcomes feed periodic retraining. Success is measured as improvement across successive simulated hiring rounds, against a control that does not learn this way.

## Problem

A candidate applies to one role. If they are rejected, that is usually the end of the story, even when they would have been a strong fit for a different opening.

The cause is structural. Recruiters are specialised: someone who has spent two years hiring content managers does not necessarily know which signals predict success in a finance or security role. Cross-role potential is therefore rarely evaluated — not because it does not exist, but because no single person is positioned to see it.

The aim is to score candidate–role fit independently of where the candidate happened to apply.

## Prior work

This problem has a name in the literature — **person–job fit** — and a dedicated venue, the [RecSys in HR](https://recsyshr.aau.dk/) workshop series at ACM RecSys.

The closest work is **[ConFit](https://arxiv.org/abs/2401.16349)** (2024): the same architecture applied to resume–job matching, reporting up to 19% / 31% absolute nDCG@10 over BM25 and OpenAI `text-ada-002`. Two follow-ups exist ([v2](https://arxiv.org/abs/2502.12361), [v3](https://arxiv.org/abs/2605.09760)); the notable point is that v3 abandons pure embedding retrieval because it lacks *controllability and explainability* — the same conclusion this project reaches independently, and the reason the design below scores pairs directly rather than relying on vector similarity.

## Where this differs

**The sparsity assumption does not hold here.** ConFit's stated motivation is that *"job seekers apply only to a few jobs, [so] interaction records in resume-job datasets are sparse."* Most of that line of work — contrastive augmentation, hypothetical resume generation, hard-negative mining — is machinery for coping with too few labelled pairs.

This project operates on roughly 2M applications across a stable set of ~50 roles: on the order of tens of thousands of labelled outcomes per role. The techniques designed to manufacture supervision are largely unnecessary, and effort moves instead to what the abundant supervision can support.

**Labels are graded, not binary.** Most prior work supervises on matched / not-matched. Funnel depth is ordinal, and ranking objectives are built to exploit exactly that.

**Outcomes, not clicks.** Much of the recommender literature supervises on clicks or applications, which measure *preference*. Interview and offer outcomes are closer to suitability.

**The target is counterfactual.** Prior work scores candidates against roles they applied to. The question here is about roles they did not.

## Approach

With ~50 stable roles, there is no retrieval problem. Fifty job representations are a few hundred kilobytes; comparing one candidate against all of them is microseconds of arithmetic. The two-stage *retrieve-then-rerank* architecture exists to avoid running an expensive model over millions of items — a constraint that simply does not apply.

So the expensive model runs on every pair:

```
candidate -> score against all ~50 roles -> ranked roles, with reasons
```

Nothing is discarded by an approximation step, so nothing can be silently lost. This matters more than it first appears: approximate indexes fail hardest in sparse regions of the space, which is precisely where unusual cross-domain candidates sit — the exact population this project exists to surface.

Per role `j`, learn

```
f_j(candidate features) -> predicted funnel depth
```

trained on structured features (education, grades, work history, competitions, projects, languages, test scores) rather than raw text similarity. A learned model can discover predictors that appear nowhere in the job description — if competitive-programming background predicts success in a role whose text never mentions it, similarity is structurally blind to that and supervision is not.

Roles share structure, so a single multi-task model with role as an input is preferable to 50 independent ones — the [category-aware mixture-of-experts](https://arxiv.org/abs/2604.21264) used for person–job fit at ACL Industry 2026 reports +19.4% CTCVR in a live A/B test.

Semantic similarity over job descriptions is retained only as a **baseline to beat**. A learned model must justify itself against it.

## Labels

Funnel depth as graded relevance:

| Depth reached | Grade |
|---|---|
| Application only | 0 |
| Passed screening | 1 |
| Reached interview | 2 |
| Reached final stages | 3 |
| Offer | 4 |

Competing outcomes — withdrawal, position closure, visa or location constraints — are labelled distinctly rather than folded into the negative class, since none of them is evidence of poor fit.

Unobserved pairs are not negatives. Surfacing exactly those pairs is the point of the system, so negatives are constructed deliberately from observed low-depth outcomes plus mined hard negatives.

## The central problem: selection bias

`f_finance` is trained on people who *applied to finance*. It will be used to score someone who applied to content management. Those populations differ systematically — background, self-assessment, reasons for choosing — so the model is being applied outside the distribution it was fit on. This is **covariate shift under self-selection**, and it is the crux of the project.

A second bias compounds it: funnel depth was produced by the current recruiters and screeners. Fit naively, the model learns to reproduce current behaviour, blind spots included. The canonical failure is Amazon's recruiting tool, scrapped in 2018 after it learned to downrank women from historical hiring data.

There is an established framework for this. **Unbiased learning to rank** treats the analogous problem in search, where outcomes are observed only for items the system chose to display. The [SIGIR 2023 tutorial](https://arxiv.org/abs/2305.02914) is the entry point and covers the link to fairness.

Applied to matching specifically: [CFRR](https://arxiv.org/abs/2508.01867) (KDD 2025) uses inverse propensity scoring for two-sided matching including talent platforms, reporting **+51% long-tail coverage** and **−24% Gini exposure inequality**. Long-tail coverage is a direct measure of the "candidate nobody looks at" problem.

**Volume does not dissolve this.** Sparsity here is `applications / (candidates × roles)`, which reduces to *applications per candidate divided by number of roles*. At ~1.4 applications per candidate across 50 roles that is 2.8% of the grid observed — and the figure is invariant to dataset size, since additional candidates add rows and grid cells at the same rate. One quarter and three years of history give the same 2.8%.

Scale therefore fixes the statistical problem (tens of thousands of examples per role is ample) and leaves the structural one untouched. Each candidate is still observed in one or two of fifty roles, and the physics-graduate-in-finance cell is still empty at three million rows. The data is not too small; it is incomplete in a specific direction — dense where people chose to apply, empty everywhere else.

Confirming the true applications-per-candidate ratio is the first query to run. If candidates in fact apply to five roles rather than 1.4, sparsity is 10% and the problem is materially easier.

Candidates who applied to more than one role are especially valuable: they give directly observed cross-role outcomes for the same person, and form a natural validation set for the counterfactual claim. Establishing how many exist is an early task.

## Evaluation

| Metric | Question |
|---|---|
| `nDCG@10` | are the best-fitting roles ranked at the top? |
| `MRR` | how far down is the first good role? |
| Long-tail coverage | are unusual candidates surfaced, or only conventional ones? |

Splits are **grouped by candidate** (no one appears in both train and test) and **ordered in time** (train on the past, test on the future). Results are sliced by seniority, function, and profile conventionality, since aggregate numbers conceal exactly the failures that matter.

The decisive test is not aggregate ranking quality but whether the system surfaces something a recruiter missed: take candidates rejected for role A, ask whether the model would have flagged them for role B, and check whether comparable profiles were later hired into role B.

## Allocation layer

Ranking produces a list per candidate. Allocation is a different question, and answering the first does not answer the second: roles have headcount, candidates can realistically enter only one or two processes, interviewers have finite hours, and locally optimal choices need not be globally optimal.

The gap is arithmetic, not rhetorical. Two candidates, two roles:

|  | Finance | Ops |
|---|---|---|
| A | 0.90 | 0.85 |
| B | 0.88 | 0.30 |

Giving each their own best role sends A to Finance and leaves B with Ops, totalling 1.20. Allocating jointly sends B to Finance and A to Ops, totalling 1.73. Nobody erred; per-candidate ranking is simply the wrong question. On synthetic data with 200 candidates and 50 roles at two seats each, independent top-choice ranking recovers 59% of the achievable total fit.

A well-constructed greedy pass recovers 97.7%, so the case for optimisation is not the residual 2.3%. It is that constraints — interviewer load, one process per candidate, reserved slots — have no principled greedy formulation and a native linear-programming one. At ~50 roles the problem solves to **proven optimality** in milliseconds, an advantage of the scale rather than a limitation of it.

Fairness has been formalised in this setting as envy-freeness and Nash social welfare; see [Fair Reciprocal Recommendation in Matching Markets](https://arxiv.org/abs/2409.00720) (RecSys 2024).

**This imposes two requirements on the scoring model.** An optimiser consumes the estimates it is given and will allocate poor ones optimally, producing a confidently wrong answer with a guarantee attached — worse than a visibly wrong one. So scores must be **calibrated** (0.7 means the outcome occurs 70% of the time, not merely that it ranks above 0.6) and must carry **uncertainty**, which the allocation layer needs in order to distinguish a confident low score from an absent one. Neither property is free, and neither is the default. Both must be built in at the modelling stage rather than retrofitted.

## Continual learning

A one-time fit is the wrong object. Roles change, sourcing channels change, the applicant pool changes, and screening standards change. A model fit on last year's outcomes and left alone degrades quietly — its ranking stays plausible while its scores stop meaning what they meant.

So the system is specified as a loop rather than an artefact:

```
score all pairs -> allocate under capacity -> selected pairs run -> outcomes
      ^                                                               |
      +------ retrain, recalibrate, monitor drift <-------------------+
```

Cross-field recommendations that are acted on generate outcomes for pairs that would otherwise never have been observed. Those outcomes are the only genuinely new information in the system, and they enter the training set.

**Three distinct maintenance operations**, routinely conflated:

*Retraining* refits the model on accumulated data. Run on a fixed cadence and additionally on trigger when monitoring fires.

*Recalibration* corrects the mapping from score to probability. This drifts faster than ranking quality does, and it drifts invisibly — a model can preserve a correct ordering while its stated probabilities become badly wrong. Since the allocation layer consumes probabilities rather than ranks, this is the failure that silently corrupts allocation.

*Drift monitoring* detects when refitting is needed, distinguishing three cases because they have different remedies:

| Drift | What moved | Example |
|---|---|---|
| Covariate | the input distribution | a new sourcing channel changes who applies |
| Label | the outcome distribution | screening standards tighten, offer rates fall |
| Concept | the input–outcome relationship | the role's actual content changes, so different features predict success |

Covariate drift is often survivable and correctable by reweighting. Concept drift is not — it invalidates what the model learned, and only refitting addresses it. Instrumentation: population stability index on feature distributions, expected calibration error on a rolling holdout, and ranking quality tracked over time.

**The loop is dangerous by construction.** Outcomes are observed only for pairs the system selected, so training data increasingly reflects the model's own prior beliefs. Retraining on it yields greater confidence without greater accuracy, and blind spots harden with each round. No amount of retraining detects this from the inside — the model's apparent performance *improves* on exactly the population it has learned to select.

The failure has been characterised precisely in an adjacent setting. [Runaway Feedback Loops in Predictive Policing](https://arxiv.org/abs/1706.09847) (FAT\* 2018) proves why the loop occurs, and distinguishes **discovered** incidents — found because the algorithm directed attention there — from **reported** incidents arriving independently of it. The finding transfers directly: organically-arriving applications are the reported channel and model-selected placements are the discovered one, and the paper's result is that the reported channel *attenuates* runaway feedback but cannot remove it without deliberate intervention. That is the argument for exploration stated as a theorem rather than an intuition, and it also demonstrates the correction can be made black-box, by changing what is fed in rather than rebuilding the model.

Two mitigations are structural rather than optional. **Propensity must be logged at recommendation time**: if the reason a pair was selected is not recorded when it is selected, the selection cannot be corrected for afterwards, and the information is unrecoverable. And some allocation must go to pairs the model is uncertain about rather than confident about — which is the extension below, and the reason a placeholder for it belongs in the loop from the start.

**Evaluation is over rounds, not over a single split.** The synthetic environment is what makes this measurable: the generator holds true fit for all pairs, so it can reveal the outcome of any pair, including those never selected. Real data cannot answer that query, which is why the round-based simulation is only possible here.

Each round scores the full grid, allocates under capacity, reveals outcomes for selected pairs only, appends them, retrains, recalibrates, and then measures counterfactual nDCG against the complete ground truth. Success is a rising curve across rounds. The control condition is a pure-exploitation policy, which is expected to plateau or decline while appearing to improve on its own selections — and recommendation diversity is tracked alongside, since self-confirmation shows up as narrowing before it shows up as error.

## Academic extension: exploration

Statistical correction extrapolates; it cannot create information that was never collected. Learning whether physics graduates succeed in finance ultimately requires that some of them enter finance processes.

The formulation reserves part of the allocation for pairs of high model uncertainty rather than high expected value:

```
maximise   expected hires + λ · information gained
subject to capacity constraints
```

where `λ` prices the value of learning against the cost of a less immediately productive placement. This connects the allocation layer to active learning and to bandit problems, and it turns the optimisation layer into the mechanism that repairs the learning layer's central weakness rather than a component bolted onto it.

Kept as a research extension deliberately: it is the part that requires an organisation to act on recommendations it is uncertain about, which is an organisational question rather than a technical one. In simulation it needs no such permission, and it is where the interesting result lies — a system with a small exploration budget should overtake pure exploitation after enough rounds, despite performing worse in the first few.

## Output

Not a score — an explainable list:

> These candidates did not pass pipeline X, but show strong signals for pipeline Y, and here is why.

Each recommendation carries its reasons: requirements met, requirements missing, where seniority sits. A recommendation whose justification cannot be stated in a sentence will not be acted on.

## Roadmap

**Core**

1. ~~Synthetic environment with known counterfactual ground truth and planted selection bias~~ ✓
2. ~~Leakage-safe evaluation: grouped by candidate, ordered in time, counterfactual nDCG~~ ✓
3. Similarity baseline — the number to beat
4. Supervised per-role scoring, then a shared multi-task model; calibrated, with uncertainty
5. Propensity modelling and IPS-weighted training; verify the gap closes under planted bias and vanishes in the unbiased control
6. Constrained allocation solved to optimality
7. Round-based loop: periodic retraining, recalibration, drift monitoring, propensity logging
8. Per-recommendation explanations; error analysis by seniority, function, profile conventionality

**On real data** — requires read-only access

9. Candidate features and graded funnel-depth labels; establish the applications-per-candidate ratio and the count of multi-application candidates
10. Cross-role backtest: candidates rejected for role A whom the model flags for role B, checked against comparable profiles later hired into B

**Academic extension**

11. Exploration budget — allocation under uncertainty, measured across simulated hiring rounds against a pure-exploitation control
12. Fairness constraints in allocation: envy-freeness, Nash social welfare

## Reading

Two papers are worth reading before writing code:

1. **[ConFit](https://arxiv.org/abs/2401.16349)** — the problem framed cleanly, with a baseline to compare against. Read the motivation carefully: its central difficulty is label sparsity, which is the one problem this project does not have.
2. **[Unbiased Learning to Rank](https://arxiv.org/abs/2305.02914)** (SIGIR 2023 tutorial) — vocabulary and methods for the selection-bias problem, which is the actual hard part here.

Useful later, not now: [CFRR](https://arxiv.org/abs/2508.01867) for debiasing applied to matching, [Fair Reciprocal Recommendation](https://arxiv.org/abs/2409.00720) for the allocation phase, [PJFNN](https://arxiv.org/abs/1810.04040) and [APJFNN](https://arxiv.org/abs/1812.08947) for where the field started.

For the loop specifically: **[Runaway Feedback Loops in Predictive Policing](https://arxiv.org/abs/1706.09847)** (FAT\* 2018) for why a self-selecting training loop diverges and how to correct it without touching the model, and *Hidden Technical Debt in Machine Learning Systems* (Sculley et al., NeurIPS 2015) for why the maintenance surface — not the model — is where these systems fail in practice.

## Scope

Built as a standalone system: generic candidate, role and outcome tables in, ranked matches out. No proprietary data, code or model artefacts belong in this repository.

A personal research project, developed independently and evaluated entirely in simulation. Should the approach demonstrate a measurable gain over rounds against its controls, it could then be proposed as something to trial on real data — but the burden of evidence sits here first, in an environment where the ground truth is known and the claims are falsifiable.

Employment-related models are consequential. This one is intended to widen the set of candidates a human considers — never to reject anyone automatically.
