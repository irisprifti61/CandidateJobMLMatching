# CandidateJobMLMatching

Scoring how well a candidate fits a role — including roles they never applied to.

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

Candidates who applied to more than one role are especially valuable: they give directly observed cross-role outcomes for the same person, and form a natural validation set for the counterfactual claim. Establishing how many exist is an early task.

## Evaluation

| Metric | Question |
|---|---|
| `nDCG@10` | are the best-fitting roles ranked at the top? |
| `MRR` | how far down is the first good role? |
| Long-tail coverage | are unusual candidates surfaced, or only conventional ones? |

Splits are **grouped by candidate** (no one appears in both train and test) and **ordered in time** (train on the past, test on the future). Results are sliced by seniority, function, and profile conventionality, since aggregate numbers conceal exactly the failures that matter.

The decisive test is not aggregate ranking quality but whether the system surfaces something a recruiter missed: take candidates rejected for role A, ask whether the model would have flagged them for role B, and check whether comparable profiles were later hired into role B.

## Assignment layer

Ranking produces a list per candidate. Allocation is a separate question: roles have capacity, candidates are finite, and locally optimal choices need not be globally optimal.

At ~50 roles this is small enough to solve to **proven optimality** rather than heuristically — an advantage of the scale, not a limitation of it. Fairness has been formalised in this setting as envy-freeness and Nash social welfare; see [Fair Reciprocal Recommendation in Matching Markets](https://arxiv.org/abs/2409.00720) (RecSys 2024) when this phase begins.

This layer is a second phase, not a prerequisite for a useful ranking.

## Output

Not a score — an explainable list:

> These candidates did not pass pipeline X, but show strong signals for pipeline Y, and here is why.

Each recommendation carries its reasons: requirements met, requirements missing, where seniority sits. A recommendation whose justification cannot be stated in a sentence will not be acted on.

## Roadmap

1. Assemble candidate features and graded funnel-depth labels; quantify multi-application candidates
2. Similarity baseline — the number to beat
3. Leakage-safe evaluation: grouped by candidate, ordered in time
4. Supervised per-role scoring, then a shared multi-task model
5. Propensity modelling and IPS-weighted training for selection bias
6. Cross-role backtest on multi-application candidates
7. Error analysis by seniority, function, profile conventionality; per-recommendation explanations
8. Optional: constrained assignment with fairness

## Reading

Two papers are worth reading before writing code:

1. **[ConFit](https://arxiv.org/abs/2401.16349)** — the problem framed cleanly, with a baseline to compare against. Read the motivation carefully: its central difficulty is label sparsity, which is the one problem this project does not have.
2. **[Unbiased Learning to Rank](https://arxiv.org/abs/2305.02914)** (SIGIR 2023 tutorial) — vocabulary and methods for the selection-bias problem, which is the actual hard part here.

Useful later, not now: [CFRR](https://arxiv.org/abs/2508.01867) for debiasing applied to matching, [Fair Reciprocal Recommendation](https://arxiv.org/abs/2409.00720) for the allocation phase, [PJFNN](https://arxiv.org/abs/1810.04040) and [APJFNN](https://arxiv.org/abs/1812.08947) for where the field started.

## Scope

Built as a standalone system: generic candidate, role and outcome tables in, ranked matches out. No proprietary data, code or model artefacts belong in this repository.

Employment-related models are consequential. This one is intended to widen the set of candidates a human considers — never to reject anyone automatically.
