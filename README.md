# CandidateJobMLMatching

Ranking job openings by how well a candidate fits them — including roles the candidate never applied to.

## Problem

A candidate applies to one role. If they are rejected, that is usually the end of the story, even when they would have been a strong fit for a different opening.

This happens for a structural reason: recruiters are specialised. Someone who has spent two years hiring content managers does not necessarily know which signals predict success in a finance or security role. Cross-role potential is therefore rarely evaluated, not because it does not exist, but because no single person is positioned to see it.

The goal is to score candidate–job fit independently of where the candidate applied.

## Approach

Represent candidates and jobs separately, then compare them in two stages.

```
CV              -> parsed facets -> candidate embedding
Job description -> parsed facets -> job embedding (precomputed, indexed)

                        candidate embedding
                                |
                    [1] retrieve top ~20 jobs        cheap, runs over all jobs
                                |
                    [2] rerank each pair in detail   expensive, runs on 20
                                |
                            ranked jobs
```

The two stages optimise different things. Retrieval maximises recall at low cost: it must not lose a good role, but it is allowed to pass along mediocre ones. Reranking maximises precision at the top: it examines each shortlisted pair closely and decides the final order.

Candidate features are extracted once, without reference to any target job, so a CV is processed a single time and compared against every opening. Job embeddings are precomputed and refreshed only when a description changes.

## Stage 0 — Parsing into facets

Raw text is parsed into structured facts before anything is embedded. Embedding raw JSON is avoided: punctuation and key names dominate the representation and dilute the signal.

Candidate facets:

| Facet | Example |
|---|---|
| Skills | Python, SQL, distributed systems |
| Experience | 4 years total; 3 backend, 1 data |
| Seniority | mid-level |
| Education | MSc computer science |
| Industries | fintech, SaaS |
| Languages | Italian (native), English (C1) |
| Location | Milan; EU work authorisation |

Job facets mirror them, with the crucial addition of a mandatory/preferred distinction:

| Facet | Example |
|---|---|
| Required skills | Python (mandatory), Kubernetes (preferred) |
| Experience | minimum 3 years backend |
| Seniority | mid to senior |
| Education | STEM degree (preferred) |
| Industry | fintech |
| Languages | English (mandatory) |
| Location | Milan, hybrid; no visa sponsorship |

Each facet is rendered as canonical text. Facets can be embedded jointly into one vector, or separately into per-facet vectors that are compared field by field. Comparing field by field is more faithful to how fit actually works, at the cost of more vectors to store and search.

## Stage 1 — Retrieval

A frozen multilingual sentence encoder produces the initial embeddings — no training required to get a working baseline. Vectors are L2-normalised so inner product equals cosine similarity.

The job index starts as an exact flat index, which is entirely adequate for tens of thousands of jobs and gives an exact-recall reference point. At larger scale it becomes an approximate index (HNSW or IVF-PQ), and the recall lost to approximation is measured against the flat index rather than assumed negligible.

Retrieval returns the top `k` jobs, with `k ≈ 20` as a starting point. `k` is a tunable recall/cost trade-off, and is set by measuring `Recall@k`: the fraction of genuinely good roles that survive into the shortlist. A good role discarded here can never be recovered downstream, which is why this stage is tuned for recall and not precision.

## Stage 2 — Reranking

Semantic similarity alone measures topical resemblance, not fitness. It handles occupation, domain, related work history and education subject reasonably well. It fails systematically on:

- junior versus senior roles — the text is nearly identical
- mandatory versus optional requirements
- years-of-experience thresholds
- location, visa and language constraints
- a technology mentioned in passing versus actual proficiency
- transferable skills expressed in different vocabulary

A candidate with three years of experience and one with fifteen produce almost the same embedding. That is precisely the distinction hiring turns on.

The reranker therefore combines the semantic score with structured checks:

```
score = w1 * semantic_similarity
      + w2 * skill_coverage
      + w3 * seniority_compatibility
      + w4 * experience_compatibility
      - w5 * missing_mandatory_requirements
```

| Term | Definition |
|---|---|
| `semantic_similarity` | cosine between candidate and job facet embeddings |
| `skill_coverage` | fraction of required skills evidenced in the CV, weighted by how central each is to the role |
| `seniority_compatibility` | signed gap between candidate level and target band |
| `experience_compatibility` | years held against the stated threshold |
| `missing_mandatory_requirements` | hard penalty per unmet mandatory requirement |

Two properties matter here. Seniority is **asymmetric**: underqualified and overqualified are both mismatches, but not the same mismatch, and they take different penalty curves. Mandatory requirements are **near-disqualifying** rather than merely costly — no amount of semantic similarity should rescue a candidate who cannot legally take the job.

Weights start hand-set as a transparent baseline, then are learned.

**Model choice.** Two options are viable. A cross-encoder reads the candidate and job text jointly and is the stronger model, but it is expensive and opaque. Gradient-boosted ranking (LambdaMART) over the retrieval score plus structured features is cheaper, natively handles heterogeneous tabular features, directly optimises a ranking objective with graded labels, and yields per-feature attributions.

The plan starts with LambdaMART, because explainability is not a nice-to-have here: a recommendation a recruiter cannot understand is a recommendation they will not act on. The cross-encoder is a later experiment, not the critical path.

## Asymmetry

Candidates and jobs are encoded by separate towers rather than a shared encoder. Fit is not symmetric: a candidate *possesses* capabilities, a job *requires* them. Holding a skill the job never asks for is harmless; lacking one it demands is disqualifying. A single shared representation cannot express that difference.

## Labels

Funnel depth gives **graded relevance**, and there is a large volume of it. How far a candidate progressed — screening, interview, final stages, offer — is an ordinal signal, not a binary one:

| Depth reached | Relevance grade |
|---|---|
| Application only | 0 |
| Passed screening | 1 |
| Reached interview | 2 |
| Reached final stages | 3 |
| Offer | 4 |

This maps directly onto graded ranking metrics and onto LambdaMART, both of which are built for exactly this shape of label. It is a considerably richer training signal than a binary hired/not-hired flag.

Two structural properties shape the design:

**Unobserved pairs are not negatives.** Outcomes exist only where a candidate actually applied. Every other candidate–job pair is unknown — and since surfacing exactly those pairs is the point of the system, they cannot be sampled as negatives. Negatives are constructed deliberately: observed low-depth pairs, plus hard negatives mined from roles that are semantically close but structurally incompatible.

**Depth partly encodes the current policy.** Progression reflects the decisions of existing screeners and recruiters, so a model fit naively to it learns to reproduce current behaviour, blind spots included. This is manageable — it is what the held-out evaluation and the cross-role slices are for — but it bounds what the training signal can prove on its own.

Competing outcomes are separated from poor fit: withdrawal, position closure, and visa or location constraints are labelled distinctly rather than folded into the negative class.

## Evaluation

| Metric | Stage | Question |
|---|---|---|
| `Recall@20` | retrieval | did the shortlist keep the good roles? |
| `NDCG@10` | reranking | are the best roles ranked at the top? |
| `MRR` | reranking | how far down is the first good role? |

Split protocol:

- **Grouped by candidate** — no candidate appears in both train and test, or the model is graded on people it has already seen.
- **Ordered in time** — train on the past, test on the future, matching how the system would actually be used.

Slices reported separately for seniority band, function, and cold-start candidates with sparse CVs, since aggregate numbers hide exactly the failures that matter.

The decisive test is not aggregate ranking quality but whether the system surfaces something a recruiter missed: take candidates rejected for role A, ask whether the model would have flagged them for role B, and check whether comparable profiles were later hired into role B.

## Cost and performance

| Operation | Cost |
|---|---|
| Candidate embedding | ~10–300 ms, depending on hardware and CV length |
| ANN search over 100k–1M jobs | milliseconds |
| Storage, 1M × 768-dim | ~3 GB at float32, ~1.5 GB at float16 |
| Job embedding | one-off, refreshed only on description change |
| LLM parsing | dominates total cost |

Parsing, not embedding or search, is the expensive part — so parsed outputs are cached and keyed by document hash. Everything downstream is cheap enough to be irrelevant to the budget.

## Output

The deliverable is not a score. It is a short, explainable list:

> These candidates did not pass pipeline X, but show strong signals for pipeline Y — and here is why.

Each recommendation carries its reasons: which requirements are met, which are missing, where the seniority sits. If the reason cannot be stated in one sentence, the recommendation is not usable.

## Roadmap

1. Parse CVs and job descriptions into role-neutral facets; cache by document hash
2. Frozen-embedding retrieval baseline over a flat index; establish `Recall@k`
3. Ranking metrics and leakage-safe splits — grouped by candidate, ordered in time
4. Structured compatibility features and a hand-weighted reranker as a transparent baseline
5. LambdaMART reranker trained on graded funnel-depth labels
6. Hard negative mining and ablations over each scoring term
7. Error analysis by seniority, function and cold-start; per-recommendation explanations
8. Optional: cross-encoder reranking, learned facet encoders

## Scope

Built as a standalone system: generic candidate, job and outcome tables in, ranked matches out. No proprietary data, code or model artefacts belong in this repository.

Employment-related models are consequential. This one is intended to widen the set of candidates a human considers — never to reject anyone automatically.
