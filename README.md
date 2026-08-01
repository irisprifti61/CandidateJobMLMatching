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
                    [3] allocate across candidates   avoids concentration
                                |
                            ranked jobs
```

Stages 1 and 2 optimise different things. Retrieval maximises recall at low cost: it must not lose a good role, but it is allowed to pass along mediocre ones. Reranking maximises precision at the top: it examines each shortlisted pair closely and decides the final order. Stage 3 is discussed under [Allocation](#allocation).

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

Each facet is rendered as canonical text and given its **own token budget**, so a long employment history cannot crowd out the education or language facets. Facets are delimited by dedicated tokens rather than concatenated into a single blob that gets truncated arbitrarily.

Facets can be embedded jointly into one vector, or separately into per-facet vectors compared field by field. Field-by-field is more faithful to how fit actually works, at the cost of more vectors to store and search.

Parsing is the dominant cost in the pipeline, so two things are worth testing before defaulting to an LLM for everything: caching parsed output keyed by document hash, and using a dedicated skill-extraction encoder instead of a general LLM. The evidence suggests lightweight bi-encoders match LLM extraction quality here at a fraction of the cost and latency.

## Stage 1 — Retrieval

A frozen multilingual sentence encoder produces the initial embeddings — no training required to get a working baseline. Vectors are L2-normalised so inner product equals cosine similarity.

The job index starts as an exact flat index, which is adequate for tens of thousands of jobs and gives an exact-recall reference point. At larger scale it becomes an approximate index (HNSW or IVF-PQ), and the recall lost to approximation is measured against the flat index rather than assumed negligible.

Retrieval returns the top `k` jobs, `k ≈ 20` as a starting point. `k` is a tunable recall/cost trade-off, set by measuring `Recall@k`: the fraction of genuinely good roles that survive into the shortlist. A good role discarded here can never be recovered downstream, which is why this stage is tuned for recall and not precision.

**A register gap sits between the two sides.** A CV says what someone did; a job description says what someone should do. They describe the same underlying competence in different grammar, which costs retrieval recall. One cheap remedy is to generate a *hypothetical CV* from each job description with an LLM and embed that instead of, or alongside, the raw description — moving both sides into the same register before comparison.

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

Two properties matter. Seniority is **asymmetric**: underqualified and overqualified are both mismatches, but not the same one, and they take different penalty curves. Mandatory requirements are **near-disqualifying** rather than merely costly — no amount of semantic similarity should rescue a candidate who cannot legally take the job.

Hard constraints are enforced by rule rather than learned. Production systems in this domain consistently keep an explicit rule layer alongside the model; visa eligibility is not something to hope a gradient discovers.

Weights start hand-set as a transparent baseline, then are learned.

**Model choice.** A cross-encoder reads candidate and job jointly and is the stronger model, but it is expensive and opaque. Gradient-boosted ranking (LambdaMART) over the retrieval score plus structured features is cheaper, natively handles heterogeneous tabular features, directly optimises a ranking objective with graded labels, and yields per-feature attributions.

The plan starts with LambdaMART. Explainability is not a nice-to-have here: a recommendation a recruiter cannot understand is one they will not act on, and — see [Compliance](#compliance) — an individual decision may have to be explained on request. The cross-encoder is a later experiment, not the critical path.

## Asymmetry

Candidates and jobs are encoded by separate towers rather than a shared encoder. Fit is not symmetric: a candidate *possesses* capabilities, a job *requires* them. Holding a skill the job never asks for is harmless; lacking one it demands is disqualifying. A single shared representation cannot express that difference.

## Labels

Funnel depth gives **graded relevance**, and there is a large volume of it. How far a candidate progressed is ordinal, not binary:

| Depth reached | Relevance grade |
|---|---|
| Application only | 0 |
| Passed screening | 1 |
| Reached interview | 2 |
| Reached final stages | 3 |
| Offer | 4 |

This maps directly onto graded ranking metrics and onto LambdaMART, both built for exactly this shape of label — a considerably richer signal than a binary hired/not-hired flag.

Three structural properties shape the design.

**Unobserved pairs are not negatives.** Outcomes exist only where a candidate actually applied. Every other pair is unknown — and since surfacing exactly those pairs is the point of the system, they cannot be sampled as negatives.

Negatives are therefore constructed deliberately. Observed low-depth pairs are genuine negatives. Beyond those, hard negatives are mined from a *percentile band* of the ranking: search the top few percent by similarity but **exclude the very top**, on the reasoning that the highest-scoring unlabelled pairs are disproportionately false negatives — good matches nobody happened to observe. Mining the band just below yields hard negatives without poisoning the training set with the exact cases the system exists to find.

**Depth partly encodes the current policy.** Progression reflects decisions made by existing screeners and recruiters, so a model fit naively to it learns to reproduce current behaviour, blind spots included. This is not hypothetical: a published system trained on real-world matching rules reached `NDCG@20 = 0.706` while agreeing with independent human judgement only 46% of the time. It had learned the rules, not the documents.

The mitigation is a **second evaluation set, hand-judged and independent of funnel outcomes**. Alignment with it is tracked as a separate metric alongside NDCG. A model that improves on funnel depth while drifting away from independent judgement is memorising the old policy, and the two numbers make that visible.

**Depth conflates qualification with preference.** A candidate who was qualified but withdrew looks identical to one who was rejected as unqualified. Withdrawal, position closure, and visa or location constraints are labelled distinctly rather than folded into the negative class.

## Allocation

Ranking each candidate independently produces a predictable failure: every recruiter is pointed at the same small set of strong candidates, who saturate, and the realised match rate falls well below what offline metrics predicted. This is a congestion effect in a two-sided market, and it does not show up in per-candidate ranking metrics at all.

The fix is to treat the final step as an **assignment problem** rather than a set of independent rankings — maximising total expected fit subject to role capacity, candidate attention limits, and hard eligibility constraints, in the spirit of stable matching.

This stage is deliberately separated from scoring. Stage 2 answers *how good is this pair*; stage 3 answers *given everyone's scores, who should actually be surfaced to whom*. Conflating them is what produces concentration.

## Evaluation

| Metric | Stage | Question |
|---|---|---|
| `Recall@20` | retrieval | did the shortlist keep the good roles? |
| `NDCG@10` | reranking | are the best roles ranked at the top? |
| `MRR` | reranking | how far down is the first good role? |
| Independent-judgement alignment | reranking | is it reading the documents, or replaying the old policy? |

Split protocol:

- **Grouped by candidate** — no candidate appears in both train and test, or the model is graded on people it has already seen.
- **Ordered in time** — train on the past, test on the future, matching how the system would actually be used.

**Metrics are sliced by job family as a primary reporting axis, not a supplementary one.** Variation in retrieval quality across job families has been found to exceed the gains from model upgrades, meaning a global improvement can conceal a regression in the function that matters most. Seniority band and cold-start candidates with sparse CVs are also reported separately.

`Recall@20` is additionally sliced by demographic proxy. A fair reranker cannot repair an unfair shortlist — if a group is disproportionately excluded at retrieval, no downstream stage sees them. Audits of embedding-based CV retrieval have found substantial disparities by name alone, along with sensitivity to incidental factors such as document length, so names and PII are stripped before embedding and length is normalised.

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

## Compliance

Systems that filter job applications or evaluate candidates are classified as high-risk under the EU AI Act (Annex III, 4(a)). Three obligations have direct engineering consequences and are cheaper to build in than to retrofit:

- **Data governance (Art. 10)** — training data must be examined for bias. Here that means the funnel labels, which is the same scrutiny the independent-judgement set is designed to provide.
- **Logging (Art. 12, 19)** — retrieval candidates, rerank features and scores are logged per decision.
- **Explanation of individual decisions (Art. 86)** — a specific decision may have to be explained on request. This is a second, independent argument for interpretable structured features in the reranker.

Human review is a weaker safeguard than it appears: biased recommendations have been shown to shift human decisions rather than being caught by them. Post-hoc fairness re-ranking of the final list is a cheap, deployable mitigation that requires no retraining, and has been shown at scale to improve representation without measurable cost to outcome quality.

## Prior art

Selected work that informed the design.

| Work | Relevance |
|---|---|
| **ConFit v2** ([code](https://github.com/jasonyux/ConFit-v2)) | Closest published analogue: facet-parsed CV/JD, contrastive bi-encoder, retrieve then rerank. Source of the percentile-band hard-negative strategy and per-facet token budgeting. Includes BM25, XGBoost and embedding baselines. |
| **LinkedIn, Learning to Retrieve for Job Matching** ([2402.13435](https://arxiv.org/abs/2402.13435)) | Production two-stage retrieval trained on confirmed-hire signal. Chose an interpretable learned retriever over embeddings for the quality objective, and keeps an explicit rule layer. |
| **JobBERT-v2 / v3** ([HF](https://huggingface.co/TechWolf/JobBERT-v2)) | Asymmetric bi-encoder with separate projection heads per side, trained on 5.5M title↔skill pairs. Independent validation of the two-tower choice. |
| **ESCO skill extractors** ([HF](https://huggingface.co/jjzha)) | Taxonomy-grounded multilingual skill extraction for the parsing layer. |
| **PJB benchmark** ([2603.17386](https://arxiv.org/abs/2603.17386)) | Cross-domain performance variance exceeds model-upgrade gains; query rewriting degrades results when combined with reranking. |
| **RankPO** ([2503.10723](https://arxiv.org/abs/2503.10723)) | Demonstrates rule-mimicry: high NDCG, low agreement with independent judgement. Motivates the second evaluation set. |
| **Reciprocal recommendation / concentration** ([2411.19214](https://arxiv.org/abs/2411.19214)) | Congestion in two-sided matching; motivates the allocation stage. |
| **Wilson & Caliskan** ([2407.20371](https://arxiv.org/abs/2407.20371)) | Audit of embedding-based CV retrieval; bias enters at retrieval, before any reranker. |
| **Fairness-aware ranking at LinkedIn** ([1905.01989](https://arxiv.org/abs/1905.01989)) | Deployed post-processing mitigation, no retraining required. |
| **RecSys Challenge 2017 (XING)** ([site](http://www.recsyschallenge.com/2017/)) | Graded interaction scoring: recruiter interest weighted 20× a click, explicit rejection priced negatively. |

## Roadmap

1. Parse CVs and job descriptions into role-neutral facets; cache by document hash
2. Frozen-embedding retrieval baseline over a flat index; establish `Recall@k`
3. Ranking metrics and leakage-safe splits — grouped by candidate, ordered in time
4. Build the independent hand-judged evaluation set
5. Structured compatibility features and a hand-weighted reranker as a transparent baseline
6. LambdaMART reranker trained on graded funnel-depth labels
7. Percentile-band hard negative mining and ablations over each scoring term
8. Allocation stage; measure concentration against per-candidate ranking
9. Error analysis by job family, seniority and cold-start; per-recommendation explanations
10. Optional: hypothetical-CV generation for the register gap, cross-encoder reranking, learned facet encoders

## Scope

Built as a standalone system: generic candidate, job and outcome tables in, ranked matches out. No proprietary data, code or model artefacts belong in this repository.

Employment-related models are consequential. This one is intended to widen the set of candidates a human considers — never to reject anyone automatically.
