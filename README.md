# CandidateJobMLMatching

Ranking job openings by how well a candidate fits them — including roles the candidate never applied to.

## Problem

A candidate applies to one role. If they are rejected, that is usually the end of the story, even when they would have been a strong fit for a different opening.

This happens for a structural reason: recruiters are specialised. Someone who has spent two years hiring content managers does not necessarily know which signals predict success in a finance or security role. Cross-role potential is therefore rarely evaluated, not because it does not exist, but because no single person is positioned to see it.

The goal of this project is to score candidate–job fit independently of where the candidate applied.

## Idea

Represent candidates and jobs separately, then compare them.

```
CV              -> parsed facts -> candidate embedding
Job description -> parsed facts -> job embedding (precomputed, indexed)

                        candidate embedding
                                |
                        retrieve top ~20 jobs
                                |
                        rerank each pair in detail
                                |
                            ranked jobs
```

Two stages, because they optimise different things:

1. **Retrieval** — cheap, approximate, runs against every job. Narrows thousands of openings down to a shortlist of roughly 20 plausible ones.
2. **Reranking** — expensive, precise, runs only on the shortlist. Examines each candidate–job pair individually.

Candidate features are extracted once, without reference to any target job, so a CV is processed a single time and compared against all openings. Job embeddings are precomputed and only refreshed when a description changes.

## Why embeddings alone are not enough

Cosine similarity between a CV and a job description measures topical resemblance. That is a good signal for occupation, domain, skills and education, and a poor signal for fitness.

Semantic similarity handles reasonably well:

- occupation and industry
- relevant skills
- related work history
- education subject
- broad responsibilities

It systematically fails on:

- junior versus senior roles (the text looks nearly identical)
- mandatory versus optional requirements
- years-of-experience thresholds
- location, visa and language constraints
- a technology mentioned in passing versus actual proficiency
- transferable skills described in different vocabulary

A candidate with three years of experience and a candidate with fifteen can produce almost the same embedding. That is exactly the distinction hiring depends on.

The reranking stage exists to catch what the embedding cannot see, by combining semantic similarity with structured checks:

```
score = semantic_similarity
      + skill_coverage
      + seniority_compatibility
      + experience_compatibility
      - missing_mandatory_requirements
```

Parsed facts are converted into canonical text or separate facets before embedding, rather than embedding raw JSON:

```
Skills:      Python, SQL, distributed systems
Experience:  4 years, backend engineering
Seniority:   mid-level
Education:   MSc computer science
Industries:  fintech, SaaS
```

The same facets are generated for job descriptions, so candidate capabilities and job requirements can be compared field by field instead of as opaque blobs.

## Asymmetry

Candidates and jobs are encoded by separate towers rather than a shared encoder. Fit is not a symmetric relation: a candidate *possesses* capabilities, a job *requires* them. Holding a skill that a job does not ask for is harmless; lacking one the job demands is disqualifying. A single shared representation cannot express that difference.

## Labels

This is the hard part, and it is worth being explicit about it.

Outcomes are only observed for pairs where the candidate actually applied. Every other candidate–job pair is unobserved — which is not the same as negative. A candidate who never applied to a role may have been an excellent fit for it.

Additional caveats:

- Progression through a hiring funnel partly reproduces the existing selection policy, including its blind spots. A model trained naively on it learns to imitate current behaviour rather than improve on it.
- Later funnel stages are extremely sparse: of roughly 40,000 applications in one observed cohort, about 34 reached the final stage.
- Withdrawal, position closure, and visa or location constraints are competing outcomes and should not be treated as evidence of poor fit.
- Evaluation splits must be grouped by candidate and ordered in time, otherwise the same person leaks across train and test.

Any claim this project makes should therefore be about *structured candidate–job compatibility*, not about predicted job performance.

## Evaluation

The question is not whether the ranking scores well in the abstract, but whether it surfaces something a recruiter would have missed.

- Take candidates rejected for role A in a past period.
- Ask whether the model would have flagged them for role B.
- Check whether comparable profiles were later hired into role B.

The output that matters is not a score. It is a short, explainable list:

> These candidates did not pass pipeline X, but show strong signals for pipeline Y — and here is why.

If the reason cannot be stated in a sentence, the recommendation is not usable.

## Status

Early. Defining the problem and the evaluation protocol before writing modelling code.

Planned order of work:

1. Parsing CVs and job descriptions into role-neutral facts
2. Frozen-embedding retrieval baseline with an approximate nearest-neighbour index
3. Ranking metrics and a leakage-safe evaluation split
4. Structured compatibility reranker
5. Hard negatives and ablations
6. Error analysis across seniority, and cold-start behaviour

## Scope and data

Built as a standalone system: generic candidate, job and outcome tables in, ranked matches out. No proprietary data, code or model artefacts are part of this repository.

Employment-related models are consequential. This one is intended to widen the set of candidates a human considers, never to reject anyone automatically.
