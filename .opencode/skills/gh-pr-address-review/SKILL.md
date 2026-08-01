---
name: gh-pr-address-review
description: Triage and address GitHub pull-request review feedback. Use when the user asks to resolve review comments or update a PR after review.
---

# Address Pull-Request Review

## Collect And Triage

1. Confirm the repository, current branch, and associated pull request.
2. Inspect the working tree without overwriting unrelated local changes.
3. Fetch unresolved review threads and actionable top-level comments with `gh` and the GitHub API.
4. Classify each item as actionable, already addressed, superseded, blocked, or a false positive.
5. Read the relevant code and verify every claim before changing anything.

Cluster comments that concern the same architectural or design decision. Resolve the underlying issue once rather than applying contradictory local patches. For structural feedback, inspect comparable code in the repository before selecting a pattern.

## Implement And Verify

1. Apply only changes required by actionable feedback.
2. Preserve meaningful comments and documentation when moving or simplifying code.
3. Run focused tests and the repository's required checks.
4. Review the resulting diff for scope, regressions, and accidental complexity.

## Publish

Before commits, pushes, GitHub replies, or thread resolution, summarize the intended mutations and proceed when authorized.

- Group commits by coherent feedback cluster and follow repository conventions.
- Reply directly and concisely to each addressed comment with the change and verification evidence.
- Explain false positives with code evidence rather than changing correct behavior.
- Resolve only threads that are fully addressed or conclusively answered.
- Leave blocked or partially addressed threads open and report what is needed.

Finish with addressed clusters, checks run, commits pushed, replies posted, threads resolved, and remaining blockers.
