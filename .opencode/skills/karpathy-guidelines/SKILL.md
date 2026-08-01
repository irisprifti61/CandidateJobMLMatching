---
name: karpathy-guidelines
description: Apply four guardrails against common AI coding mistakes. Use when implementing, reviewing, or refactoring code.
---

# Karpathy Guidelines

These rules bias toward correctness and maintainability over speed. Use judgment for trivial tasks.

## Think Before Coding

- State assumptions explicitly.
- Surface multiple interpretations instead of choosing silently.
- Point out simpler approaches and unclear requirements.

## Simplicity First

- Implement only what was requested.
- Avoid abstractions for one-off code and configurability without a concrete need.
- If the solution is substantially larger than necessary, simplify it.

## Surgical Changes

- Match existing style and avoid unrelated cleanup.
- Remove only imports, variables, and functions made obsolete by the current change.
- Every changed line should trace directly to the requested outcome.

## Goal-Driven Execution

- Translate the request into observable success criteria.
- Work in short implementation and verification loops.
- Finish only after the criteria have been exercised successfully.
