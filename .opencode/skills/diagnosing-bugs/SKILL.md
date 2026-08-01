---
name: diagnosing-bugs
description: Diagnose bugs and performance regressions through a reproducible feedback loop. Use when behavior is broken, failing, flaky, or unexpectedly slow.
---

# Diagnosing Bugs

## Build The Loop

Create the fastest deterministic command that exercises the reported symptom and can distinguish broken from fixed behavior. Prefer, in order:

1. A failing test at the real public seam.
2. A minimal CLI, HTTP, or browser reproduction.
3. A replayable fixture or differential harness.
4. Targeted instrumentation tied to a specific hypothesis.

Do not settle for "does not crash" when the bug concerns incorrect output.

## Minimize And Explain

1. Reproduce the exact symptom.
2. Remove inputs and dependencies until every remaining element is necessary.
3. List several falsifiable hypotheses and rank them by evidence.
4. Test one variable at a time. Prefer debuggers and focused measurements over broad logging.

## Fix And Verify

1. Turn the minimal reproduction into a regression test when a correct seam exists.
2. Apply the smallest fix that addresses the demonstrated cause.
3. Run both the regression test and the original reproduction.
4. Remove temporary instrumentation and record the root cause in commit or pull-request metadata.

For performance regressions, establish a repeatable benchmark before changing code and compare measurements afterward.
