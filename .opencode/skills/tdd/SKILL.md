---
name: tdd
description: Develop behavior through a red-green loop. Use when the user requests test-first development, regression tests, or red-green-refactor.
---

# Test-Driven Development

Work in vertical slices:

1. Identify the public seam and one observable behavior.
2. Write the smallest test that expresses that behavior using an independent expected value.
3. Run it and confirm it fails for the intended reason.
4. Implement only enough production code to make it pass.
5. Run the relevant test suite, then repeat for the next behavior.

Good tests survive internal refactoring because they exercise public contracts. Avoid tests that mock internal collaborators, call private methods, duplicate the implementation in the assertion, or rely on snapshots without meaningful review.

Keep refactoring separate from making the test pass. Review the completed behavior first, then simplify without changing the contract.
