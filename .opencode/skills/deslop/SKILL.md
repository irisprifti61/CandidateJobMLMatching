---
name: deslop
description: Remove unnecessary AI-generated complexity before completion. Use after implementation and before committing changes.
---

# Deslop

Review only the changes introduced for the current task:

- Remove unrelated edits and speculative features.
- Simplify needless abstractions, wrappers, configuration, and defensive branches.
- Remove comments that restate obvious code or do not match local style.
- Replace type-system escapes with accurate types where practical.
- Let errors propagate instead of catching exceptions that cannot be handled meaningfully.
- Confirm naming, structure, and formatting match surrounding code.
- Check for missed cases, behavior regressions, generated files, and sensitive data.

Every remaining changed line should contribute directly to the requested result.
