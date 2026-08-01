# Agent Instructions

## Start With Context

- Read the repository's README and relevant documentation before changing code.
- Inspect nearby files and follow established structure, naming, and style.
- State assumptions when requirements are ambiguous. Ask instead of silently guessing.

## Engineering Rules

Apply the `karpathy-guidelines` skill for non-trivial implementation, review, and refactoring work.

## Code Quality

- Preserve existing behavior unless the task explicitly changes it.
- Keep abstractions proportional to demonstrated complexity. Avoid speculative flexibility.
- Let errors propagate unless the current layer can genuinely handle them.
- Test behavior through public interfaces rather than implementation details.
- Never commit secrets, credentials, personal data, generated caches, or local environment files.

## Documentation And Artifacts

- Treat code, tests, documentation, examples, and configuration as descriptions of the current system.
- Put change history and rationale in commit or pull-request metadata, not in lasting artifacts.
- Keep each fact in one authoritative place and link to it instead of duplicating it.
- Match existing documentation locations and conventions before creating new directories.
- Prefer primary sources for research and cite claims with stable links.
- Do not create status reports, migration notes, or planning documents unless they will remain useful after the task.

## Completion

- Run the narrowest relevant tests, formatting, linting, and type checks available.
- Review the final diff for unrelated changes, accidental complexity, stale comments, and sensitive data.
- Report what changed, how it was verified, and any remaining limitations.
