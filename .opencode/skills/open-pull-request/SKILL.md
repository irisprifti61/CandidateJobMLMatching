---
name: open-pull-request
description: Prepare and open a GitHub pull request from the current work. Use when the user asks to create or open a PR.
---

# Open A Pull Request

## Inspect

1. Read repository contribution guidance and pull-request templates.
2. Inspect the current branch, status, staged and unstaged diffs, remote tracking, and recent commit style.
3. Identify the intended base branch and check whether a pull request already exists.
4. Check whether the authenticated user can push to the repository.
5. Separate intended changes from unrelated work. Never stage unrelated files.

## Verify

Run the checks appropriate to the changed surface. Review the final diff for secrets, generated files, accidental complexity, and missing tests. Stop and report any failing check that cannot be fixed safely within scope.

## Publish

Before mutating Git or GitHub, summarize the exact branch, files, commits, push, and pull-request operation. Proceed when the user has authorized that operation.

1. Create a feature branch when currently on the base branch.
2. Stage only intended files and create the smallest coherent commit or commits.
3. Follow repository commit conventions inferred from documentation and history.
4. If direct push access is unavailable, propose creating or reusing the user's fork and obtain authorization for the fork and remote changes.
5. Push the branch without rewriting remote history.
6. Open the pull request with `gh`, preserving the repository template and specifying the fork as the head when applicable.
7. Choose draft or ready-for-review based on user intent; ask when unclear.

The title should describe the resulting behavior. The body should explain what changed, why, verification performed, and known limitations. Return the pull-request URL.
