# Playbook: CI Failure

You're here because `check_suite.completed` fired with a `failure` conclusion, or you saw failing required checks during merge-gate evaluation.

## Step 1: Identify exactly what failed

Use `mcp__github__pull_request_read` to get the check runs for `PR_HEAD_SHA`. For each failing check:
- Check name (e.g., `lint`, `test`, `build`, `typecheck`)
- Conclusion (`failure`, `cancelled`, `timed_out`)
- Logs URL or summary

If logs aren't in the check output, fetch via `mcp__github__list_commits` → workflow run → job logs. If you genuinely can't read the failure detail, comment:
```text
🛩️ claude-pilot: CI failure on `<check>` but logs unreadable. Re-running once.
```
…and re-run the workflow itself (do **not** use `mcp__github__update_pull_request_branch` for this — that mutation rebases base into the PR branch, which is not what you want). Re-run options, in order of preference:

1. `gh run rerun <run-id> --failed` from `Bash` — re-runs only failed jobs against the same SHA.
2. `gh run rerun <run-id>` — full re-run.
3. As a last resort, ask the user to re-trigger via the Actions UI.

If the re-run fails again with logs still unreadable, escalate (label `needs-human`).

## Step 2: Classify the failure

| Type | Signal | Action |
|---|---|---|
| Flaky test | Same test passes locally, fails intermittently in CI; previously known flaky from issue tracker | Retry once. If it fails twice, treat as real failure. |
| Lint/format | Linter exit code, formatter diff | Run formatter locally, push fix |
| Type error | `tsc`, `mypy`, `pyright` failure | Read the error location, fix the type, push |
| Test failure | Specific assertion or error trace | Read the test, read the code under test, fix. **Do not change the test to make it pass unless the test itself is wrong.** |
| Build failure | Compilation, bundling, dependency resolution | Read the error, fix imports/syntax/deps |
| Infrastructure | "Resource not available", "exit code 137", network timeout to external service | Retry once via re-running the job. If repeats, escalate as `infra-flake`. |
| Unrelated to PR diff | Failure in code the PR didn't touch | Check `main` branch — if it's also broken, this is an existing issue. Comment + escalate. Don't fix outside scope. |

## Step 3: Fix it

Make the minimum change that fixes the failure without expanding scope:
- One commit per logical fix (e.g., one for lint, one for the actual test failure — not bundled).
- Commit message: `pilot: fix <check-name> — <one-line cause>`.
- Push to the PR head branch via `git push origin HEAD`.

After pushing, exit. The `pull_request.synchronize` event will re-trigger the workflow, and the next run will re-evaluate.

## Step 4: Loop detector

Before pushing your fix, check the loop detector:

1. List the last 3 commits on the PR branch authored by `claude[bot]` or with messages starting `pilot: fix`.
2. Extract the (file, symbol) pairs they touched.
3. If your *current* fix touches a (file, symbol) that appears in **all 3** of those prior commits, you are looping. **Stop.** Do not push.

Instead:
- Comment: `🛩️ claude-pilot: looping on \`<file>:<symbol>\` after 3 fix attempts. The CI failure isn't responding to the obvious fixes. Handing off.`
- Add label `needs-human`, remove `claude-pilot-active`.
- Discord ping with the loop reason.
- Exit.

## Step 5: Don't fix what you shouldn't

- **Don't disable failing tests** to make CI pass. Ever.
- **Don't change required-checks config** to remove the failing one.
- **Don't bypass branch protection** or merge with failures via admin rights even if the action could.
- **Don't commit `.skip()` / `xit()` / `@pytest.mark.skip`** unless the test is genuinely obsolete and you explain why in the commit message.
- **Don't suppress errors** with `try/except: pass` or empty catch blocks.

If you find yourself wanting to do any of these, that's the loop detector telling you the task needs a human.

## Common patterns

**Snapshot test diffs** — if the PR's intent is to change UI/output and snapshots fail accordingly, regenerate the snapshots and commit them. Reference the regen command in the PR/repo's CI logs.

**Lockfile drift** — if `package-lock.json` / `pnpm-lock.yaml` / `poetry.lock` is stale, run the install command and commit the updated lockfile.

**Coverage threshold failures** — if a coverage gate fails because new code isn't tested, write the missing test rather than lowering the threshold.

**Migration / schema check failures** — schema changes need explicit approval. Comment, label `breaking-change`, escalate.
