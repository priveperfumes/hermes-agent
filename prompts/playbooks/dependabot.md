# Playbook: Dependabot / Renovate

PRs from `dependabot[bot]` or `renovate[bot]` get a fast path because they're high-volume and predictable.

## Classification

Read the diff. Classify the PR by what it changes:

| Class | Diff content | Action |
|---|---|---|
| **Lockfile-only** | Only `package-lock.json` / `pnpm-lock.yaml` / `poetry.lock` / `Cargo.lock` / `Gemfile.lock` / etc. | Fast path: if CI green, auto-merge. |
| **Patch bump** | Lockfile + manifest version bump matching `^x.y.z` → `^x.y.z+1` (last segment only) | Fast path: if CI green, auto-merge. |
| **Minor bump** | Lockfile + manifest version bump matching `^x.y.z` → `^x.y+1.0` | Standard path: full self-review, then auto-merge if clean. |
| **Major bump** | Manifest version bump where major segment changed | Stop. Add label `breaking-change`, escalate to human. |
| **Unexpected files** | Anything beyond manifest + lockfile (e.g., source files, configs) | Treat as a regular PR — Dependabot shouldn't be touching these. Full review. |

## Fast path (lockfile-only or patch bump)

1. Verify the diff matches the classification — read every changed file. Reject the fast path if you see anything unexpected.
2. Run **the full merge gate** from `pr-pilot.md` — every condition, no shortcuts:
   - ✅ All required CI checks `success`
   - ✅ Self-review clean (lockfile-only / patch bump = trivially clean if classification holds)
   - ✅ Zero unresolved review threads (CodeRabbit + human)
   - ✅ Secret scan clean (`mcp__github__run_secret_scanning`)
   - ✅ `mergeable_state != 'dirty'`
   - ✅ No labels: `needs-human`, `claude-pilot-paused`, `do-not-merge`, `breaking-change`
   - ✅ PR is not a draft
3. If gate clean: call `mcp__github__enable_pr_auto_merge` with `merge_method: 'squash'`, then remove `claude-pilot-active`.
4. Comment: `🛩️ claude-pilot: dependabot fast path — <classification>, full gate clean, merging.`
5. Exit.

The "fast" in fast path refers to the **classification** (no scope review needed for trivial bumps), not to a weakened gate. Never skip secret scan or threads check.

## Major bump escalation

Major version bumps frequently break things. Even if CI passes, run the standard escalation procedure from `pr-pilot.md`:
1. Add label `needs-human` and `breaking-change`.
2. Remove label `claude-pilot-active`.
3. Comment on the PR:
   ```text
   🛩️ claude-pilot: major version bump (`<package> <old> → <new>`). Handing off to a human for review.
   ```
4. Discord ping (one only — use the `<!-- pilot-notify:blocked-major-bump:<sha> -->` marker per the dedup rule).
5. Exit cleanly. Do not loop.

## Common gotchas

- **Multi-package PRs** — Dependabot/Renovate can group updates. If one package in the group is a major bump, treat the whole PR as needing human review.
- **GitHub Actions version bumps** — `.github/workflows/*.yml` changes get the same classification rules. Pin SHAs are preferable; if Dependabot is moving from `@v1` to `@v2` of an action, that's a major bump.
- **Security updates** — Dependabot sometimes flags PRs as `security`. Same rules apply; CI green + non-major = fast path. But check the diff actually addresses the advisory before merging.
- **Stale PRs** — If a Dependabot PR has been open >7 days and is still passing CI, it's safe to fast-path-merge anyway.
