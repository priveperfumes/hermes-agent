# Playbook: Merge Conflict

You're here because `mergeable_state == 'dirty'` or you saw conflict markers blocking the merge gate.

## Step 1: Try a clean rebase

Call `mcp__github__update_pull_request_branch`. This asks GitHub to update the PR branch with the latest base via merge or rebase. If GitHub can resolve it cleanly, your job here is done — the next `synchronize` event will re-trigger and you'll proceed.

If `update_pull_request_branch` succeeds: comment `🛩️ claude-pilot: rebased onto <base>.` and exit.

## Step 2: If GitHub rebase fails

GitHub returns an error if the merge isn't clean. Do **not** attempt manual conflict resolution — you don't have enough context about author intent for both branches.

Instead:
1. Comment on the PR:
   ```
   🛩️ claude-pilot: merge conflict with `<base>`. Cannot auto-resolve. Please rebase locally and push, or click "Update branch" if conflicts are simple.
   ```
2. Add label `needs-human`, remove `claude-pilot-active`.
3. Discord ping:
   ```
   :crossed_swords: <REPO> PR #<N> merge conflict — needs manual rebase. <pr_url>
   ```
4. Exit.

## Step 3: After human rebases

When the human pushes a rebase, `pull_request.synchronize` fires. The recursion guard skips Claude's own commits, so the human's rebase commit triggers a fresh run. That run will see `mergeable_state == 'clean'` (assuming they resolved correctly) and proceed normally.

If the rebase still leaves conflicts (rare), you'll loop — but the `needs-human` label is still set, so the workflow's `if:` guard will skip you. Stay out until the human removes the label.

## Hard rules

- **Never** force-push to a branch you didn't author, even to "fix" a conflict.
- **Never** delete and recreate the branch.
- **Never** merge with `--strategy theirs` or `--strategy ours` automatically.
- **Never** edit conflict markers and commit. The author has context you don't.

Conflicts are one of the few places where machine confidence is misplaced. Defer to the human.
