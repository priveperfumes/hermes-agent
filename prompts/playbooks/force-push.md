# Playbook: Force-push detected

`pull_request.synchronize` fires for both regular pushes and force-pushes. Force-pushes rewrite history — your prior reasoning about the diff is invalidated.

## Detection

The `synchronize` event payload includes `before` and `after` SHAs. Force-push signal:
- `before` is **not** an ancestor of `after` on the PR branch.

You can verify with `git merge-base --is-ancestor <before> <after>` from the checkout. If it returns non-zero, the new HEAD is not a descendant of the old HEAD: force-push.

You can also detect by reading the PR's commits via `mcp__github__pull_request_read` and noticing that commits referenced in your prior PR comments no longer exist in the branch.

## Action

Treat a force-push as if the PR was just opened:

1. Discard any in-flight reasoning. Don't reference prior commit SHAs in your next comment — they may not exist on the branch anymore.
2. Re-read the entire diff against the base branch.
3. Re-run self-review from scratch on the new diff.
4. Re-evaluate the merge gate from scratch.
5. Comment once at the top of the PR:
   ```
   🛩️ claude-pilot: force-push detected. Re-reviewing from `<new_head_sha>`.
   ```
6. Proceed normally.

## Open review threads after force-push

Inline review comments survive force-pushes but become "outdated" on GitHub if the line they referenced moved or disappeared. CodeRabbit will typically re-review automatically.

For unresolved threads:
- If the line they reference still exists in the new diff: still applicable, handle per `coderabbit.md` or human-review rules.
- If the line is gone (the code was rewritten): comment on the thread `🛩️ claude-pilot: code rewritten in force-push; please re-flag if still applicable.` and resolve.

## Don't fight the author

Force-pushes are usually intentional — the author rewrote their history to clean it up before merge. Don't comment scolding them. Don't refuse to re-review. Just re-baseline and continue.

## Exception: force-push by a non-author

If `ACTOR` on the synchronize event isn't the PR's author, that's unusual. Comment:
```
🛩️ claude-pilot: force-push by `<actor>` (not PR author). Re-reviewing.
```
…and proceed. The PR author will see the notification.
