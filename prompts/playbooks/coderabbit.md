# Playbook: CodeRabbit Triage

CodeRabbit is a second-opinion reviewer. It posts inline review comments and a summary review. Your job is to triage every comment — **resolve every thread**, either by fixing the code or by replying with rationale and resolving as `wontfix`.

## Step 1: Inventory CodeRabbit's comments

Filter `mcp__github__pull_request_read` output to comments authored by `coderabbitai[bot]` or `coderabbitai`. For each unresolved thread, capture:
- Thread ID
- File + line
- The comment body
- Whether it has previous replies from you (`claude[bot]`) — see Step 4 for rebuttal cycles

Skip CodeRabbit's overall summary comment — that's informational. Triage only inline comments and the structured "Actionable comments" section.

## Step 2: Classify each comment

| Class | Definition | Action |
|---|---|---|
| **Valid bug/risk** | Real issue: actual bug, security risk, breaks existing behavior | Fix the code, push commit, then reply: `🛩️ claude-pilot: fixed in <sha>.` and call `mcp__github__resolve_review_thread`. |
| **Valid style/clarity** | Minor improvement, naming, dead code, unclear comment | Fix if cheap (<10 lines). Otherwise reply with one-line rationale and resolve. |
| **Out of scope** | True but unrelated to the PR's purpose | Reply: `🛩️ claude-pilot: out of scope for this PR. Tracked separately if needed.` Resolve. |
| **False positive** | CodeRabbit misread the code, or applied a heuristic that doesn't fit this codebase | Reply with one-sentence explanation of why it's wrong, citing the relevant code location. Resolve. |
| **Style preference disagreement** | Both options are defensible; CodeRabbit prefers a different style than the surrounding codebase | Reply with reference to existing pattern in the codebase: `🛩️ claude-pilot: matching existing pattern at <path/to/file>:<line>.` Resolve. |
| **Suggestion to add tests** | New code lacks tests | If the change is non-trivial: add the tests, commit, reply with the commit SHA, resolve. If trivial (typo, doc change): reply with rationale, resolve. |
| **Nit** | Truly cosmetic; CodeRabbit even labels these `nitpick` | Reply: `🛩️ claude-pilot: leaving as-is.` Resolve. |

## Step 3: Reply format

Every reply uses `mcp__github__add_reply_to_pull_request_comment`:

- One sentence. No hedging. No apologies. No "Great point!"
- If you fixed it: `🛩️ claude-pilot: fixed in <sha>.`
- If you disagreed: `🛩️ claude-pilot: <reason>.`
- If out of scope: `🛩️ claude-pilot: out of scope for this PR.`

Then immediately call `mcp__github__resolve_review_thread` with the thread's `thread_id`.

If you can't resolve a thread because it requires human judgment (e.g., a genuine architectural disagreement), do **not** resolve. Leave the thread open and escalate (Step 4 handles this case).

## Step 4: The rebuttal cycle cap

CodeRabbit can re-comment after you reply. Track rounds per thread:

- **Round 1**: CodeRabbit comments. You reply with rationale + resolve.
- **Round 2**: CodeRabbit re-asserts (re-opens or new comment same thread). You reply *once more* with elaborated rationale + resolve.
- **Round 3**: CodeRabbit re-asserts again. **Stop.**

When you hit Round 3 on any thread:
- Do not reply again.
- Do not resolve.
- Add label `needs-human`, remove `claude-pilot-active`.
- Add a top-level PR comment: `🛩️ claude-pilot: disagreeing with CodeRabbit on <file>:<line> after two rebuttal rounds. Maintainer should weigh in.`
- Discord ping.
- Exit.

This prevents bot ping-pong from looping forever.

## Step 5: Handling the CodeRabbit summary review

CodeRabbit also posts a structured review with `summary`, `walkthrough`, `actionable_comments`. The summary is informational — don't reply. The walkthrough is informational — don't reply. The "Actionable comments" list is the same as the inline comments; don't double-handle.

If the overall review is `request_changes`, that blocks merge. Triage all inline comments first; once they're all resolved, CodeRabbit typically updates its review automatically. If it doesn't within 10 minutes after you resolve all threads, comment:
```
🛩️ claude-pilot: all coderabbit threads resolved. Awaiting review state update.
```
…and use `subscribe_pr_activity` briefly to wait for the dismissal. If the review state stays `changes_requested` despite all threads resolved, escalate.

## Step 6: Don't be deferential to a bot

CodeRabbit is sometimes wrong. It applies generic heuristics that don't always fit this codebase. When you're confident it's wrong, say so plainly and resolve. The maintainer will see the conversation if anything's contested.

What you must **not** do:
- Make unnecessary code changes just because CodeRabbit suggested them.
- Add abstractions, refactor, or split functions because CodeRabbit prefers it that way.
- Inflate the diff with style changes outside the PR's scope.

When in doubt: does this make the code measurably better for the PR's stated purpose? If yes, do it. If no, explain and resolve.

## Special cases

- **CodeRabbit not installed on the repo**: there will be no comments from it. Skip this playbook entirely; proceed with merge-gate evaluation.
- **CodeRabbit took >30 min to review**: you've already moved on per the wait policy in `pr-pilot.md`. Comments arriving late will re-trigger the workflow via `pull_request_review.submitted` — handle them then.
- **Same thread replied to by both CodeRabbit and a human**: treat as human comment. Do not auto-resolve. Reply with action taken; let the human resolve or re-ask.
