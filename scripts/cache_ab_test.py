#!/usr/bin/env python3
"""A/B test: prompt caching before vs after the fix.

Runs the same 4-turn conversation twice against the live Anthropic API:
  A) OLD behavior — ephemeral system prompt mutates the system prefix every turn
  B) NEW behavior — ephemeral content moves to user message, system prefix stays stable

Prints per-turn cache stats and a side-by-side summary at the end.

Usage:
    cd /path/to/hermes-agent
    python scripts/cache_ab_test.py
"""

import os
import sys
import time

from dotenv import load_dotenv
load_dotenv(os.path.expanduser("~/.hermes/.env"))

from run_agent import AIAgent

PROMPTS = [
    "What is 2+2? Reply in one word.",
    "What is 3+3? Reply in one word.",
    "What is 4+4? Reply in one word.",
    "What is 5+5? Reply in one word.",
]

MODEL = os.environ.get("AB_MODEL", "claude-sonnet-4-20250514")


def run_session(label: str, use_fix: bool) -> dict:
    """Run a multi-turn session and return cache stats."""
    agent = AIAgent(
        model=MODEL,
        provider="anthropic",
        quiet_mode=True,
    )

    # Both paths get an ephemeral system prompt — this is the content that
    # was previously appended to the system prompt (mutating it every turn).
    agent.ephemeral_system_prompt = (
        f"Session timestamp: {time.time():.0f}. "
        "Always reply concisely."
    )

    if not use_fix:
        # Simulate OLD behavior: disable prompt caching so the ephemeral
        # system prompt is appended directly to the system message.
        agent._use_prompt_caching = False
        agent._use_native_anthropic_auto_cache = False

    history = []
    turns = []

    for i, prompt in enumerate(PROMPTS, 1):
        # Pause between turns: lets cache populate and avoids rate limits.
        if i > 1:
            time.sleep(5)
        r = agent.run_conversation(prompt, conversation_history=history)
        history = r["messages"]

        turns.append({
            "turn": i,
            "cache_read": r["cache_read_tokens"],
            "cache_write": r["cache_write_tokens"],
            "input": r["input_tokens"],
        })

        total = r["input_tokens"] + r["cache_read_tokens"] + r["cache_write_tokens"]
        hit = (r["cache_read_tokens"] / total * 100) if total > 0 else 0
        print(f"  [{label}] Turn {i}: read={r['cache_read_tokens']:>6,}  "
              f"write={r['cache_write_tokens']:>6,}  "
              f"input={r['input_tokens']:>6,}  "
              f"hit={hit:.0f}%")

    total_read = agent.session_cache_read_tokens
    total_write = agent.session_cache_write_tokens
    total_input = agent.session_input_tokens
    total_all = total_input + total_read + total_write
    hit_pct = (total_read / total_all * 100) if total_all > 0 else 0

    return {
        "label": label,
        "turns": turns,
        "total_read": total_read,
        "total_write": total_write,
        "total_input": total_input,
        "total_all": total_all,
        "hit_pct": hit_pct,
    }


def main():
    print(f"Model: {MODEL}")
    print(f"Turns: {len(PROMPTS)}")
    print()

    # ── A: OLD behavior (no caching fix) ──
    print("A) OLD behavior — ephemeral prompt in system message (cache-busting)")
    print("─" * 70)
    old = run_session("OLD", use_fix=False)

    print()

    # Pause between runs to avoid rate limits
    print("\nWaiting 30s between runs (rate limit cooldown)...\n")
    time.sleep(30)

    # ── B: NEW behavior (caching fix active) ──
    print("B) NEW behavior — ephemeral prompt in user message (cache-preserving)")
    print("─" * 70)
    new = run_session("NEW", use_fix=True)

    # ── Summary ──
    print()
    print("=" * 70)
    print("SIDE-BY-SIDE COMPARISON")
    print("=" * 70)
    print(f"{'':>20} {'OLD (no fix)':>18} {'NEW (with fix)':>18} {'Δ':>12}")
    print(f"{'─'*20} {'─'*18} {'─'*18} {'─'*12}")
    print(f"{'Cache reads':>20} {old['total_read']:>14,} tk {new['total_read']:>14,} tk {new['total_read'] - old['total_read']:>+10,}")
    print(f"{'Cache writes':>20} {old['total_write']:>14,} tk {new['total_write']:>14,} tk {new['total_write'] - old['total_write']:>+10,}")
    print(f"{'Uncached input':>20} {old['total_input']:>14,} tk {new['total_input']:>14,} tk {new['total_input'] - old['total_input']:>+10,}")
    print(f"{'Total input':>20} {old['total_all']:>14,} tk {new['total_all']:>14,} tk")
    print(f"{'Hit rate':>20} {old['hit_pct']:>17.1f}% {new['hit_pct']:>17.1f}%")
    print()

    if new["total_all"] > 0 and old["total_all"] > 0:
        # Cost model: reads at 10% of input cost, writes at 125%
        old_effective = old["total_input"] + old["total_read"] * 0.1 + old["total_write"] * 1.25
        new_effective = new["total_input"] + new["total_read"] * 0.1 + new["total_write"] * 1.25
        if old_effective > 0:
            savings_pct = (1 - new_effective / old_effective) * 100
            print(f"Effective input cost (first session):  {savings_pct:+.0f}%")
            print(f"  First session has writes at 1.25x — a one-time overhead.")
            print(f"  On subsequent turns/sessions that hit the cache,")
            print(f"  reads cost 0.1x → net savings grow with each cache hit.")
            print()

        # Show what happens if all those writes become reads next time
        if new["total_write"] > 0:
            future_effective = new["total_input"] + new["total_write"] * 0.1  # writes become reads
            future_savings = (1 - future_effective / old_effective) * 100
            print(f"Effective input cost (cached session):  {future_savings:+.0f}%")
            print(f"  If the same prefix is reused (cache reads instead of writes),")
            print(f"  effective cost drops to ~{future_effective:,.0f} token-equivalents")
            print(f"  vs {old_effective:,.0f} uncached — a {future_savings:.0f}% reduction.")
    print()


if __name__ == "__main__":
    main()
