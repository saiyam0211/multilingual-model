"""Multi-turn evaluation: run conversational attacks on eval prompts.

Compares single-turn vs multi-turn ASR to measure the delta
from follow-up attacks. Saves results to results/multi_turn_eval.jsonl.

Usage:
    python scripts/eval_multi_turn.py [--max-turns 3] [--limit 50]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from polyglot_redteam.multi_turn import multi_turn_attack, _is_refused, LANG_NAMES


def load_eval_prompts(path: str = "data/eval_prompts_v3.jsonl") -> list[dict]:
    """Load eval prompts from JSONL."""
    records = []
    p = Path(path)
    if not p.exists():
        # Fall back to vulnerability dataset
        p = Path("data/vulnerability_dataset_final.jsonl")
    if not p.exists():
        print(f"ERROR: No eval data found at {path}")
        sys.exit(1)

    with p.open() as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def main():
    parser = argparse.ArgumentParser(description="Multi-turn evaluation")
    parser.add_argument("--max-turns", type=int, default=3, help="Max conversation turns")
    parser.add_argument("--limit", type=int, default=50, help="Max prompts to evaluate")
    parser.add_argument("--attacker-model", type=str, default=None,
                        help="HF model ID for attacker follow-ups (optional)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="results/multi_turn_eval.jsonl")
    args = parser.parse_args()

    random.seed(args.seed)

    # Load prompts
    print("→ Loading eval prompts...")
    all_prompts = load_eval_prompts()

    # Filter to confirmed gaps only (for vulnerability dataset)
    gap_prompts = [p for p in all_prompts if p.get("is_safety_gap", True)]
    if len(gap_prompts) > args.limit:
        gap_prompts = random.sample(gap_prompts, args.limit)
    print(f"  {len(gap_prompts)} prompts selected (limit={args.limit})")

    # Run multi-turn attacks
    print(f"\n→ Running multi-turn attacks (max_turns={args.max_turns})...")
    results = []
    single_turn_success = 0
    multi_turn_success = 0

    for i, p in enumerate(gap_prompts):
        prompt_text = p.get("prompt") or p.get("indic_prompt", "")
        lang = p.get("lang") or p.get("language", "hi")
        cat = p.get("category", "unknown")

        if not prompt_text.strip():
            continue

        print(f"  [{i+1}/{len(gap_prompts)}] {lang} / {cat}: {prompt_text[:60]}...")

        result = multi_turn_attack(
            prompt=prompt_text,
            lang=lang,
            category=cat,
            max_turns=args.max_turns,
            attacker_model=args.attacker_model,
        )

        results.append(result)

        # Track single-turn vs multi-turn
        if result.success:
            multi_turn_success += 1
            if result.success_turn == 1:
                single_turn_success += 1

        # Progress logging every 10 prompts
        if (i + 1) % 10 == 0:
            st_asr = single_turn_success / (i + 1) * 100
            mt_asr = multi_turn_success / (i + 1) * 100
            print(f"    Progress: single-turn ASR={st_asr:.1f}%, "
                  f"multi-turn ASR={mt_asr:.1f}% (+{mt_asr - st_asr:.1f}pp)")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        for r in results:
            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
    print(f"\n✓ Saved {len(results)} results to {args.output}")

    # Summary
    total = len(results)
    if total == 0:
        print("No results to summarize.")
        return

    st_asr = single_turn_success / total * 100
    mt_asr = multi_turn_success / total * 100
    delta = mt_asr - st_asr

    print(f"\n{'='*60}")
    print(f"  MULTI-TURN EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"  Total prompts:       {total}")
    print(f"  Max turns:           {args.max_turns}")
    print(f"  Single-turn ASR:     {st_asr:.1f}% ({single_turn_success}/{total})")
    print(f"  Multi-turn ASR:      {mt_asr:.1f}% ({multi_turn_success}/{total})")
    print(f"  Delta:               +{delta:.1f}pp")
    print()

    # Breakdown by success turn
    turn_dist = {}
    for r in results:
        if r.success and r.success_turn:
            turn_dist[r.success_turn] = turn_dist.get(r.success_turn, 0) + 1
    print("  Success by turn:")
    for turn, count in sorted(turn_dist.items()):
        print(f"    Turn {turn}: {count} ({count/total*100:.1f}%)")

    # Breakdown by language
    lang_stats = {}
    for r in results:
        if r.lang not in lang_stats:
            lang_stats[r.lang] = {"total": 0, "st_success": 0, "mt_success": 0}
        lang_stats[r.lang]["total"] += 1
        if r.success:
            lang_stats[r.lang]["mt_success"] += 1
            if r.success_turn == 1:
                lang_stats[r.lang]["st_success"] += 1

    print("\n  By language:")
    for lang, s in sorted(lang_stats.items()):
        st = s["st_success"] / max(s["total"], 1) * 100
        mt = s["mt_success"] / max(s["total"], 1) * 100
        name = LANG_NAMES.get(lang, lang)
        print(f"    {name}: ST={st:.0f}% → MT={mt:.0f}% (+{mt-st:.0f}pp) [{s['total']} prompts]")

    print(f"{'='*60}")


if __name__ == "__main__":
    main()
