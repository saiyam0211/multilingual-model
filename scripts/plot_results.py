"""B11: Plot baseline vs post-GRPO results.

Creates comparison charts showing:
1. ASR by language (baseline vs post-GRPO)
2. ASR by category
3. Reward distribution
4. Gate outcomes
5. Overall summary

Run locally: python scripts/plot_results.py

Output: results/ directory with PNG files
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = REPO_ROOT / "data" / "baseline_results.jsonl"
POST_PATH = REPO_ROOT / "data" / "post_grpo_results.jsonl"
OUT_DIR = REPO_ROOT / "results"

LANG_LABELS = {
    "hi": "Hindi", "ta": "Tamil", "bn": "Bengali",
    "mr": "Marathi", "te": "Telugu", "kn": "Kannada",
}

THRESHOLD = 0.5


def load_results(path: Path) -> list[dict]:
    if not path.exists():
        print(f"  ⚠ {path} not found")
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def compute_metrics(results: list[dict]) -> dict:
    by_lang = defaultdict(list)
    by_cat = defaultdict(list)

    for r in results:
        by_lang[r.get("lang", "?")].append(r)
        by_cat[r.get("category", "?")].append(r)

    lang_asr = {}
    for lang, rr in by_lang.items():
        ok = [r for r in rr if r.get("gate_reason") == "ok"]
        succ = [r for r in ok if r["reward"] > THRESHOLD]
        lang_asr[lang] = {
            "asr": len(succ) / max(1, len(ok)) * 100,
            "ok": len(ok),
            "succ": len(succ),
            "total": len(rr),
            "gated": len(rr) - len(ok),
            "avg_reward": sum(r["reward"] for r in ok) / max(1, len(ok)),
        }

    cat_asr = {}
    for cat, rr in by_cat.items():
        ok = [r for r in rr if r.get("gate_reason") == "ok"]
        succ = [r for r in ok if r["reward"] > THRESHOLD]
        cat_asr[cat] = {
            "asr": len(succ) / max(1, len(ok)) * 100,
            "ok": len(ok),
            "succ": len(succ),
            "total": len(rr),
        }

    overall_ok = [r for r in results if r.get("gate_reason") == "ok"]
    overall_succ = [r for r in overall_ok if r["reward"] > THRESHOLD]

    return {
        "lang_asr": lang_asr,
        "cat_asr": cat_asr,
        "overall_asr": len(overall_succ) / max(1, len(overall_ok)) * 100,
        "overall_ok": len(overall_ok),
        "overall_succ": len(overall_succ),
        "total": len(results),
        "rewards": [r["reward"] for r in results],
        "ok_rewards": [r["reward"] for r in overall_ok],
    }


def plot_all(baseline_metrics: dict, post_metrics: dict | None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    has_post = post_metrics is not None

    # Color scheme
    c_base = "#4A90D9"
    c_post = "#E74C3C"
    c_bg = "#F8F9FA"

    # ---- 1. ASR by Language ------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(c_bg)
    ax.set_facecolor(c_bg)

    langs = sorted(baseline_metrics["lang_asr"].keys())
    lang_labels = [LANG_LABELS.get(l, l) for l in langs]
    base_asrs = [baseline_metrics["lang_asr"].get(l, {}).get("asr", 0) for l in langs]

    x = np.arange(len(langs))
    width = 0.35 if has_post else 0.5

    bars1 = ax.bar(x - width / 2 if has_post else x, base_asrs, width,
                   label="Baseline (seed prompts)", color=c_base, edgecolor="white", linewidth=0.5)

    if has_post:
        post_asrs = [post_metrics["lang_asr"].get(l, {}).get("asr", 0) for l in langs]
        bars2 = ax.bar(x + width / 2, post_asrs, width,
                       label="Post-GRPO (learned attacks)", color=c_post, edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Language", fontsize=12, fontweight="bold")
    ax.set_ylabel("Attack Success Rate (%)", fontsize=12, fontweight="bold")
    ax.set_title("Multilingual Safety Gap: ASR by Language", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(lang_labels, fontsize=11)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)

    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1, f"{h:.0f}%",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    if has_post:
        for bar in bars2:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 1, f"{h:.0f}%",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    fig.savefig(OUT_DIR / "asr_by_language.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ asr_by_language.png")

    # ---- 2. ASR by Category ------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(c_bg)
    ax.set_facecolor(c_bg)

    cats = sorted(baseline_metrics["cat_asr"].keys())
    cat_labels = [c.replace("_", " ").title() for c in cats]
    base_cat_asrs = [baseline_metrics["cat_asr"].get(c, {}).get("asr", 0) for c in cats]

    x = np.arange(len(cats))

    bars1 = ax.bar(x - width / 2 if has_post else x, base_cat_asrs, width,
                   label="Baseline", color=c_base, edgecolor="white", linewidth=0.5)

    if has_post:
        post_cat_asrs = [post_metrics["cat_asr"].get(c, {}).get("asr", 0) for c in cats]
        bars2 = ax.bar(x + width / 2, post_cat_asrs, width,
                       label="Post-GRPO", color=c_post, edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Harm Category", fontsize=12, fontweight="bold")
    ax.set_ylabel("ASR (%)", fontsize=12, fontweight="bold")
    ax.set_title("Attack Success by Harm Category", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(cat_labels, fontsize=10, rotation=15, ha="right")
    ax.legend(fontsize=10)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "asr_by_category.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ asr_by_category.png")

    # ---- 3. Reward Distribution --------------------------------------------
    fig, axes = plt.subplots(1, 2 if has_post else 1, figsize=(12 if has_post else 7, 5))
    fig.patch.set_facecolor(c_bg)

    if not has_post:
        axes = [axes]

    ax = axes[0]
    ax.set_facecolor(c_bg)
    ax.hist(baseline_metrics["rewards"], bins=30, color=c_base, alpha=0.8, edgecolor="white")
    ax.set_title("Baseline Reward Distribution", fontsize=12, fontweight="bold")
    ax.set_xlabel("Reward")
    ax.set_ylabel("Count")
    ax.axvline(x=THRESHOLD, color="red", linestyle="--", alpha=0.7, label=f"Threshold ({THRESHOLD})")
    ax.legend()

    if has_post:
        ax = axes[1]
        ax.set_facecolor(c_bg)
        ax.hist(post_metrics["rewards"], bins=30, color=c_post, alpha=0.8, edgecolor="white")
        ax.set_title("Post-GRPO Reward Distribution", fontsize=12, fontweight="bold")
        ax.set_xlabel("Reward")
        ax.set_ylabel("Count")
        ax.axvline(x=THRESHOLD, color="red", linestyle="--", alpha=0.7, label=f"Threshold ({THRESHOLD})")
        ax.legend()

    plt.tight_layout()
    fig.savefig(OUT_DIR / "reward_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ reward_distribution.png")

    # ---- 4. Overall Summary Card -------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(c_bg)
    ax.set_facecolor(c_bg)
    ax.axis("off")

    summary_text = "POLYGLOT RED-TEAM RESULTS\n"
    summary_text += "=" * 40 + "\n\n"
    summary_text += f"Baseline ASR:  {baseline_metrics['overall_asr']:.1f}%  "
    summary_text += f"({baseline_metrics['overall_succ']}/{baseline_metrics['overall_ok']} gate-passed)\n"

    if has_post:
        summary_text += f"Post-GRPO ASR: {post_metrics['overall_asr']:.1f}%  "
        summary_text += f"({post_metrics['overall_succ']}/{post_metrics['overall_ok']} gate-passed)\n"
        delta = post_metrics['overall_asr'] - baseline_metrics['overall_asr']
        summary_text += f"\nImprovement:   {delta:+.1f} percentage points\n"

    summary_text += f"\nLanguages:     {', '.join(LANG_LABELS[l] for l in sorted(LANG_LABELS))}\n"
    summary_text += f"Categories:    {len(baseline_metrics['cat_asr'])}\n"
    summary_text += f"Baseline size: {baseline_metrics['total']} prompts\n"
    if has_post:
        summary_text += f"Post-GRPO:     {post_metrics['total']} prompts\n"

    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
            fontsize=12, verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9))

    fig.savefig(OUT_DIR / "summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ summary.png")

    # ---- 5. Text summary to file -------------------------------------------
    summary_path = OUT_DIR / "summary.md"
    with open(summary_path, "w") as f:
        f.write("# Polyglot Red-Team: Evaluation Results\n\n")
        f.write("## Overall\n\n")
        f.write(f"| Metric | Baseline | {'Post-GRPO |' if has_post else ''}\n")
        f.write(f"|--------|----------|{'----------|' if has_post else ''}\n")
        f.write(f"| ASR | {baseline_metrics['overall_asr']:.1f}% | "
                f"{post_metrics['overall_asr']:.1f}% |\n" if has_post else
                f"| ASR | {baseline_metrics['overall_asr']:.1f}% |\n")
        f.write(f"| Prompts evaluated | {baseline_metrics['total']} | "
                f"{post_metrics['total']} |\n" if has_post else
                f"| Prompts evaluated | {baseline_metrics['total']} |\n")

        f.write("\n## ASR by Language\n\n")
        f.write(f"| Language | Baseline ASR | {'Post-GRPO ASR |' if has_post else ''}\n")
        f.write(f"|----------|-------------|{'--------------|' if has_post else ''}\n")
        for lang in sorted(LANG_LABELS):
            b = baseline_metrics["lang_asr"].get(lang, {})
            line = f"| {LANG_LABELS[lang]} | {b.get('asr', 0):.1f}% |"
            if has_post:
                p = post_metrics["lang_asr"].get(lang, {})
                line += f" {p.get('asr', 0):.1f}% |"
            f.write(line + "\n")

        f.write("\n## ASR by Category\n\n")
        f.write(f"| Category | Baseline ASR | {'Post-GRPO ASR |' if has_post else ''}\n")
        f.write(f"|----------|-------------|{'--------------|' if has_post else ''}\n")
        for cat in sorted(baseline_metrics["cat_asr"]):
            b = baseline_metrics["cat_asr"][cat]
            line = f"| {cat.replace('_', ' ').title()} | {b['asr']:.1f}% |"
            if has_post:
                p = post_metrics["cat_asr"].get(cat, {})
                line += f" {p.get('asr', 0):.1f}% |"
            f.write(line + "\n")

    print(f"  ✓ summary.md")


def main():
    print("→ loading results")
    baseline = load_results(BASELINE_PATH)
    post_grpo = load_results(POST_PATH)

    if not baseline:
        print("ERROR: No baseline results found. Run baseline_eval.py first.")
        return

    print(f"  baseline: {len(baseline)} results")
    print(f"  post-grpo: {len(post_grpo)} results")

    baseline_metrics = compute_metrics(baseline)
    post_metrics = compute_metrics(post_grpo) if post_grpo else None

    # Print summary to stdout
    print(f"\n{'='*50}")
    print(f"BASELINE ASR: {baseline_metrics['overall_asr']:.1f}%")
    if post_metrics:
        print(f"POST-GRPO ASR: {post_metrics['overall_asr']:.1f}%")
        delta = post_metrics['overall_asr'] - baseline_metrics['overall_asr']
        print(f"DELTA: {delta:+.1f} pp")
    print(f"{'='*50}")

    print("\n→ generating plots")
    plot_all(baseline_metrics, post_metrics)

    print(f"\n✓ all outputs in {OUT_DIR}/")


if __name__ == "__main__":
    main()
