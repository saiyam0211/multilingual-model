"""Export true step-wise reward curve from W&B to plots/reward_curve.png.

Usage:
  source .venv/bin/activate
  export WANDB_API_KEY=...
  python scripts/export_wandb_reward_curve.py --run saiyam0211/polyglot-redteam/<run_id>
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        required=True,
        help="W&B run path, e.g. entity/project/run_id",
    )
    parser.add_argument(
        "--out",
        default="plots/reward_curve.png",
        help="Output plot path",
    )
    args = parser.parse_args()

    import numpy as np
    import pandas as pd
    import wandb
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    api = wandb.Api(timeout=60)
    run = api.run(args.run)

    # Pull full history; pick first usable reward-like column.
    hist = run.history(samples=50000, pandas=True)
    reward_cols = [
        "train/reward_mean",
        "reward",
        "rewards/mean",
        "metrics/reward",
        "reward_mean",
    ]
    col = next((c for c in reward_cols if c in hist.columns), None)
    if col is None:
        candidates = [c for c in hist.columns if "reward" in c.lower()]
        if not candidates:
            raise RuntimeError(
                "No reward column found in W&B history. Available cols: "
                + ", ".join(list(hist.columns)[:40])
            )
        col = candidates[0]

    df = hist[["_step", col]].dropna().copy()
    if df.empty:
        raise RuntimeError("Reward column exists but no non-null points found.")

    y = df[col].astype(float).to_numpy()
    x = df["_step"].astype(float).to_numpy()

    k = max(5, min(31, len(y) // 15 if len(y) >= 60 else 7))
    smooth = pd.Series(y).rolling(window=k, min_periods=1).mean().to_numpy()

    # Approximate CI band using rolling std.
    std = pd.Series(y).rolling(window=k, min_periods=2).std().fillna(0.0).to_numpy()
    lo = smooth - 1.96 * std / np.sqrt(np.maximum(1.0, k))
    hi = smooth + 1.96 * std / np.sqrt(np.maximum(1.0, k))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, y, alpha=0.25, color="#9ca3af", linewidth=1, label="raw reward")
    ax.plot(x, smooth, color="#2563eb", linewidth=2, label=f"rolling mean (k={k})")
    ax.fill_between(x, lo, hi, color="#93c5fd", alpha=0.3, label="approx 95% CI")
    ax.set_title("Training Reward Curve (W&B step-wise export)")
    ax.set_xlabel("Step")
    ax.set_ylabel("Reward")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"✓ wrote {out} from {args.run} using column `{col}` ({len(df)} points)")


if __name__ == "__main__":
    main()

