# Block 13 Final Checklist

## Automated checks (completed)

- [x] Required Block 12 artifacts exist (`plots/*`, `results/eval_*.jsonl`, `manual_audit.csv`)
- [x] README external links resolve (or are marked private)
- [x] Space health endpoint responds: `https://saiyam0211-polyglot-redteam.hf.space/api/health`
- [x] Local report generated: `results/block13_check.json`

## Manual checks (you need to do)

- [ ] Record final demo video and upload
- [ ] Paste video URL in README/submission form
- [ ] Fill and submit hackathon form
- [ ] Optional: make adapters public if you want judges to open model links directly

## W&B true reward curve replacement

Current `plots/reward_curve.png` is an eval-proxy curve.

To replace with true training step curve:

```bash
source .venv/bin/activate
export WANDB_API_KEY=...
python scripts/export_wandb_reward_curve.py --run <entity>/<project>/<run_id>
```

This overwrites `plots/reward_curve.png`.

