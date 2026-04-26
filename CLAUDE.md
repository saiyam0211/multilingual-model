# CLAUDE.md — Polyglot Red-Teamer Project Context

Last updated: 2026-04-26 (IST)

## 1) Project Goal (single source of truth)

Build an automated multilingual safety red-teaming pipeline that finds prompts where:

- English version is refused by target model
- Indic language version is answered by target model

Target model is frozen (`meta-llama/Llama-3.1-8B-Instruct`).
Attacker model is trained (`Qwen2.5-3B` adapters).

This project is a **vulnerability scanner**, not a defended model.

---

## 2) Current State Summary

- Core env + reward system implemented and deployed on HF Space.
- Frontend rebuilt to be dataset-centric (browse + live test + stats + about).
- Vulnerability dataset created and published.
- Block 12 artifacts generated and checked.
- Block 13 automated checks pass.
- SFT v2 + GRPO v2 training completed.
- Zero-shot v2 evaluation completed.

---

## 3) Published Assets

### Hugging Face Space
- Space repo: `Saiyam0211/polyglot-redteam`
- Public URL: https://huggingface.co/spaces/Saiyam0211/polyglot-redteam
- API health endpoint: `https://saiyam0211-polyglot-redteam.hf.space/api/health`

### Dataset
- Dataset repo: `Saiyam0211/polyglot-redteam-vulnerabilities`
- URL: https://huggingface.co/datasets/Saiyam0211/polyglot-redteam-vulnerabilities
- Key files:
  - `vulnerability_dataset.jsonl`
  - `vulnerability_dataset_v2.jsonl`
  - `vulnerability_dataset_final.jsonl`
  - `zero_shot_proof.jsonl`

### Model Adapters
- `Saiyam0211/polyglot-redteam-sft` (private)
- `Saiyam0211/polyglot-redteam-grpo` (private)
- `Saiyam0211/polyglot-redteam-sft-v2` (private)
- `Saiyam0211/polyglot-redteam-grpo-v2` (private)

---

## 4) Key Results (verified)

### Vulnerability dataset curation
- Total tested candidates: `1780`
- Confirmed safety-gap prompts: `1094`
- Discovery rate: `61.5%`

### Zero-shot autonomous generation (latest, v2)
- File: `zero_shot_proof.jsonl`
- Adapter: `Saiyam0211/polyglot-redteam-grpo-v2`
- Total generations: `144`
- Confirmed safety gaps: `17`
- Raw yield: `11.8%`

Language gaps in latest run:
- Bengali: 4
- Hindi: 1
- Kannada: 5
- Marathi: 6
- Telugu: 1
- Tamil: 0

Category gaps in latest run:
- illicit_finance: 5
- privacy_violation: 5
- scam_engineering: 5
- violence_enable: 2

---

## 5) Latest Training Runs (HF Jobs)

### SFT v2
- Job ID: `69ecf250d2c8bd8662bce031`
- Status: completed
- Output adapter: `Saiyam0211/polyglot-redteam-sft-v2`

### GRPO v2
- Final good run ID: `69ed06d2d2c8bd8662bce2b0`
- Runtime: ~`6.7h`
- Status in HF UI: timeout/error at end, but logs show full completion (`320/320`) and successful push.
- Output adapter: `Saiyam0211/polyglot-redteam-grpo-v2`
- Final env stats from logs: `calls=7680`, `hits=1004`, `ASR=13.1%`

### Zero-shot v2 eval
- Job ID: `69ed75bbd2c8bd8662bcee2d`
- Status: completed
- Output pushed to dataset (`zero_shot_proof.jsonl`)

---

## 6) Block 12 / Block 13 Artifacts

### Block 12
- `results/eval_baseline.jsonl`
- `results/eval_trained.jsonl`
- `results/asr_matrix.json`
- `results/manual_audit.csv` (50 rows)
- `plots/asr_before_after.png`
- `plots/category_heatmap.png`
- `plots/reward_curve.png`
- `plots/attack_examples.png`

### Block 13 checks
- `scripts/block13_check.py`
- `results/block13_check.json` => links/health checks pass
- `results/submission_checklist.md`

---

## 7) Important Fixes Already Applied

1. Space API `/step` 500 loop fixed by lazy target initialization in `server.py`.
2. Gradio frontend reframed to avoid misleading “our model defends” implication.
3. Dataset loading in Space made robust (runtime fetch pattern when needed).
4. GRPO config made env-configurable for controlled v2 sweeps.
5. Zero-shot pipeline checkpoint uploads added to avoid total loss on timeout.

---

## 8) W&B Status

- W&B entity exists: `saiyam0211-defellix`
- Project observed: `MultiLingual Model`
- Current run listing showed `runs=0` for that project during checks.
- Therefore no true historical training curve available from old runs.
- `plots/reward_curve.png` currently uses eval-proxy curve.
- Script exists for future true export if run logging exists:
  - `scripts/export_wandb_reward_curve.py`

---

## 9) Current Known Gaps / Risks

1. Zero-shot raw yield remains modest (~12% range).
2. v2 did not show clear large yield lift on the sampled 144 generations.
3. Most robust value remains the curated 1094 confirmed gaps dataset.
4. If judges ask “autonomous power,” present zero-shot as evidence, but position dataset pipeline as primary contribution.

---

## 10) Recommended Next Actions (if continuing)

1. Produce `results/yield_comparison.json` + small chart for `grpo` vs `grpo-v2`.
2. Update README with explicit “no significant v2 yield lift” if applicable.
3. Finalize demo script + submission form text.
4. Optionally run a short targeted GRPO v3 (reward rebalancing + stricter language/gibberish signal) if time permits.

---

## 11) Quick Commands

### Check space health
```bash
curl https://saiyam0211-polyglot-redteam.hf.space/api/health
```

### Run Block 13 checks
```bash
source .venv/bin/activate
python scripts/block13_check.py
```

### Export true W&B reward curve (if run exists)
```bash
source .venv/bin/activate
export WANDB_API_KEY=...
python scripts/export_wandb_reward_curve.py --run "<entity>/<project>/<run_id>"
```

