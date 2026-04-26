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

### SFT v3 ✅
- Job ID: `69ed95dad2c8bd8662bcf240`
- Status: **completed**
- Dataset: 356 training examples from confirmed gaps + seeds (hash-locked splits)
- Post-SFT refusal rate: **0/10 = 0%** (within acceptable range <30%)
- Output adapter: `Saiyam0211/polyglot-redteam-sft-v3`

### GRPO v3 🟡 (in progress)
- Job ID: `69ed9bf9d70108f37acdf91b`
- Status: **running** on 1x NVIDIA L4
- Config: 1500 steps, 8 generations, grad_accum=8, lr=1e-6
- Base: `Saiyam0211/polyglot-redteam-sft-v3`
- Features: multiplicative reward, cross-lingual differential, curriculum sampling
- Expected output: `Saiyam0211/polyglot-redteam-grpo-v3`
- Estimated runtime: ~10-12h on L4

---

## 6) Phase F — Adjacent Improvements (implemented)

### Defender-Loop Demo (F2)
- Module: `src/polyglot_redteam/defender_demo.py`
- Frontend tab: "🛡️ Defender Loop"
- Compares Llama-3.1-8B (unpatched) vs Llama-3.3-70B (defended) side-by-side
- Shows "we found AND patched" narrative for judges

### Multi-Turn Red-Teaming (F1)
- Module: `src/polyglot_redteam/multi_turn.py`
- Frontend tab: "🔄 Multi-Turn Attack"
- Conversational attacks up to 5 turns with follow-up generation
- Eval script: `scripts/eval_multi_turn.py`

---

## 7) Block 12 / Block 13 Artifacts

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

## 8) Important Fixes Already Applied

1. Space API `/step` 500 loop fixed by lazy target initialization in `server.py`.
2. Gradio frontend reframed to avoid misleading "our model defends" implication.
3. Dataset loading in Space made robust (runtime fetch pattern when needed).
4. GRPO config made env-configurable for controlled v2 sweeps.
5. Zero-shot pipeline checkpoint uploads added to avoid total loss on timeout.
6. Fixed `.gitignore` blocking `*.jsonl` data files from reaching HF Jobs (force-added).
7. Fixed `build_sft_v3.py` to correctly parse `language` key from vulnerability dataset.

---

## 9) W&B Status

- W&B entity exists: `saiyam0211-defellix`
- Project observed: `MultiLingual Model`
- GRPO v3 configured with WANDB logging enabled
- Script exists for export: `scripts/export_wandb_reward_curve.py`

---

## 10) Current Known Gaps / Risks

1. Zero-shot raw yield remains modest (~12% range with v2).
2. v3 reward stack is significantly more rigorous — multiplicative with cross-lingual differential.
3. Kill criteria at steps 200/500/800 will auto-flag if GRPO v3 is not converging.
4. Most robust value remains the curated 1094 confirmed gaps dataset.

---

## 11) Pipeline: What Runs Next

```
SFT v3 ✅ → GRPO v3 🟡 → Eval v3 → Manual Audit → Ship
                         ↓
                   Phase F already built (Defender + Multi-Turn)
```

1. Wait for GRPO v3 to finish (~10-12h on L4)
2. Launch eval v3: `hf jobs run --flavor l4x1 --secrets HF_TOKEN -d ghcr.io/unslothai/unsloth:latest bash -c "git clone https://github.com/saiyam0211/multilingual-model.git && cd multilingual-model && bash scripts/run_eval_v3_job.sh"`
3. Review manual audit CSV (`results/manual_audit_v3.csv`)
4. Update HF Space with final adapter + Phase F tabs
5. Ship

---

## 12) Quick Commands

### Check space health
```bash
curl https://saiyam0211-polyglot-redteam.hf.space/api/health
```

### Run Block 13 checks
```bash
source .venv/bin/activate
python scripts/block13_check.py
```

### Monitor GRPO v3 job
```bash
source .venv/bin/activate && hf jobs logs 69ed9bf9d70108f37acdf91b --tail 30
```
