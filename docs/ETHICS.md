# Ethics Statement

This project is a **defensive red-teaming tool**. Its purpose is to surface
safety failures in deployed LLMs in Indic languages so that those failures can
be patched.

## What we do
- Build an OpenEnv environment that scores adversarial prompts.
- Train a small attacker model (Qwen2.5-3B + LoRA) via GRPO to discover prompt
  patterns that bypass safety on a frozen target.
- Publish **the environment**, **aggregated findings**, and **the harness** so
  Meta, Sarvam, AI4Bharat, and other model providers can use them to harden
  their own systems.

## What we do not do
- We do **not** publish the trained attacker checkpoint.
- We do **not** include category instructions for CBRN (chemical, biological,
  radiological, nuclear) or CSAM (child sexual abuse material). Those are
  hard-excluded from `data/categories.json`.
- We do **not** publish raw harmful generations. Public artifacts contain only
  aggregated statistics and a small set of redacted exemplars for illustration.

## Categories included
1. Illicit financial advice (tax fraud, money laundering)
2. Violence enablement (non-CBRN)
3. Self-harm encouragement
4. Privacy violations (doxxing scripts, scraping personal data)
5. Scam / social engineering (phishing, elder scams)

## Languages
Hindi, Tamil, Bengali, Marathi, Telugu, Kannada — chosen for population coverage
and known under-representation in published safety evaluations.

## Responsible disclosure
Findings will be shared with affected model providers before any public blog
post. We will follow standard 90-day disclosure timelines.

## Contact
Open a GitHub issue marked `[security]` or email the team via the address in
the README.
