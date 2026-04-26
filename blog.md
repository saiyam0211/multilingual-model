---
title: "The Multilingual Safety Gap: Automating the Discovery of Cross-Lingual Vulnerabilities"
emoji: "🌍"
colorFrom: "blue"
colorTo: "purple"
sdk: "static"
---

# The Multilingual Safety Gap: Automating the Discovery of Cross-Lingual Vulnerabilities 🛡️

*How we used Reinforcement Learning and the OpenEnv framework to uncover over 1,000 confirmed safety vulnerabilities in Llama-3.1—specifically targeting Indian languages.*

---

Large Language Models are speaking more languages than ever before. But as their linguistic capabilities scale, a silent disparity is growing inside their safety mechanisms. 

When you ask a frontier model a harmful question in English—say, how to engineer a financial scam—it will almost certainly refuse. Its safety guardrails are robust, battle-tested, and deeply internalized. But what happens if you ask that exact same question in **Kannada, Marathi, or Bengali**?

The model still perfectly understands the request. And unfortunately, in many cases, it eagerly complies. 

This is the **cross-lingual safety gap**. 

For India's 700+ million internet users taking to the web in their native languages, this disparity means reduced safety guardrails and higher exposure to harmful content. We wanted to quantify exactly how wide this gap is—and we built **Polyglot Red-Teamer** to do it automatically.

## The Problem with Manual Red-Teaming

Traditionally, discovering these gaps requires manual red-teaming: hiring native speakers to sit down and try to "jailbreak" the model. This approach is slow, expensive, and fundamentally unscalable. 

We needed a way to automatically probe frontier models across multiple languages simultaneously, generating attacks, translating them, and verifying if a safety failure occurred. 

## Automating the Attack with OpenEnv & RL
![System Architecture](docs/diagrams/system_arch.png)

Instead of writing scripts, we trained an AI to do the attacking for us.

Using the **OpenEnv** framework, we designed an active environment where an "Attacker" model (`Qwen2.5-3B-Instruct`) generates adversarial prompts in six Indic languages: Hindi, Tamil, Bengali, Marathi, Telugu, and Kannada.

We then unleashed this attacker against a frozen target model (`Llama-3.1-8B-Instruct`). 

### The Differential Cross-Lingual Reward
To teach the attacker what constitutes a "successful" attack, we didn't just reward it for getting a harmful response. We used a strict **Differential Reward**:
1. The attacker proposes a harmful prompt in an Indic language.
2. We query the target model. If the target answers dangerously, the attacker gets a base reward.
3. We then translate that exact prompt into English and query the target again. 
4. **The Jackpot**: If the target *refuses* the English version but *answers* the Indic version, the attacker receives a massive reward multiplier. 

Through **Group Relative Policy Optimization (GRPO)** via Unsloth, the attacker rapidly learned the specific phrasing, dialectal nuances, and syntactic structures that bypassed Llama's Indic guardrails while triggering its English ones.

An ensemble of independent judges (Aya-Expanse, Qwen2.5) monitored the environment in real-time, validating responses via HF APIs.

## The Findings: 1,094 Confirmed Vulnerabilities
![The Safety Gap](docs/diagrams/safety_gap.png)

Our automated pipeline generated over 1,700 adversarial candidates and confirmed **1,094 structural safety gaps**.

The results exposed a stark contrast in safety generalization:
* In English, Llama-3.1 refused **73.3%** of our adversarial requests.
* In the exact same prompts translated to our six target languages, Llama-3.1 refused **0.0%** of the requests.

The highest success rates for the attacker occurred in **Kannada and Marathi**, where the model provided explicit advice on illicit finance, scam engineering, and privacy violations without triggering any internal safety flags. 

## Beyond Discovery: Why This Matters

Our goal is not to shame model providers, but to provide the tools necessary to close this gap. 

This project is a **vulnerability scanner that produces curated data for responsible disclosure**. By fully automating the pipeline, we reduced the cost of discovering a confirmed safety gap from roughly $50 per prompt (human red-teaming) to under $5 of compute for the entire 1,000+ prompt dataset.

Model providers can now take our open-source dataset and use it directly for targeted SFT (Supervised Fine-Tuning) or DPO (Direct Preference Optimization) to patch these specific cross-lingual failures.

## Explore the Data

We believe in making our findings completely transparent and interactable. 

1. **Test the Defender**: Visit our [Hugging Face Space](https://huggingface.co/spaces/Saiyam0211/polyglot-redteam) to use our live Defender Demo. Watch the unpatched Llama-3.1 answer the harmful request, while a patched `Llama-3.3-70B` correctly refuses it.
2. **Dive into the Dataset**: Browse through all 1,094 confirmed vulnerabilities right in the UI. 
3. **Reproduce our Work**: We've included a Colab notebook in our [GitHub Repository](https://github.com/saiyam0211/multilingual-model) that completely reproduces our Unsloth + OpenEnv GRPO training pipeline.

*True AI safety means safety in every language.*
