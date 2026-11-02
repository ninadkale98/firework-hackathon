# Methods to revise before the hackathon

Ordered by how much we'll use each one. ★ = we definitely use it tomorrow.

## A. Training

| # | Method | What it is | Why we use it / what to know |
|---|---|---|---|
| 1 ★ | **Knowledge distillation (sequence-level)** | A small "student" learns to copy a big "teacher" by training on the teacher's outputs | Our whole approach: the open teacher's answers → Qwen3.5-9B. The student's ceiling is the teacher |
| 2 ★ | **Supervised fine-tuning (SFT)** | Train on (input → target output) pairs with next-token loss | Fireworks `sftj`. Know: **loss masking** (only the assistant reply is trained; `weight: 0` on other messages), and the chat template |
| 3 ★ | **LoRA** | Train small low-rank adapter matrices instead of all the weights | Knobs: **rank** (8–64; we plan 16), alpha, target modules, learning rate (higher than full fine-tuning). Cheap, fast, one base model can serve many adapters |
| 4 ★ | **Hyperparameters and overfitting** | Epochs, learning rate, batch size, max context length | Watch validation loss vs training loss. 1–3 epochs for ~8K examples. Examples longer than the max context are cut off |
| 5 ★ | **Structured output / constrained decoding** | Forcing output to match a JSON schema or grammar while it's generated | Our moves are JSON. Fine-tuning makes the format reliable; constrained decoding guarantees it. On Fireworks, json_schema turns off reasoning |
| 6 | **Data curation: rejection sampling / filtering** | Keep only teacher outputs that pass checks (judge, rules) | Our GPT judge + leak check drop bad teacher answers before training |
| 7 | **Coverage / stratified sampling, deduplication** | Make sure every situation is represented and nothing is over-repeated | Our bucket and all-pairs coverage, the 8% per-question cap, near-duplicate filter |
| 8 | **DPO (Direct Preference Optimization)** | Train on (chosen, rejected) pairs so the model prefers the better answer; knob **β** | Stretch goal: chosen = teacher answer, rejected = a leaking or wrong-move answer. Fireworks supports it ($1/M tokens). Also read about **ORPO** (combines SFT + preference) |
| 9 | **RFT / RL with verifiable rewards (GRPO)** | Reinforcement learning where a program scores each output | Our rule checks (legal move, no leak, grade match) are natural rewards. Fireworks has RFT + Evaluators (Eval Protocol). Probably no time, but good to know for the pitch |
| 10 | **Full fine-tuning vs LoRA vs QLoRA** | Train all weights / adapters / adapters on a 4-bit base | Know the trade-offs: quality, cost, memory |

## B. Evaluation

| # | Method | What to know |
|---|---|---|
| 11 ★ | **Held-out test set, data leakage** | Split by question so test questions are never trained on; freeze the test set |
| 12 ★ | **LLM-as-judge** | Rubric/criteria scoring vs **pairwise** (A vs B → win rate). Known biases: **position bias** (randomise order), **length bias**, **self-preference** (a judge likes its own family's style) |
| 13 ★ | **Judge calibration** | Compare the judge with ~30 hand labels; agreement % or **Cohen's kappa** |
| 14 ★ | **Confidence intervals (bootstrap)** | Resample test turns → 95% interval on each score; overlapping intervals = no real difference |
| 15 | **Robustness via paired perturbations** | Same answer, clean vs messy → does the decision change? |
| 16 | **Slice analysis** | Scores per bucket (clock, candidate type, task) to find where the model is weak, then target more data there |

## C. Serving and efficiency (the "dimension we push")

| # | Method | What to know |
|---|---|---|
| 17 ★ | **Latency metrics** | TTFT (time to first token), tokens/s, p50/p95 per turn, cost per turn / per interview |
| 18 ★ | **Quantization** | FP16 → **FP8** / INT8 / INT4: faster and cheaper, small quality loss. Fireworks serves FP8 |
| 19 | **Prefix / prompt caching, KV cache** | Repeated prompt starts are computed once; our prompts put the stable part first |
| 20 | **Speculative decoding** | A small draft model proposes tokens and the big model verifies them; faster with the same output |
| 21 | **Model routing / cascades** | The small model takes most turns and escalates uncertain ones to a bigger model. Stretch goal |
| 22 | **Dedicated vs serverless serving, autoscaling, cold starts** | Fine-tuned models need a dedicated deployment (H100 $8/h); scale-to-zero means a cold start |

## D. Qwen3.5 specifics
- **Chat template** and **thinking mode** (`enable_thinking` / `/no_think`): we train and serve with thinking **off**, so outputs match the targets.
- Context: Qwen3.5-9B trains up to 131k on Fireworks; we use ~32k.

## E. Our pipeline in one line
GPT-mini plays candidates → the open teacher interviews (training) / Opus interviews (test) → the GPT judge filters → SFT LoRA on Qwen3.5-9B → score on the frozen test set vs Opus, Haiku, GPT and base → charts: quality vs $, quality vs latency.
