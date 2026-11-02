# codepraxis-interviewer-dpo-v1

Approval (user, 2026-11-02 ~14:20): "start DPO" (after: "Train on the 1,223 pairs, starting from the tuned model. Managed DPO on Fireworks, 1 epoch.")

| | |
|---|---|
| Surface | Managed DPO (SDK `dpo_jobs.create`), account ninadnar98-z0upz59iv |
| Start from | `accounts/ninadnar98-z0upz59iv/models/codepraxis-interviewer-v1` (SFT LoRA r=16, warm start) |
| Dataset | `codepraxis-dpo-opta-v1`: 1,223 pairs, system+user -> preferred (DeepSeek teacher, judge-passed) vs non-preferred (base Qwen, judged bad: much_worse 872, fails_safety 180, gives_answer_away 167, invalid 4) |
| Loss | method DPO (explicit: the API defaults to GRPO), beta 0.1 |
| Epochs | 1; max context 32768; intermediate sampler checkpoints every 10 steps |
| LR, batch | platform-resolved, read back after create |
| Output | `codepraxis-interviewer-dpo-v1` |
| Cost | ~20.8M tokens (2P+C+R) x 1 epoch x $1.00/M (LoRA DPO <=16B, fireworks.ai/pricing fetched 2026-11-02) = ~$21 (range $15-27) |
| Eval | same 300-turn shared sample, DeepSeek judge, vs SFT and base |
