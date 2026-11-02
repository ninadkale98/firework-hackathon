# Run: codepraxis-interviewer-sft-v1

entry_skill: configure
approved: 2026-11-02, user: "1. lets go with 2 epoch 2. DeepSeek"

| | |
|---|---|
| account | ninadnar98-z0upz59iv |
| method / surface | Managed LoRA SFT (Fireworks SDK) |
| base model | accounts/fireworks/models/qwen3p5-9b |
| training data | codepraxis-distill-train-v1 (8,361 rows, READY) |
| evaluation data | codepraxis-distill-val-v1 (682 rows, READY) |
| output model | accounts/ninadnar98-z0upz59iv/models/codepraxis-interviewer-v1 |
| lora_rank / epochs / max_context_length | 16 / 2 / 32768 |
| learning rate, batch size, warmup | platform defaults (read back after create) |
| cost estimate | ~63–80M tokens/epoch × 2 × $0.50/M ≈ $63–80 (fireworks.ai/pricing, fetched 2026-11-02) |
| monitoring | poll every ~3 min; no-progress timeout 30 min |
| final evaluation judge | DeepSeek V4.1 Flash (Opus and base Qwen re-judged with it for a like-for-like comparison) |
| deployment | not approved; separate decision |

## Result (2026-11-02 13:58)

- State: JOB_STATE_COMPLETED, 524 steps, 0 failed rows, 0 length-capped rows.
- Output: `accounts/ninadnar98-z0upz59iv/models/codepraxis-interviewer-v1` (HF_PEFT_ADDON, READY, r=16, base qwen3p5-9b).
- Eval loss: 1.588 (step 0) -> 0.874 (262, end epoch 1) -> 0.848 (524, end epoch 2). Train loss last 10 steps: 0.828.
- Render: `qwen3_5_interleaved`; target is `<think>\n\n</think>\n\n{"move_object":...}<|im_end|>` (empty think block), so serve with thinking off.
- Not yet deployed (needs separate approval).
