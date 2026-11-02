# Run: codepraxis-muse-v2

User (2026-11-02): "lets skip DPO then. if u have any idea to get numbers better lets do it too."

Second SFT from base Muse Glimmer 30B (serverless can't warm-start from v1's adapter), three changes vs v1:
1. LR: 20-step linear warmup, then linear decay from 1e-4 to 0 (v1: constant 1e-4; loss flat after step ~250).
2. Data: train.jsonl + val.jsonl (val was unused in v1).
3. Assess turns that close a stage with a grade (1,192) appear twice (v1 regression: same grade 94% -> 82%).
File: data/export/train_v2.jsonl, 10,235 rows, shuffled (seed 0). Everything else as v1: rank 16, alpha 32, batch 16, 1 epoch, max_seq_len 32768, n_eval 40.
Script: scripts/muse_sft_v2.py (cookbook starter + LR schedule). Training $0 (promotion).

## Log
- Launched 15:05 PST.
- Stopped 15:07 PST before any training session opened (still tokenising): no time for a 1.7 h run before the presentation; a shorter run on less data would start from base and likely land below v1. v1 stays the result.
- The session ts-9c801414... had opened at the moment of the kill (0 steps trained); deleted to free the slot.
