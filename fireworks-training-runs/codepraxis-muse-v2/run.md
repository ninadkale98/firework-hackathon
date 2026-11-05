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
- User: "sure go on" to a shorter run with checkpoints. Relaunched 15:09 PST: --n-train 4800 (first 4,800 rows of the shuffled v2 mix) -> 300 steps, LR warmup 20 then linear decay to 0 over 300 steps, checkpoint saved and promoted every 100 steps (models codepraxis-muse-v2-step100, -step200; final promoted as muse-starter-sl-codepraxis-muse-v2), n_eval 20. Rationale: v1's loss plateaued by step ~250.
- Session ts-d6f597ec4c144e12afc4b95ea97be740 (run-0dd50c3f011546be9c7e24aa938a7851) opened 15:11 PST.
- 16:05 step-100 checkpoint promoted: accounts/vishakha-d933-0nmp64/models/codepraxis-muse-v2-step100
- 16:06 Deployment muse-v2-step100 (1x H200) created for the 300-turn eval.
- 16:08 User: 'stop v2, evaluate the prompt'. Training stopped at step 104 (session deleted), v2 eval cancelled, deployment muse-v2-step100 deleted. Model codepraxis-muse-v2-step100 kept, not evaluated.
- 16:12 User deployed the step-100 model as j6zbfgxf; benchmarking it on the 300 turns (not deleting it).
