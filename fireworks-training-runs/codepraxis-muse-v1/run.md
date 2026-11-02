# Run: codepraxis-muse-v1

entry_skill: configure
Approved by user, 2026-11-02: "i have 50$ credit, lets do it"

| | |
|---|---|
| Account | vishakha-d933-0nmp64 |
| Method | Managed LoRA SFT (Fireworks Python SDK, fireworks-ai 1.2.30) |
| Base model | accounts/fireworks/models/muse-glimmer-30b |
| Train dataset | codepraxis-distill-train (data/export/train.jsonl, 8,361 rows) |
| Eval dataset | codepraxis-distill-val (data/export/val.jsonl, 682 rows) |
| Job ID / output model | codepraxis-muse-v1 |
| LoRA rank | 16 |
| Epochs | 1 |
| Max context length | 32768 |
| Learning rate, batch size | platform defaults (read back after create) |
| Cost | $0 training (Muse Glimmer LoRA SFT promotion, jobs started before 2026-12-01T00:00:00-08:00). Deployment and evaluation priced separately, need separate approval. |
| Monitoring | render samples after create; progress % and eval loss until terminal |

## Log
- Managed SFT dataset upload rejected: "dataset uploads are temporarily disabled". Empty dataset record codepraxis-distill-train left in place.
- Switched to Serverless Training API (cookbook 57968a2, muse_glimmer_starter.py). lr 1e-4, batch 16, rank 16, alpha 32, max_seq_len 32768, 1 epoch.
- User chose "Pilot first": 800 train rows, 20 holdout, --no-promote.
- Pilot session opened: ts-b4c8d59879bc4b8bb7abf4c1a3f91f58 (run-624da3a5730145b3accaa307f14cebeb). Started 2026-11-02 11:07 PST.
- Pilot done (exit 0), ~10 s/step, 50 steps. Loss 1.437 -> 0.727. Not promoted.
  Holdout 20: base valid JSON 0/20, 6 truncated; tuned valid JSON 16/20, 0 truncated, same move as gold 11/20. Exact match 0 for both (whole-object compare).
  Eval sampling: 274,508 prompt + 38,826 sampled tokens (billed separately).
- Correction: the 4 "invalid JSON" tuned answers are cut at 2,000 chars by the starter's predictions writer (answer_text[:2000]); all terminated cleanly. Tuned valid JSON is likely 20/20.
- User: "first one run, then we monitor and see if we can run another and maybe a LR sweep + maybe Rank adjustment".
  Full run launched: run-id codepraxis-muse-v1, all train rows (8,321 after 40 holdout), 1 epoch, lr 1e-4, batch 16, rank 16, max_seq_len 32768, n_eval 40, promote at end.
- Full run attempt 1 failed at create: 429 'maximum concurrent serverless training runs for muse-glimmer-30b: in use 1, limit 1' (pilot session holding slot). Account limit is 1 concurrent Muse Glimmer serverless run. Deleted pilot session ts-b4c8d598... (unpromoted pilot checkpoints only) to free the slot.
- Full run attempt 2: session ts-04c8fa923655443fb647f006662615e7 (run-82ce51075453463a97bb8bd7b1d0a06b) opened 11:28 PST.
- Full run COMPLETED (521 steps, ~9.5 s/step, ~11:30-13:02). Loss 1.329 (step 1) -> ~0.78 (50-step avg, plateau from step ~250).
  Holdout 40: base 0/40 move_object, 0/40 same move, 7 unfinished; tuned 40/40 move_object, 33/40 same move as gold, 0 unfinished.
  Eval sampling: 562,444 prompt + 72,177 sampled tokens (billed separately). Training $0.
  Saved model: accounts/vishakha-d933-0nmp64/models/muse-starter-sl-codepraxis-muse-v1 (HF_PEFT_ADDON, READY, r=16, targets q/k/v/o/gate/up/down).
  Training session ts-04c8fa92... deleted after promotion to free the account's single Muse Glimmer slot. Nothing billable running; no deployment.
- Eval: tuned + base Muse answered the 300-turn shared sample (scripts/run_tuned.py). Base deployment muse-glimmer-base-eval deleted after the run; tuned eqb76xt9 parked at 0 replicas. Judge: DeepSeek V4.1 Flash (scripts/judge_ds.py, kinds *_ds).
