# Muse Glimmer 30B: LoRA SFT and evaluation (2026-11-02)

A second student, trained alongside the Qwen3.5-9B run: **Muse Glimmer 30B + LoRA**, fine-tuned on the same training export and evaluated on the same frozen test set with the leaderboard's metrics. This file records what was done, how, the results, and how to reproduce or extend it.

**Status: final.** Training done, model saved, evaluation complete on the 300-turn shared sample: every valid answer of every model judged (base Muse's 16 invalid answers have no move and are not judged, as on the leaderboard). DPO and a v2 SFT were investigated but not run; see [Open items](#7-open-items).

## 1. Results in one table

300-turn shared sample (`runs.stratified(…, 300)`, the leaderboard sample, `in_shared_sample = true` in `test_eval.jsonl`). Metric definitions: [EVALUATION.md §3](EVALUATION.md). Judge columns use **DeepSeek V4.1 Flash** as judge (no OpenAI credit; see §5.3).

| Metric | Opus 5 (ref) | DeepSeek V4.1 Flash (teacher) | Qwen3.5-9B base | **Muse base** | **Muse tuned** |
|---|---|---|---|---|---|
| Quality | 95% | 95% | 82% | 92% | **94%** |
| Valid output | 99% | 100% | 99% | 95% | **100%** |
| Rules kept | 98% | 99% | 85% | 96% | **98%** |
| Gives answer away | 1% | 0% | 10% | 0% | **1%** |
| Resists fishing / injection | 97% (n=71) | 97% (n=70) | 73% (n=71) | 85% (n=67) | **97%** (n=71) |
| Reveals answer when moving on | 0% | 0% | 11% | 3% | **0%** |
| Same move as Opus | – | 64% | 42% | 45% | **65%** |
| Same grade as Opus (Assess) | – | 97% (n=38) | 68% (n=37) | 94% (n=31) | **82%** (n=40) |
| Judge 1–5 | 4.32 | 4.36 | 3.16 | 4.00 | **4.25** |
| Judge prefers it or ties vs Opus | – | 58% | 7% | 24% | **41%** |
| Latency p50 | 9.2 s | 12.0 s | 5.1 s | 18.0 s | **8.0 s** |
| Median output tokens | – | 1,354 | 1,434 | 1,720 | **278** |

Judge columns cover all 300 turns for every model except base Muse (284: its 16 invalid answers aren't judged) and DeepSeek (299). Code-scored columns (valid output, same move, same grade, latency, tokens) are on all 300. Full per-mode tables with 95% bootstrap intervals and exact n: [results/muse_eval/REPORT_deepseek_judge.md](results/muse_eval/REPORT_deepseek_judge.md). Raw numbers for charts: [board_deepseek_judge.json](results/muse_eval/board_deepseek_judge.json), loss curve [training_loss.csv](results/muse_eval/training_loss.csv).

**Base → tuned, paired on the same turns (exact McNemar):**

| Outcome | base only | tuned only | n | p |
|---|---|---|---|---|
| Valid output | 0 | 16 | 300 | 3e-05 |
| Same move as Opus | 38 | 99 | 298 | 2e-07 |
| Judge score ≥ 4 | 13 | 55 | 284 | 3e-07 |
| Did not give answer away | 4 | 1 | 284 | 0.38 (n.s.) |

### What it means
- **Fine-tuning worked.** Same move 45% → 65% (interview 34% → 65%, assess 54% → 78%), valid output 95% → 100%, judge 4.00 → 4.25, win/tie vs Opus 24% → 41%, resists fishing / injection 85% → 97%. The paired tests are significant.
- **The student matches its teacher on move choice** (65% vs 64%) and on not leaking (1% vs 0%), at 1/5 of the output length and 2/3 of the latency.
- **Still behind the teacher on judged quality:** win/tie vs Opus 41% vs 58%, judge 4.25 vs 4.36.
- **One regression: Assess grading.** Same grade as Opus 94% (base, n=31) → 82% (tuned, n=40); teacher 97%. Small n, and the tuned model closes more stages, but this is the clearest next target (more Assess stage-closing rows, or DPO on grading).
- Versus the other student, **Qwen3.5-9B base**, tuned Muse is far ahead on every column; compare with Qwen3.5-9B *tuned* once that run is evaluated with the same judge.

## 2. The model

| | |
|---|---|
| Fireworks account | `vishakha-d933-0nmp64` (not the account the Qwen run uses) |
| Model | `accounts/vishakha-d933-0nmp64/models/muse-starter-sl-codepraxis-muse-v1` (`HF_PEFT_ADDON`, READY) |
| Base | `accounts/fireworks/models/muse-glimmer-30b` (dense, 29.8B, BF16) |
| Adapter | LoRA r=16, alpha=32, modules q/k/v/o/gate/up/down (the unembed was requested but not in the saved adapter) |
| Deployment | `eqb76xt9`: 1× H200 141GB, BF16, LoRA merged at deploy. **Parked at 0 replicas** (min 0 / max 0). To use it, raise `max_replica_count` to 1 |
| Calling it | model `accounts/vishakha-d933-0nmp64/models/muse-starter-sl-codepraxis-muse-v1#accounts/vishakha-d933-0nmp64/deployments/eqb76xt9` on `https://api.fireworks.ai/inference/v1/chat/completions`. Text only; thinking off (trained without it) |
| In code | `distill/models.py`: `muse-glimmer-tuned` and `muse-glimmer-base` |

## 3. How it was trained

### 3.1 Why Muse Glimmer, and why the Serverless Training API
- Muse Glimmer 30B LoRA SFT is **free** for jobs started before 2026-12-01 00:00 PT (Fireworks promotion; SFT only, not DPO/RL/deployments). Qwen3.5-9B SFT is $0.50/1M tokens (~$70 for this data).
- **Dataset uploads are disabled on this account too** (`PermissionDeniedError: dataset uploads are temporarily disabled`), so Managed SFT was impossible. The Serverless Training API trains from local files: a loop on the laptop sends batches, Fireworks GPUs do forward/backward.
- Script: the `fw-ai/cookbook` starter `training/examples/sft/muse_glimmer_starter.py` (cookbook commit `57968a2`, `fireworks-ai` 1.2.30), installed through the `fireworks-training` Claude Code plugin.

### 3.2 Data
`data/export/train.jsonl` from the data repo, unchanged: 8,361 rows of `system` / `user` / `assistant`, teacher DeepSeek V4.1 Flash, filtered by the GPT-5.4 judge. Checked before training: 0 bad JSON, 0 empty messages, 0 duplicates, 0 `<think>` tags; with the Muse tokenizer, median ~6.7k tokens, max ~27k, all ≤ 32,768. The starter holds out 40 random rows for a base-vs-tuned check and trains on **8,321**. `val.jsonl` was not used by the trainer.

### 3.3 Configuration
| | |
|---|---|
| Method | LoRA SFT (not QLoRA: BF16 base, no quantisation) |
| Loss | cross-entropy on the **final assistant turn only** (`TrainOnWhat.LAST_ASSISTANT_TURN`); system prompt and user message are context |
| LoRA | rank 16, alpha 32, attention + MLP |
| Optimiser | Adam β1 0.9, β2 0.999, ε 1e-8, no weight decay, grad clip 1.0 |
| Learning rate | 1e-4 **constant** (no warmup, no decay) |
| Batch | 16 rows per step → 521 steps for 1 epoch |
| Epochs | 1 |
| Max sequence length | 32,768 |
| Renderer / tokenizer | `muse_glimmer` / `meta-models/Muse-Glimmer-30B` |

### 3.4 Runs
1. **Pilot** (800 rows, 50 steps, not saved): loss 1.44 → 0.73; on 20 held-out rows base produced 0/20 `move_object` answers (6 hit the length limit), tuned 20/20.
2. **Full run** `codepraxis-muse-v1`: 521 steps, ~9.5 s/step, 11:30–13:02 local. Loss 1.33 (step 1) → ~0.78 (50-step average), **flat from about step 250** (constant LR). On its 40 held-out rows: tuned 40/40 `move_object`, 33/40 same move as the reference, 0 unfinished; base 0/40, 7 unfinished.
3. Adapter promoted to the model above; training session deleted to free the slot.

Logs and curves: [fireworks-training-runs/codepraxis-muse-v1/](fireworks-training-runs/codepraxis-muse-v1/) (`run.md` is the decision log, `full/stdout.log` every step, `dashboard.html` the live loss page; serve the folder with `python3 -m http.server 8765 --bind 127.0.0.1`).

### 3.5 Things that bit, and the fix
| Problem | Fix |
|---|---|
| Dataset upload disabled | Serverless Training API from local JSONL |
| `429 maximum concurrent serverless training runs for muse-glimmer-30b: in use 1, limit 1` | The account allows **one** Muse Glimmer training session at a time, and a finished session holds the slot until it expires. Delete it after promoting: `FireworksClient.delete_training_session(name)`. So no parallel hyper-parameter sweeps on this account |
| `metrics.jsonl` stays empty during training | The starter buffers it; read `stdout.log` (flushed every step) |
| Deployment create: `payment method is required` | A card must be on the account even with credits |
| Deployment delete: `received inference requests in the last hour` | `deployments.delete(..., ignore_checks=True)` |
| Muse Glimmer has no serverless inference (retired 2026-09-25) | Base and tuned both need on-demand deployments for evaluation |

## 4. Cost
| Item | Cost |
|---|---|
| Training (pilot + full) | $0 (promotion) |
| In-session eval sampling (pilot 20 + full 40 rows, base and tuned) | ~0.84M prompt + 0.11M sampled tokens at serverless rates (~$2) |
| Tuned deployment `eqb76xt9` (H200, $8/h) | while up for the 300-turn run; now parked at 0 |
| Base deployment `muse-glimmer-base-eval` (H200, $8/h) | ~35 min, then deleted |
| DeepSeek judge (Fireworks serverless) | $13.10 for all rows (~$0.004 per call) |

## 5. How it was evaluated

### 5.1 Answers (no backend needed)
The CodePraxis backend isn't on this machine, so `distill.runs run` (which imports it) was replaced by [scripts/run_tuned.py](scripts/run_tuned.py): same system prompt (blocks joined with blank lines), same user message, strict JSON-schema output with the turn's schema wrapped as `{"move_object": …}`, temperature 0, max 4,096 output tokens, and rows written to `runs` in the same shape as `distill.runs`.

Checked against `test_eval.jsonl` (pushed to the data repo the same afternoon): the 300 sample ids are identical; system and user messages are byte-identical on all 1,525 turns; the schema has the same constraints, but `test_eval.jsonl` additionally states `"type"` on `const`/`enum` fields (`move`, `hint_id`, `probe_id`, `level`). That doesn't change what's allowed. Next runs should read `test_eval.jsonl`'s `response_format` directly.

```bash
python scripts/run_tuned.py --name muse-glimmer-tuned --sample 300 --concurrency 8
python scripts/run_tuned.py --name muse-glimmer-base  --sample 300 --concurrency 8
python -m distill.runs score --split test        # code-scored columns, free
```

Base Muse writes long answers (median 1,720 tokens, 16/300 hit the 4,096 cap); those count as invalid output, as for any model.

### 5.2 Judge
[scripts/judge_ds.py](scripts/judge_ds.py) calls `distill.judge.criteria_task` / `pairwise_task` / `safety_task` unchanged (same prompts, criteria, schemas, randomised pairwise order) with model `accounts/fireworks/models/deepseek-v4p1-flash` through the Fireworks endpoint. Verdicts are stored as **`judge_ds`, `pairwise_ds`, `safety_ds`**, so the GPT-5.4 verdicts (`judge`, `pairwise`, `safety`) are untouched. It skips rows already judged, so rerunning fills gaps (DeepSeek serverless returns 429s above ~16 concurrent calls).

Judged with DeepSeek: Opus (reference), DeepSeek, Qwen3.5-9B base, Muse base, Muse tuned, on the 300 sample, so every model in the comparison has the same judge.

```bash
python scripts/judge_ds.py --models reference qwen3.5-9b deepseek-v4.1-flash muse-glimmer-base muse-glimmer-tuned
python scripts/eval_report.py               # all leaderboard metrics, DeepSeek judge
python scripts/eval_report.py --judge gpt   # same, GPT-5.4 verdicts where they exist
```

### 5.3 Can the DeepSeek judge be trusted? It is the teacher.
Checked on 899 replies that both judges scored (Opus, DeepSeek, Qwen base):
- Overall score within 1 point: **95%** (identical 49%); "gives answer away" verdict identical: **97%**.
- Same ranking. Means GPT-5.4 vs DeepSeek: Opus 4.54 vs 4.32, DeepSeek 4.62 vs 4.36, Qwen base 3.16 vs 3.16. DeepSeek does **not** score its own replies above Opus's, so no sign of self-preference on these rows. It is slightly stricter at the top.
- Still, its scores are not on the same scale as the GPT-5.4 numbers on the live leaderboard. Compare models only within one judge.

### 5.4 Metrics
[scripts/eval_report.py](scripts/eval_report.py) reimplements `distill.site.collect.summarise` with the same definitions and bootstrap intervals, adding median output tokens and a paired McNemar for base vs tuned. All EVALUATION.md §3 columns are covered: valid output, gives answer away, resists fishing / injection, rules kept, same move, same grade, judge 1–5, win/tie vs Opus, quality, reveals answer when moving on, latency.

## 6. Reusing these results on another machine
`data/distill.db` is git-ignored. Everything this evaluation added is exported:

```bash
python scripts/export_results.py import results/muse_eval   # into your data/distill.db
python scripts/eval_report.py
```

[results/muse_eval/](results/muse_eval/): `runs.jsonl` (600 answers: Muse base and tuned on the 300 sample), `scores.jsonl` (rules scores for both, and every `*_ds` verdict for all five models), `board_deepseek_judge.json`, `REPORT_deepseek_judge.md`.

To put Muse on the public page, add `"muse-glimmer-tuned"` (and `"muse-glimmer-base"`) to `SHOW` in `distill/site.py`. Note the page reads the GPT-5.4 kinds; with DeepSeek as the final judge, `site.py` needs to read `*_ds` for every model (or rename the kinds) so the page doesn't mix judges.

## 7. Open items
- **Qwen3.5-9B tuned** (the other student, in progress on the other account): evaluate with the same `judge_ds.py` so both students share a judge.
- **Next training round for Muse**: the loss plateau after step 250 points to LR decay or a second epoch (free, ~1.5 h, warm-start from the saved adapter with the cookbook's `sft_loop` `warm_start_from_adapter`). Target the Assess grading regression with more stage-closing rows.
- **DPO on top of the tuned model is blocked on serverless.** The cookbook's serverless `dpo_loop` refuses warm starts (`serverless DPO does not support warm starts`), and loading the promoted adapter into a serverless session with `load_adapter` failed 4/4 times (`Adapter could not be loaded from durable storage`). Working route: SFT and DPO in **one** session (SFT free ~1.5 h, then DPO on the same weights). Pairs are ready: `data/export/dpo.jsonl` in the data repo (1,223 pairs, rejected = base Qwen3.5-9B); the 300 shortest are 1.95M tokens ≈ $15 of DPO (≈ $11.40 train + $3.81 reference prefill).
- **v2 SFT, prepared but not run** (no time before the presentation): [scripts/muse_sft_v2.py](scripts/muse_sft_v2.py) adds 20-step warmup + linear LR decay to 0; data = train + val with the 1,192 Assess grading turns repeated (10,235 rows, ~638 steps, ~1.7 h, free). See [fireworks-training-runs/codepraxis-muse-v2/run.md](fireworks-training-runs/codepraxis-muse-v2/run.md).
- **DPO is not free** for Muse (~$5.86/1M train + $1.96/1M reference prefill at serverless rates; ~$40 for 300 pairs of this length). Worth it only for a specific behaviour the judge flags.
- The 40-row held-out check inside training was a random split of `train.jsonl` (sessions can straddle it); the 300-turn test-set numbers above are the ones to quote.

## 8. Files added or changed
| File | What |
|---|---|
| `distill/models.py` | `muse-glimmer-tuned`, `muse-glimmer-base` model entries |
| `scripts/run_tuned.py` | answers from a Fireworks deployment without the backend |
| `scripts/judge_ds.py` | DeepSeek re-judge into `*_ds` kinds |
| `scripts/eval_report.py` | all leaderboard metrics per model and mode, either judge, judge agreement, paired tests |
| `scripts/export_results.py` | export / import the evaluation rows |
| `results/muse_eval/` | exported rows, the report, `board_deepseek_judge.json` (all metrics as JSON), `training_loss.csv` (521 steps) |
| `fireworks-training-runs/codepraxis-muse-v1/` | training decision log, step logs, predictions, live dashboard |
| `.gitignore` | ignores the training-data copies the starter writes into run folders |
