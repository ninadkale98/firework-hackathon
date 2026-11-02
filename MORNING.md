# Morning summary: 2026-11-02

## Overnight result

| | |
|---|---|
| Training turns generated | **9,577**: Interview 5,186 (4,486 from sessions + 400 clock edits + 300 messy pairs), Study 2,641, Assess 1,750 |
| Teacher | DeepSeek V4.1 Flash (all modes). Candidates and learners: gpt-5.4-mini |
| Filter | **GPT-5.4 judged all 9,577** (none left unjudged) + JSON check + code leak pre-check |
| Kept | **9,043 (94%)** → `data/export/train.jsonl` **8,361 lines (261 MB)** + `data/export/val.jsonl` **682 lines (22 MB)**, split by session |
| Dropped | 240 code leak pre-check · 104 invalid JSON · 97 judge "gave the answer away" · 85 judge score < 3 · 8 no move |
| Train mix | Interview 4,471 · Study 2,395 · Assess 1,495 |
| Length | median ~6.7k tokens, p99 ~23k, max ~27k (estimated at 3.5 chars/token) → **`--max-context-length 32768` fits** |
| Coverage | pairs of main-field values covered: Interview 707/781, Assess 210/262, Study 94/98. Full report: `logs/overnight/f_coverage_final.txt` |
| Spend (tracked; overstates real) | Fireworks $84 · OpenAI $389 · Anthropic $70 |

## Upload and train

Fireworks blocks AI agents from creating resources: run these yourself, or through the `fireworks-training` plugin.

```bash
cd ~/codepraxis/fullstack_v2/hackathon/fireworks-distill
bin/fw dataset create codepraxis-distill-train data/export/train.jsonl
bin/fw dataset create codepraxis-distill-val   data/export/val.jsonl

bin/fw sftj create \
  --base-model accounts/fireworks/models/qwen3p5-9b \
  --dataset codepraxis-distill-train \
  --evaluation-dataset codepraxis-distill-val \
  --output-model codepraxis-interviewer-v1 \
  --lora-rank 16 --epochs 2 --max-context-length 32768
```

Cost estimate: ~56M tokens × 2 epochs ≈ 112M training tokens × $0.50/M ≈ **$56** (LoRA, models up to 16B). Fireworks has ~$15–20 left after tonight, so this needs **hackathon credits or a top-up**.

## After training

1. Deploy the tuned model:
   ```bash
   bin/fw deployment create <tuned-model-id> --deployment-shape default \
     --min-replica-count 0 --max-replica-count 1 --scale-to-zero-window 20m --wait
   ```
2. Smoke-test 5 validation turns. Check the replies are JSON in the `{"move_object": …}` shape with no thinking text: we train and serve with Qwen3.5 thinking off.
3. Run the tuned model on the frozen test set (1,525 turns), then the final judge on the 300-turn sample. This is Claude's job: add a `qwen3.5-9b-tuned` entry in `distill/models.py` pointing at the deployment, then `runs run / score / judge / safety` and `site`. The page already has a slot for "Qwen3.5-9B tuned (ours)".
4. Measure throughput under load to give the tuned model a real $/turn on the cost chart.

## Final baseline to beat (300-turn sample, final judge)

| | Opus 5 | DeepSeek V4.1 Flash (teacher) | Qwen3.5-9B base |
|---|---|---|---|
| Quality | 95% | 96% | 81% |
| Judge 1–5 | 4.54 | 4.62 | 3.16 |
| Judge prefers it or ties vs Opus | – | 57% | 12% |
| Reveals answer when moving on | 2% | 5% | 39% |
| Same grade as Opus (Assess) | – | 97% | 68% |
| Same move on messy English | 93% | 83% | 43% |
