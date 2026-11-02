#!/bin/sh
# Push tuned Qwen3.5-9B's numbers up, in the background:
#   A. the tuned model at temperature 0 (the Muse runs' setting) on the 300-turn sample, judged;
#   B. on-policy preference pairs (the tuned model's own bad answers vs the teacher's),
#      then ORPO from the SFT adapter with an explicit learning rate (the first DPO run had
#      learning rate 0 and a saturated loss), deployed and judged on the same sample.
cd "$(dirname "$0")/.."
export PYTHONPATH=../../backend:.
PY=../../backend/.venv/bin/python
FW=.venv-fw/bin/python
L=logs/improve
mkdir -p $L
quiet() { grep --line-buffered -v -i warning; }
log() { echo "$(date '+%H:%M:%S') $*" >> $L/progress.log; }

log "A: temperature 0 run on the 300-turn sample"
(
  $PY -m distill.runs run --split test --models qwen3.5-9b-tuned-t0 --run-sample 300 --concurrency 12 --cap-fireworks 400 2>&1 | quiet | tail -2 >> $L/a.log
  $PY -m distill.runs score --split test 2>&1 | quiet | tail -1 >> $L/a.log
  DISTILL_JUDGE_MODEL=deepseek-v4.1-flash $PY -m distill.runs judge --split test --sample 300 --pairwise \
      --models qwen3.5-9b-tuned-t0 --concurrency 24 --cap-fireworks 400 2>&1 | quiet | tail -2 >> $L/a.log
  DISTILL_JUDGE_MODEL=deepseek-v4.1-flash $PY -m distill.runs safety --split test --sample 300 \
      --models qwen3.5-9b-tuned-t0 --concurrency 24 --cap-fireworks 400 2>&1 | quiet | tail -2 >> $L/a.log
  log "A: DONE"
) &

log "B: tuned model answers 1,200 risky training prompts"
$PY -m distill.dpo run --student qwen3.5-9b-tuned-t0 --limit 1200 --concurrency 16 --cap-fireworks 400 2>&1 | quiet | tail -3 >> $L/b.log
$PY -m distill.dpo export --student qwen3.5-9b-tuned-t0 --out data/export/orpo_onpolicy.jsonl 2>&1 | quiet | tail -2 >> $L/b.log
log "B: pairs exported: $(wc -l < data/export/orpo_onpolicy.jsonl)"
$FW scripts/upload_dataset.py codepraxis-orpo-onpolicy-v1 data/export/orpo_onpolicy.jsonl 2>&1 | quiet | tail -2 >> $L/b.log
log "B: launching ORPO"
$FW - <<'PYEOF' 2>&1 | quiet >> $L/b.log
import json, time, httpx
from dotenv import dotenv_values
from fireworks import Fireworks
A = "ninadnar98-z0upz59iv"
c = Fireworks(api_key=dotenv_values(".env")["FIREWORKS_API_KEY"], account_id=A)
JOB = "codepraxis-interviewer-orpo-v1"
c.dpo_jobs.create(dataset=f"accounts/{A}/datasets/codepraxis-orpo-onpolicy-v1", dpo_job_id=JOB,
    display_name="CodePraxis interviewer ORPO v1 (on-policy pairs)",
    training_config={"warm_start_from": f"accounts/{A}/models/codepraxis-interviewer-v1", "epochs": 1,
                     "lora_rank": 16, "learning_rate": 2e-5, "max_context_length": 32768,
                     "output_model": f"accounts/{A}/models/codepraxis-interviewer-orpo-v1",
                     "sampler_checkpoint_save_interval_steps": 10},
    loss_config={"method": "ORPO"})
checked = False
while True:
    j = c.dpo_jobs.get(JOB)
    st = str(j.state)
    print(time.strftime("%H:%M:%S"), st, j.status.message if j.status else "", flush=True)
    if st in ("JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"):
        raise SystemExit(1)
    if st == "JOB_STATE_COMPLETED":
        break
    if not checked:
        try:
            rows = [json.loads(l) for l in httpx.get(c.dpo_jobs.get_metrics_file_endpoint(JOB).signed_url, timeout=30).text.splitlines() if l.strip()]
        except Exception:
            rows = []
        if len(rows) >= 5:
            checked = True
            lr = max(r.get("train/lr", 0) for r in rows)
            loss = [round(r.get("train/loss", 0), 4) for r in rows[:5]]
            print("first steps: max lr", lr, "loss", loss, flush=True)
            if lr == 0 or all(x == 0 for x in loss):
                print("NOT LEARNING: cancelling", flush=True)
                c.dpo_jobs.cancel(JOB, body={})
                raise SystemExit(1)
    time.sleep(60)
print("ORPO completed; deploying", flush=True)
c.deployments.create(
    base_model=f"accounts/{A}/models/codepraxis-interviewer-orpo-v1",
    deployment_id="codepraxis-interviewer-orpo-v1", display_name="Qwen3.5-9B tuned + ORPO",
    deployment_shape="accounts/fireworks/deploymentShapes/rft-qwen3p5-9b-v2/versions/in9eo1d6",
    min_replica_count=0, max_replica_count=1,
    autoscaling_policy={"scale_to_zero_window": "1200s", "scale_down_window": "600s", "scale_up_window": "30s"})
PYEOF
grep -q "ORPO completed" $L/b.log || { log "B: ORPO did not complete (see b.log)"; wait; log "ALL DONE"; exit 0; }
$PY - <<'PYEOF' >> $L/b.log 2>&1
import asyncio, time
from distill.models import _completer
toy = {"anyOf": [{"type": "object", "additionalProperties": False, "required": ["move"], "properties": {"move": {"const": "FOLLOW_UP"}}}]}
start = time.time()
for i in range(80):
    try:
        r = asyncio.run(_completer("qwen3.5-9b-orpo").complete(system=[{"type": "text", "text": "Pick a move."}], message="hi", schema=toy))
        print(f"deployment awake after {time.time()-start:.0f}s: {r.move}"); break
    except Exception as e:
        print(f"  waiting ({time.time()-start:.0f}s): {str(e)[:90]}"); time.sleep(30)
PYEOF
log "B: ORPO model on the 300-turn sample"
$PY -m distill.runs run --split test --models qwen3.5-9b-orpo --run-sample 300 --concurrency 12 --cap-fireworks 400 2>&1 | quiet | tail -2 >> $L/b.log
$PY -m distill.runs score --split test 2>&1 | quiet | tail -1 >> $L/b.log
DISTILL_JUDGE_MODEL=deepseek-v4.1-flash $PY -m distill.runs judge --split test --sample 300 --pairwise \
    --models qwen3.5-9b-orpo --concurrency 24 --cap-fireworks 400 2>&1 | quiet | tail -2 >> $L/b.log
DISTILL_JUDGE_MODEL=deepseek-v4.1-flash $PY -m distill.runs safety --split test --sample 300 \
    --models qwen3.5-9b-orpo --concurrency 24 --cap-fireworks 400 2>&1 | quiet | tail -2 >> $L/b.log
log "B: DONE"
wait
log "ALL DONE"
