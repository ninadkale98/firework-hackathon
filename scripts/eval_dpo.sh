#!/bin/sh
# When the DPO job finishes: live-merge deploy it on the base's B200 shape, run the
# 300-turn shared sample, and judge it with DeepSeek like the SFT model.
cd "$(dirname "$0")/.."
export PYTHONPATH=../../backend:.
PY=../../backend/.venv/bin/python
L=logs/eval_dpo
mkdir -p $L
quiet() { grep --line-buffered -v -i warning; }
echo "$(date '+%H:%M:%S') waiting for the DPO job, then deploying" >> $L/progress.log
.venv-fw/bin/python - <<'PYEOF' 2>&1 | quiet >> $L/progress.log
import time
from dotenv import dotenv_values
from fireworks import Fireworks
A = "ninadnar98-z0upz59iv"
c = Fireworks(api_key=dotenv_values(".env")["FIREWORKS_API_KEY"], account_id=A)
while True:
    j = c.dpo_jobs.get("codepraxis-interviewer-dpo-v1")
    st = str(j.state)
    if st in ("JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"):
        print("DPO job ended:", st, j.status); raise SystemExit(1)
    if st == "JOB_STATE_COMPLETED":
        break
    time.sleep(60)
print(time.strftime("%H:%M:%S"), "DPO job completed; creating deployment")
c.deployments.create(
    base_model=f"accounts/{A}/models/codepraxis-interviewer-dpo-v1",
    deployment_id="codepraxis-interviewer-dpo-v1", display_name="Qwen3.5-9B tuned + DPO",
    deployment_shape="accounts/fireworks/deploymentShapes/rft-qwen3p5-9b-v2/versions/in9eo1d6",
    min_replica_count=0, max_replica_count=1,
    autoscaling_policy={"scale_to_zero_window": "1200s", "scale_down_window": "600s", "scale_up_window": "30s"})
PYEOF
grep -q "DPO job ended" $L/progress.log && exit 1
$PY - <<'PYEOF' >> $L/progress.log 2>&1
import asyncio, time
from distill.models import _completer
toy = {"anyOf": [{"type": "object", "additionalProperties": False, "required": ["move"], "properties": {"move": {"const": "FOLLOW_UP"}}}]}
start = time.time()
for i in range(80):
    try:
        r = asyncio.run(_completer("qwen3.5-9b-dpo").complete(system=[{"type": "text", "text": "Pick a move."}], message="hi", schema=toy))
        print(f"deployment awake after {time.time()-start:.0f}s: {r.move}"); break
    except Exception as e:
        print(f"  waiting ({time.time()-start:.0f}s): {str(e)[:90]}"); time.sleep(30)
PYEOF
echo "$(date '+%H:%M:%S') running the 300-turn sample" >> $L/progress.log
$PY -m distill.runs run --split test --models qwen3.5-9b-dpo --run-sample 300 --concurrency 12 --cap-fireworks 250 2>&1 | quiet | tail -3 >> $L/progress.log
$PY -m distill.runs score --split test 2>&1 | quiet | tail -2 >> $L/progress.log
echo "$(date '+%H:%M:%S') judging (DeepSeek)" >> $L/progress.log
export DISTILL_JUDGE_MODEL=deepseek-v4.1-flash
$PY -m distill.runs judge --split test --sample 300 --pairwise --models qwen3.5-9b qwen3.5-9b-tuned qwen3.5-9b-dpo \
    --concurrency 32 --cap-fireworks 250 2>&1 | quiet | tail -3 >> $L/progress.log
$PY -m distill.runs safety --split test --sample 300 --models qwen3.5-9b qwen3.5-9b-tuned qwen3.5-9b-dpo \
    --concurrency 32 --cap-fireworks 250 2>&1 | quiet | tail -3 >> $L/progress.log
echo "$(date '+%H:%M:%S') DONE" >> $L/progress.log
