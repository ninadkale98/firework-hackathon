#!/bin/sh
# Final evaluation: the tuned Qwen answers the 300-turn shared sample (full set later), then the
# DeepSeek judge scores Opus, base Qwen and tuned Qwen on the same 300-turn sample
# (criteria, head-to-head vs Opus, safety), so the comparison uses one judge.
cd "$(dirname "$0")/.."
export PYTHONPATH=../../backend:.
PY=../../backend/.venv/bin/python
TUNED="accounts/ninadnar98-z0upz59iv/models/codepraxis-interviewer-v1#accounts/ninadnar98-z0upz59iv/deployments/codepraxis-interviewer-v1"
L=logs/eval_tuned
mkdir -p $L
quiet() { grep --line-buffered -v -i warning; }
echo "$(date '+%H:%M:%S') waking tuned deployment" >> $L/progress.log
$PY - <<PYEOF >> $L/progress.log 2>&1
import asyncio, time
from distill.models import MODELS, ModelSpec, _completer
MODELS["warm"] = ModelSpec("warm", "fireworks", "$TUNED")
toy = {"anyOf": [{"type": "object", "additionalProperties": False, "required": ["move"], "properties": {"move": {"const": "FOLLOW_UP"}}}]}
start = time.time()
for i in range(80):
    try:
        r = asyncio.run(_completer("warm").complete(system=[{"type": "text", "text": "Pick a move."}], message="hi", schema=toy))
        print(f"deployment awake after {time.time()-start:.0f}s: {r.move}"); break
    except Exception as e:
        print(f"  waiting ({time.time()-start:.0f}s): {str(e)[:90]}"); time.sleep(30)
PYEOF
echo "$(date '+%H:%M:%S') running the 300-turn sample" >> $L/progress.log
$PY -m distill.runs run --split test --models qwen3.5-9b-tuned --run-sample 300 --concurrency 12 --cap-fireworks 250 2>&1 | quiet | tail -3 >> $L/progress.log
$PY -m distill.runs score --split test 2>&1 | quiet | tail -2 >> $L/progress.log
echo "$(date '+%H:%M:%S') judging (DeepSeek) reference, base, tuned" >> $L/progress.log
export DISTILL_JUDGE_MODEL=deepseek-v4.1-flash
$PY -m distill.runs judge --split test --sample 300 --pairwise --models qwen3.5-9b qwen3.5-9b-tuned \
    --concurrency 32 --cap-fireworks 250 2>&1 | quiet | tail -3 >> $L/progress.log
$PY -m distill.runs safety --split test --sample 300 --models qwen3.5-9b qwen3.5-9b-tuned \
    --concurrency 32 --cap-fireworks 250 2>&1 | quiet | tail -3 >> $L/progress.log
$PY -m distill.runs board --split test 2>&1 | quiet > $L/board.txt
echo "$(date '+%H:%M:%S') DONE" >> $L/progress.log
