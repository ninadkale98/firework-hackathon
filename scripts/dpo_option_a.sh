#!/bin/sh
# DPO option A: base Qwen3.5-9B answers the 2,500 selected training prompts, DeepSeek
# judges, failures become rejected answers. Wakes the scaled-to-zero deployment first.
cd "$(dirname "$0")/.."
export PYTHONPATH=../../backend:.
PY=../../backend/.venv/bin/python
STUDENT="accounts/fireworks/models/qwen3p5-9b#accounts/ninadnar98-z0upz59iv/deployments/qwen35-9b-base"
L=logs/dpo
echo "$(date '+%H:%M:%S') waking deployment" >> $L/progress.log
$PY - <<PYEOF >> $L/progress.log 2>&1
import asyncio, time
from distill.models import MODELS, ModelSpec, _completer
MODELS["warm"] = ModelSpec("warm", "fireworks", "$STUDENT")
toy = {"anyOf": [{"type": "object", "additionalProperties": False, "required": ["move"], "properties": {"move": {"const": "FOLLOW_UP"}}}]}
start = time.time()
for i in range(60):
    try:
        r = asyncio.run(_completer("warm").complete(system=[{"type": "text", "text": "Pick a move."}], message="hi", schema=toy))
        print(f"deployment awake after {time.time()-start:.0f}s: {r.move}"); break
    except Exception as e:
        print(f"  waiting ({time.time()-start:.0f}s): {str(e)[:90]}"); time.sleep(30)
PYEOF
echo "$(date '+%H:%M:%S') running 2,500 prompts" >> $L/progress.log
$PY -m distill.dpo run --student "$STUDENT" --concurrency 12 --cap-fireworks 150 2>&1 | grep --line-buffered -v -i warning | tail -5 >> $L/progress.log
$PY -m distill.dpo export --student "$STUDENT" 2>&1 | grep -v -i warning | tail -2 >> $L/progress.log
echo "$(date '+%H:%M:%S') DONE" >> $L/progress.log
