#!/bin/sh
# Final overnight steps, run once generation is complete: GPT-5.4 judge on every
# training turn not yet judged, export, coverage, summary.
cd "$(dirname "$0")/.."
export PYTHONPATH=../../backend:.
PY=../../backend/.venv/bin/python
L=logs/overnight
stamp() { echo "$(date '+%H:%M:%S') final: $*" >> $L/progress.log; }
stamp "judge remaining training turns (GPT-5.4)"
$PY -m distill.runs judge --split train --sample 100000 --concurrency 16 --cap-openai 400 2>&1 \
  | grep --line-buffered -v -i warning | tail -3 > $L/f_judge_final.log
stamp "judge: $(tail -1 $L/f_judge_final.log)"
$PY -m distill.export --split train --val-share 0.08 2>&1 | grep -v -i warning > $L/f_export_final.log
$PY - >> $L/f_export_final.log <<'PYEOF'
import json, sqlite3
from collections import Counter
c = sqlite3.connect("data/distill.db")
judged = {r for (r,) in c.execute("select record_id from scores where model='reference' and kind='judge'")}
train = c.execute("select id, task from records where split='train'").fetchall()
print("judged by task:", dict(Counter(t for r, t in train if r in judged)), "unjudged:", sum(1 for r, t in train if r not in judged))
for name in ("train", "val"):
    lines = [json.loads(l) for l in open(f"data/export/{name}.jsonl")]
    lens = sorted(sum(len(m["content"]) for m in x["messages"]) for x in lines)
    print(f"{name}: {len(lines)} lines, chars p50={lens[len(lens)//2]} p99={lens[int(len(lens)*.99)]} max={lens[-1]} (~{lens[-1]/3.5:.0f} tokens)")
print("spend (tracked):", c.execute("select provider, round(sum(usd),2) from spend group by 1").fetchall())
PYEOF
$PY -m distill.coverage --split train 2>&1 | grep -v -i warning > $L/f_coverage_final.txt
stamp "FINAL DONE"
