#!/bin/sh
# Overnight training-data run (2026-11-01). DeepSeek V4.1 Flash writes every target;
# gpt-5.4-mini plays candidates and learners and filters. Training questions only.
cd "$(dirname "$0")/.."
export PYTHONPATH=../../backend:.
PY=../../backend/.venv/bin/python
L=logs/overnight
mkdir -p $L
stamp() { echo "$(date '+%H:%M:%S') $*" >> $L/progress.log; }
quiet() { grep -v -i warning; }

stamp "start"
# A: sessions, interview and Learn in parallel. Caps are the tracked provider totals.
( $PY -m distill.gen sessions --split train --interviewer deepseek-v4.1-flash --sessions 460 \
    --concurrency 8 --cap 95 --draw-rate 0.2 --seed 101 2>&1 | quiet > $L/a_interview.log
  stamp "interview sessions done: $(tail -1 $L/a_interview.log)"
  $PY -m distill.gen edits --split train --interviewer deepseek-v4.1-flash --n 400 --concurrency 8 --cap 95 --seed 102 2>&1 | quiet | tail -3 > $L/b_edits.log
  $PY -m distill.gen pairs --split train --interviewer deepseek-v4.1-flash --n 300 --concurrency 8 --cap 95 --seed 103 2>&1 | quiet | tail -3 > $L/b_pairs.log
  stamp "edits+pairs done: $(tail -1 $L/b_edits.log) / $(tail -1 $L/b_pairs.log)" ) &
( $PY -m distill.learn_gen --task study --split train --tutor deepseek-v4.1-flash --sessions 350 \
    --concurrency 5 --cap 95 --seed 201 2>&1 | quiet > $L/a_study.log
  stamp "study sessions done: $(tail -1 $L/a_study.log)"
  $PY -m distill.learn_gen --task assess --split train --tutor deepseek-v4.1-flash --sessions 170 \
    --concurrency 5 --cap 95 --seed 301 2>&1 | quiet > $L/a_assess.log
  stamp "assess sessions done: $(tail -1 $L/a_assess.log)" ) &
wait
stamp "generation done"

# C: coverage of the training split.
$PY -m distill.coverage --split train 2>&1 | quiet > $L/c_coverage.txt
stamp "coverage written"

# D: GPT-mini judge on every training target (mode-aware criteria), used as a filter.
$PY -m distill.runs judge --split train --sample 100000 \
    --concurrency 16 --cap-openai 300 2>&1 | quiet | tail -2 > $L/d_judge.log
stamp "judge filter done: $(tail -1 $L/d_judge.log)"

# E: export + longest example.
$PY -m distill.export --split train --val-share 0.08 2>&1 | quiet > $L/e_export.log
$PY - >> $L/e_export.log <<'PYEOF'
import json
for name in ("train", "val"):
    lens = sorted(sum(len(m["content"]) for m in json.loads(l)["messages"]) for l in open(f"data/export/{name}.jsonl"))
    print(f"{name}: {len(lens)} lines, chars p50={lens[len(lens)//2]} p99={lens[int(len(lens)*.99)]} max={lens[-1]} (~{lens[-1]//3.5:.0f} tokens at 3.5 chars/token)")
PYEOF
$PY -c "import sqlite3; c=sqlite3.connect('data/distill.db'); print(c.execute('select provider, round(sum(usd),2) from spend group by 1').fetchall())" >> $L/e_export.log
stamp "export done"
stamp "ALL DONE"
