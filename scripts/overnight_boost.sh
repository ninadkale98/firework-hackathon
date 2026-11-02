#!/bin/sh
# More parallel generation for the overnight run, a watcher that stops each task at
# its target, then a final judge + export pass once everything has finished.
cd "$(dirname "$0")/.."
export PYTHONPATH=../../backend:.
PY=../../backend/.venv/bin/python
L=logs/overnight
stamp() { echo "$(date '+%H:%M:%S') boost: $*" >> $L/progress.log; }
quiet() { grep --line-buffered -v -i warning; }
count() { $PY -c "import sqlite3; c=sqlite3.connect('data/distill.db'); print(c.execute(\"select count(*) from records where split='train' and task='$1' and source='sim'\").fetchone()[0])"; }

stamp "start extra workers"
$PY -m distill.gen sessions --split train --interviewer deepseek-v4.1-flash --sessions 400 \
    --concurrency 16 --cap 95 --draw-rate 0.2 --seed 111 2>&1 | quiet > $L/boost_interview.log &
$PY -m distill.learn_gen --task study --split train --tutor deepseek-v4.1-flash --sessions 300 \
    --concurrency 10 --cap 95 --seed 211 2>&1 | quiet > $L/boost_study.log &
$PY -m distill.learn_gen --task assess --split train --tutor deepseek-v4.1-flash --sessions 200 \
    --concurrency 10 --cap 95 --seed 311 2>&1 | quiet > $L/boost_assess.log &

# Watcher: stop a task's generators once it has enough simulated turns.
I=0; S=0; A=0
while pgrep -f "distill.gen sessions|distill.learn_gen" > /dev/null; do
  sleep 60
  [ $I = 0 ] && [ "$(count interview)" -ge 4400 ] && pkill -f "distill.gen sessions" && I=1 && stamp "interview target reached"
  [ $S = 0 ] && [ "$(count study)" -ge 2600 ] && pkill -f "distill.learn_gen --task study" && S=1 && stamp "study target reached"
  [ $A = 0 ] && [ "$(count assess)" -ge 1750 ] && pkill -f "distill.learn_gen --task assess" && A=1 && stamp "assess target reached"
done
stamp "all generation stopped: interview=$(count interview) study=$(count study) assess=$(count assess)"

# Wait for the main script (edits, pairs, its judge and export) to finish.
while ! grep -q "ALL DONE" $L/progress.log; do sleep 60; done

# Final pass: judge anything generated after the main filter, then export again.
$PY -m distill.runs judge --split train --sample 100000 --concurrency 16 --cap-openai 340 2>&1 | quiet | tail -2 > $L/f_judge_final.log
stamp "final judge: $(tail -1 $L/f_judge_final.log)"
$PY -m distill.export --split train --val-share 0.08 2>&1 | quiet > $L/f_export_final.log
$PY - >> $L/f_export_final.log <<'PYEOF'
import json, sqlite3
c = sqlite3.connect("data/distill.db")
judged = {r for (r,) in c.execute("select record_id from scores where model='reference' and kind='judge'")}
train = [r for (r,) in c.execute("select id from records where split='train'")]
print(f"train records {len(train)}, judged {sum(1 for r in train if r in judged)}, unjudged {sum(1 for r in train if r not in judged)}")
for name in ("train", "val"):
    lens = sorted(sum(len(m["content"]) for m in json.loads(l)["messages"]) for l in open(f"data/export/{name}.jsonl"))
    print(f"{name}: {len(lens)} lines, chars p50={lens[len(lens)//2]} p99={lens[int(len(lens)*.99)]} max={lens[-1]} (~{lens[-1]/3.5:.0f} tokens)")
print(c.execute("select provider, round(sum(usd),2) from spend group by 1").fetchall())
PYEOF
$PY -m distill.coverage --split train 2>&1 | quiet > $L/f_coverage_final.txt
stamp "FINAL DONE"
