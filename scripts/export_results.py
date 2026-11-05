"""Export or import the Muse Glimmer evaluation rows, so another machine's
``data/distill.db`` can hold the same results without rerunning anything.

Exports the ``runs`` of the new models and every ``scores`` row the evaluation
added: rules scores for the new models, and all DeepSeek re-judge verdicts
(``judge_ds`` / ``pairwise_ds`` / ``safety_ds``) for every model.

    python scripts/export_results.py export results/muse_eval
    python scripts/export_results.py import results/muse_eval   # insert or replace
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from distill import store  # noqa: E402

NEW_MODELS = ("muse-glimmer-base", "muse-glimmer-tuned", "muse-glimmer-tuned-prompt-opt",
              "muse-glimmer-tuned-prompt-cot", "muse-glimmer-v2-step100")


def export(out: Path) -> None:
    conn = store.connect()
    out.mkdir(parents=True, exist_ok=True)
    q = ",".join("?" * len(NEW_MODELS))
    n = 0
    with (out / "runs.jsonl").open("w") as f:
        for rid, model, doc, ts in conn.execute(f"select * from runs where model in ({q})", NEW_MODELS):
            f.write(json.dumps({"record_id": rid, "model": model, "doc": json.loads(doc), "ts": ts}) + "\n"); n += 1
    m = 0
    with (out / "scores.jsonl").open("w") as f:
        for rid, model, kind, doc, ts in conn.execute(
                f"select * from scores where kind like '%\\_ds' escape '\\' or (kind='rules' and model in ({q}))",
                NEW_MODELS):
            f.write(json.dumps({"record_id": rid, "model": model, "kind": kind, "doc": json.loads(doc), "ts": ts}) + "\n"); m += 1
    print(f"exported {n} runs, {m} scores to {out}")


def load(src: Path) -> None:
    conn = store.connect()
    n = m = 0
    for line in (src / "runs.jsonl").open():
        r = json.loads(line)
        conn.execute("insert or replace into runs values (?,?,?,?)",
                     (r["record_id"], r["model"], json.dumps(r["doc"]), r["ts"])); n += 1
    for line in (src / "scores.jsonl").open():
        s = json.loads(line)
        conn.execute("insert or replace into scores values (?,?,?,?,?)",
                     (s["record_id"], s["model"], s["kind"], json.dumps(s["doc"]), s["ts"])); m += 1
    print(f"imported {n} runs, {m} scores from {src}")


if __name__ == "__main__":
    cmd, path = sys.argv[1], Path(sys.argv[2])
    export(path) if cmd == "export" else load(path)
