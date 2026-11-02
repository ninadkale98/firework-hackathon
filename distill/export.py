"""Records -> Fireworks SFT JSONL.

    python -m distill.export --split train --val-share 0.08      # data/export/train.jsonl + val.jsonl
    python -m distill.export --split test                          # data/export/test.jsonl (reference = Opus)

One line per kept record, in the two messages production sends plus the target:

    system    = the system blocks joined as ``OpenAISchemaCompleter`` joins them
    user      = the exact user message
    assistant = the interviewer's move, compact JSON, wrapped as {"move_object": ...}
                (the shape the OpenAI-style client asks for when serving)

A record is kept only if its target passes the deterministic checks (valid, legal,
no leak) and, when it was judged, the judge passed it. Validation is carved out by
session, so no session is split across train and val.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from distill import store
from distill.score import score_output

WRAPPER_KEY = "move_object"
OUT_DIR = store.ROOT / "data" / "export"


def line(rec) -> dict:
    system = "\n\n".join(b["text"] for b in rec["system"])
    move = rec["interviewer"]["move"]
    target = json.dumps({WRAPPER_KEY: move}, ensure_ascii=False, separators=(",", ":"))
    return {"messages": [
        {"role": "system", "content": system},
        {"role": "user", "content": rec["message"]},
        {"role": "assistant", "content": target},
    ]}


def keep(conn, rec) -> tuple:
    out = rec.get("interviewer") or {}
    if not out.get("move"):
        return False, "no_move"
    s = score_output(rec, out)
    if not s.get("format_ok"):
        return False, "format"
    if s.get("leak"):
        return False, "leak"
    row = conn.execute("select doc from scores where record_id=? and model='reference' and kind='judge'",
                       (rec["id"],)).fetchone()
    if row:
        j = json.loads(row[0])
        if j.get("leak_ok") is False:
            return False, "judge_leak"
        if (j.get("overall") or 5) < 3:
            return False, "judge"
    return True, "ok"


def export(split: str, val_share: float, seed: int) -> None:
    conn = store.connect()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    recs = list(store.records(conn, split=split))
    why = Counter()
    kept = []
    for rec in recs:
        ok, reason = keep(conn, rec)
        why[reason] += 1
        if ok:
            kept.append(rec)
    files = {}
    if split == "train":
        sessions = sorted({r.get("session_id") for r in kept})
        rng = random.Random(seed)
        val = set(rng.sample(sessions, max(1, int(len(sessions) * val_share)))) if sessions else set()
        files["train"] = [r for r in kept if r.get("session_id") not in val]
        files["val"] = [r for r in kept if r.get("session_id") in val]
    else:
        files[split] = kept
    for name, rows in files.items():
        path = OUT_DIR / f"{name}.jsonl"
        with path.open("w") as fh:
            for r in rows:
                fh.write(json.dumps(line(r), ensure_ascii=False) + "\n")
        size = path.stat().st_size / 1e6
        print(f"{path}: {len(rows)} lines, {size:.1f} MB, tasks={dict(Counter(r['task'] for r in rows))}")
    print("filter:", dict(why))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--val-share", type=float, default=0.08)
    ap.add_argument("--seed", type=int, default=13)
    a = ap.parse_args()
    export(a.split, a.val_share, a.seed)
