"""Build the Model Arena scenarios for the CodePraxis hackathon demo page.

Each scenario is one real test session, played along Opus's path: the transcript is
Opus's own interview, and on the key turns every model answered the same prompt.
Writes one JSON file for the backend; prompts and schemas stay server-side (the
endpoint strips them before sending a scenario to the browser).

    PYTHONPATH=../../backend:. python scripts/build_arena_fixture.py <out.json>
"""

from __future__ import annotations

import json
import sys

from distill import store
from distill.models import cost

LANES = ["reference", "qwen3.5-9b", "qwen3.5-9b-tuned", "muse-glimmer-tuned"]
#: Where each lane's recorded answers come from: tuned Qwen is served at temperature 0,
#: as the Muse runs were and as the live endpoint calls it.
SOURCE = {"qwen3.5-9b-tuned": "qwen3.5-9b-tuned-t0"}
#: $ per turn for the GPU-served models: one GPU at the Fireworks on-demand rate
#: (B200 $13/h, H200 $8/h), at the ~200 turns a minute measured for the 9B.
GPU_TURN_USD = {"qwen3.5-9b": 13 / 12000, "qwen3.5-9b-tuned": 13 / 12000, "muse-glimmer-tuned": 8 / 4000}

SCENARIOS = [
    {"id": "change-review", "title": "Production change review", "task": "interview",
     "pitch": "The candidate pushes back, then asks where the facts are.",
     "key": ["b4fde8d34dc5-t9-c0", "b4fde8d34dc5-t10-c0"]},
    {"id": "dma-hinglish", "title": "Linux DMA driver, answered in Hinglish", "task": "interview",
     "pitch": "A Hinglish answer that reasons from structure instead of the evidence.",
     "key": ["bce8ca877226-t2-c0", "bce8ca877226-t3-c0"]},
    {"id": "cost-spike", "title": "AI agent cost spike", "task": "interview",
     "pitch": "A terse candidate names the culprit but skips the evidence.",
     "key": ["a79f796057a3-t0-c0"]},
    {"id": "workflow-outage", "title": "Approval workflow outage", "task": "interview",
     "pitch": "A strong candidate lays out a plan; the interviewer must not hand over the diagnosis.",
     "key": ["9d5b3a2724c7-t0-c0"]},
    {"id": "assess-closures", "title": "Assessment: Rust closures", "task": "assess",
     "pitch": "The learner gets a stage right; the examiner must grade it and move on.",
     "key": ["7cb0acb4c9b1-t2-c0"]},
]


def _said(move):
    """What the candidate reads, plus the move and (Assess) the grade."""
    if not move:
        return None
    out = {"move": move.get("move"), "response": move.get("response") or ""}
    for k in ("level", "probe_id", "hint_id"):
        if move.get(k) is not None:
            out[k] = move[k]
    return out


def main(out_path: str) -> None:
    conn = store.connect()
    recs = {r["id"]: r for r in store.records(conn, split="test")}

    def run(rid, model):
        if model == "reference":
            return recs[rid]["interviewer"]
        row = conn.execute("select doc from runs where record_id=? and model=?", (rid, model)).fetchone()
        return json.loads(row[0]) if row else None

    def score(rid, model, kind):
        row = conn.execute("select doc from scores where record_id=? and model=? and kind=?",
                           (rid, model, kind)).fetchone()
        return json.loads(row[0]) if row else {}

    def verdict(rid, model):
        j, s, r = score(rid, model, "judge_ds"), score(rid, model, "safety_ds"), score(rid, model, "rules")
        failed = {k: v.get("evidence", "") for k, v in j.items() if isinstance(v, dict) and v.get("pass") is False}
        return {
            "valid": bool(r.get("format_ok", True)),
            "same_move": r.get("same_move"),
            "judge": j.get("overall"),
            "gave_answer_away": j.get("leak_ok") is False,
            "safety_ok": s.get("ok") if s else None,
            "failed": failed,
        }

    scenarios = []
    for sc in SCENARIOS:
        first = recs[sc["key"][0]]
        sess = sorted((r for r in recs.values() if r["session_id"] == first["session_id"]
                       and r["id"].endswith("-c0")), key=lambda r: r["turn"])
        last_key = max(recs[k]["turn"] for k in sc["key"])
        turns = []
        for r in sess:
            if r["turn"] > last_key:
                break
            f = r.get("facts") or {}
            t = {
                "id": r["id"], "turn": r["turn"], "candidate": r["message"],
                "stage": f.get("stage"), "stage_position": f.get("stage_position"),
                "question_position": f.get("question_position"),
                "time_left_in_stage_s": f.get("time_left_in_stage_s"),
                "offered": r.get("offered"), "key": r["id"] in sc["key"],
                "opus": _said(r["interviewer"].get("move")),
            }
            if t["key"]:
                lanes = {}
                for m in LANES:
                    src = SOURCE.get(m, m)
                    o = run(r["id"], src) or {}
                    toks = o.get("tokens") or {}
                    usd = (cost("opus-5", toks.get("in") or 0, toks.get("out") or 0)
                           if m == "reference" else GPU_TURN_USD[m])
                    lanes[m] = {"said": _said(o.get("move")), "raw_if_invalid": None if o.get("move") else (o.get("raw") or "")[:400],
                                "latency_ms": o.get("latency_ms") or o.get("wall_ms"), "usd": usd,
                                "verdict": verdict(r["id"], src)}
                t["lanes"] = lanes
                # Server-only: the exact production prompt, for the live tab.
                t["_system"] = r["system"]
                t["_schema"] = r["schema"]
            turns.append(t)
        facts = first.get("facts") or {}
        scenarios.append({**{k: sc[k] for k in ("id", "title", "task", "pitch")},
                          "question_name": facts.get("question_name"), "seniority": facts.get("seniority"),
                          "turns": turns})
    with open(out_path, "w") as fh:
        json.dump({"lanes": LANES, "scenarios": scenarios}, fh, ensure_ascii=False)
    print(out_path, [(s["id"], len(s["turns"])) for s in scenarios])


if __name__ == "__main__":
    main(sys.argv[1])
