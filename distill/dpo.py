"""Preference pairs for DPO / ORPO: the teacher's good answer against the student's
own bad answer to the same training prompt.

    python -m distill.dpo select --n 2500                    # pick risky training prompts (no spend)
    python -m distill.dpo run --student <model> --limit 50   # student answers + judge (spends)
    python -m distill.dpo export                             # data/export/dpo.jsonl

``<model>`` is a name in ``distill.models.MODELS`` or a Fireworks model or
deployment id (``accounts/.../models/...`` or ``…#accounts/.../deployments/...``),
so the rejected answers can come from the fine-tuned Qwen or from Muse.

Only training-split prompts are used. A pair is kept when the teacher's answer
passed the GPT-5.4 filter and the student's answer is judged bad: invalid output,
gives the answer away, fails the safety check on a fishing or injection turn, or
scores 2+ points below the teacher's. The judge is ``DISTILL_JUDGE_MODEL``
(default here: DeepSeek V4.1 Flash on Fireworks; it flags real failures but
misses some leaks, see scripts/calibrate_judge.py).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import traceback
from pathlib import Path
from collections import Counter
from typing import Any, Dict, List

os.environ.setdefault("DISTILL_JUDGE_MODEL", "deepseek-v4.1-flash")

from distill import store  # noqa: E402
from distill.models import MODELS, ModelSpec, completer, cost  # noqa: E402
from distill.runs import attempt_turn  # noqa: E402
from distill.score import score_output  # noqa: E402

WRAPPER_KEY = "move_object"
SELECTION = store.ROOT / "data" / "dpo_prompts.json"
OUT = store.ROOT / "data" / "export" / "dpo.jsonl"


def _teacher_ok(conn, rid: str):
    row = conn.execute("select doc from scores where record_id=? and model='reference' and kind='judge'",
                       (rid,)).fetchone()
    if not row:
        return None
    j = json.loads(row[0])
    if j.get("leak_ok") is False or (j.get("overall") or 0) < 4:
        return None
    return j


def _val_sessions(conn, val_share: float = 0.08, seed: int = 13) -> set:
    """The SFT export's validation sessions, recomputed with its own rule and seed."""
    from distill.export import keep

    kept = [r for r in store.records(conn, split="train") if keep(conn, r)[0]]
    sessions = sorted({r.get("session_id") for r in kept})
    rng = random.Random(seed)
    return set(rng.sample(sessions, max(1, int(len(sessions) * val_share)))) if sessions else set()


def _risk(rec: Dict[str, Any]) -> str:
    """Which weak spot of the base model this prompt exercises (or ``other``)."""
    b, cand = rec.get("buckets") or {}, rec.get("candidate") or {}
    if attempt_turn(rec):
        return "fishing_or_injection"
    if cand.get("persona") == "wrong_belief" or cand.get("kind") == "confused":
        return "wrong_belief"
    if rec["task"] in ("interview", "assess") and any(m in (rec.get("offered") or []) for m in ("PROBE", "QUESTION_DONE")):
        return "can_close_stage"
    if str(b.get("style", "clean")) not in ("clean", "-"):
        return "messy_english"
    if b.get("stage_clock") in ("0", "1-59"):
        return "low_time"
    return "other"


def select(n: int, seed: int) -> None:
    """Risky training prompts whose teacher answer is good, spread over tasks and
    weak spots. Validation sessions are left out, as in the SFT export."""
    conn = store.connect()
    val = _val_sessions(conn)
    by: Dict[tuple, List[str]] = {}
    for rec in store.records(conn, split="train"):
        if rec.get("session_id") in val or not (rec.get("interviewer") or {}).get("move"):
            continue
        if _teacher_ok(conn, rec["id"]) is None:
            continue
        by.setdefault((rec["task"], _risk(rec)), []).append(rec["id"])
    rng = random.Random(seed)
    for ids in by.values():
        rng.shuffle(ids)
    picked: List[str] = []
    keys = sorted(by, key=lambda k: (k[1] == "other", k))     # weak spots first
    while len(picked) < n and any(by[k] for k in keys):
        for k in keys:
            if by[k] and len(picked) < n:
                picked.append(by[k].pop())
    SELECTION.write_text(json.dumps(picked))
    mix = Counter()
    for rec in store.records(conn, split="train"):
        if rec["id"] in set(picked):
            mix[(rec["task"], _risk(rec))] += 1
    print(f"selected {len(picked)} prompts -> {SELECTION}")
    for k, v in sorted(mix.items()):
        print(f"  {k[0]:<9} {k[1]:<22} {v}")


def _student_name(student: str) -> str:
    """A registry name for ``student``, registering a raw Fireworks id on the fly."""
    if student in MODELS:
        return student
    name = "dpo-student"
    MODELS[name] = ModelSpec(name, "fireworks", student, None, None, "DPO rejected-answer source")
    return name


async def run(student: str, limit: int, concurrency: int, cap: float) -> None:
    from distill import judge as J

    conn = store.connect()
    name = _student_name(student)
    run_key = f"dpo:{MODELS[name].model}"
    ids = json.loads(SELECTION.read_text())[: limit or None]
    recs = {r["id"]: r for r in store.records(conn, split="train") if r["id"] in set(ids)}
    comp = completer(name)
    sem = asyncio.Semaphore(concurrency)
    done = Counter()
    judge_provider = MODELS[J.DEFAULT_MODEL].provider if J.DEFAULT_MODEL in MODELS else "openai"

    async def one(rid: str):
        async with sem:
            if store.has_run(conn, rid, run_key) or store.spent(conn, "fireworks") >= cap:
                done["skipped"] += 1
                return
            rec = recs[rid]
            try:
                reply = await comp.complete(system=rec["system"], message=rec["message"], schema=rec["schema"])
                out = {"model": reply.model, "stop_reason": reply.stop_reason, "raw": reply.raw_text,
                       "move": reply.move, "latency_ms": reply.latency_ms,
                       "tokens": {"in": reply.input_tokens, "out": reply.output_tokens}}
                verdict = await J.criteria_task(rec, out)
                if attempt_turn(rec) and out.get("move"):
                    verdict["safety"] = await J.safety_task(rec, out)
                for u in (verdict.get("usage") or {}, (verdict.get("safety") or {}).get("usage") or {}):
                    store.log_spend(conn, judge_provider, J.DEFAULT_MODEL, "dpo:judge", u.get("in", 0),
                                    u.get("out", 0), cost(J.DEFAULT_MODEL, u.get("in", 0), u.get("out", 0)))
                store.put_run(conn, rid, run_key, out)
                store.put_score(conn, rid, run_key, "judge_dpo", verdict)
                done["ok"] += 1
            except Exception:
                done["error"] += 1
                traceback.print_exc(limit=2)

    await asyncio.gather(*(one(r) for r in ids if r in recs))
    print(dict(done), "spend", {p: round(store.spent(conn, p), 2) for p in ("fireworks", "openai")})


def _rejected(rec, out, verdict, teacher) -> str:
    """Why the student's answer is a good "rejected" example, or '' if it is not."""
    s = score_output(rec, out)
    if not s.get("format_ok"):
        return "invalid_output"
    if verdict.get("leak_ok") is False:
        return "gives_answer_away"
    if (verdict.get("safety") or {}).get("ok") is False:
        return "fails_safety"
    if verdict.get("overall") is not None and teacher.get("overall") is not None \
            and verdict["overall"] <= teacher["overall"] - 2:
        return "much_worse"
    return ""


def export(student: str, out=None) -> None:
    out = Path(out) if out else OUT
    conn = store.connect()
    name = _student_name(student)
    run_key = f"dpo:{MODELS[name].model}"
    ids = json.loads(SELECTION.read_text())
    recs = {r["id"]: r for r in store.records(conn, split="train") if r["id"] in set(ids)}
    why, kept = Counter(), 0
    with out.open("w") as fh:
        for rid in ids:
            rec = recs.get(rid)
            row = conn.execute("select doc from runs where record_id=? and model=?", (rid, run_key)).fetchone()
            vrow = conn.execute("select doc from scores where record_id=? and model=? and kind='judge_dpo'",
                                (rid, run_key)).fetchone()
            teacher = _teacher_ok(conn, rid)
            if not (rec and row and vrow and teacher):
                why["missing"] += 1
                continue
            out, verdict = json.loads(row[0]), json.loads(vrow[0])
            reason = _rejected(rec, out, verdict, teacher)
            if not reason:
                why["student_ok"] += 1
                continue
            chosen = json.dumps({WRAPPER_KEY: rec["interviewer"]["move"]}, ensure_ascii=False, separators=(",", ":"))
            rejected = (json.dumps({WRAPPER_KEY: out["move"]}, ensure_ascii=False, separators=(",", ":"))
                        if out.get("move") else (out.get("raw") or ""))
            if not rejected.strip() or rejected == chosen:
                why["empty_or_same"] += 1
                continue
            system = "\n\n".join(b["text"] for b in rec["system"])
            fh.write(json.dumps({
                "input": {"messages": [{"role": "system", "content": system},
                                       {"role": "user", "content": rec["message"]}]},
                "preferred_output": [{"role": "assistant", "content": chosen}],
                "non_preferred_output": [{"role": "assistant", "content": rejected}],
            }, ensure_ascii=False) + "\n")
            why[reason] += 1
            kept += 1
    print(f"{out}: {kept} pairs  reasons={dict(why)}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["select", "run", "export"])
    ap.add_argument("--n", type=int, default=2500)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--student", default="qwen3.5-9b")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=12)
    ap.add_argument("--cap-fireworks", type=float, default=150)
    ap.add_argument("--out", default=None, help="export: output path (default data/export/dpo.jsonl)")
    a = ap.parse_args(argv)
    if a.what == "select":
        select(a.n, a.seed)
    elif a.what == "run":
        asyncio.run(run(a.student, a.limit, a.concurrency, a.cap_fireworks))
    else:
        export(a.student, a.out)


if __name__ == "__main__":
    main()
