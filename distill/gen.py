"""Make interview records: simulated sessions, then clock edits and clean/messy pairs.

    python -m distill.gen sessions --split test --interviewer opus-5 --sessions 40
    python -m distill.gen edits    --split test --interviewer opus-5 --n 80
    python -m distill.gen pairs    --split test --interviewer opus-5 --n 60

Every call goes through the production engine (``engine.run_session`` /
``engine.replay``). Spend is logged per call and each command stops at its cap.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import random
import sys
import traceback
from collections import Counter
from typing import Any, Dict, List, Optional

from distill import store
from distill.bank import SPLITS, load, local_ids, resolve
from distill.corrupt import corrupt
from distill.engine import replay, run_session
from distill.models import MODELS, completer, cost
from distill.simulator import PERSONAS, Profile, Simulator

INTERVIEW_TRACKS = ["embedded", "frontend", "backend", "ai_eng"]


def _available(ref) -> bool:
    return isinstance(ref, int) or ref in local_ids()


def questions(track: str, split: str) -> List[int]:
    return [resolve(q) for q in SPLITS[track][split] if _available(q)]


def track_of(qid: int) -> Optional[str]:
    for track, parts in SPLITS.items():
        for ids in parts.values():
            if any(_available(q) and resolve(q) == qid for q in ids):
                return track
    return None


def _log(conn, model_name: str, role: str, tin: int, tout: int):
    spec = MODELS[model_name]
    store.log_spend(conn, spec.provider, model_name, role, tin, tout, cost(model_name, tin, tout))


def _annotate(rec: Dict[str, Any]) -> Dict[str, Any]:
    cand = rec.get("candidate") or {}
    rec.setdefault("buckets", {})
    rec["buckets"]["persona"] = cand.get("persona", "-")
    rec["buckets"]["style"] = cand.get("style", "-")
    rec["buckets"]["channels"] = ",".join(cand.get("channels", ["text"]))
    msg = rec.get("message") or ""
    words = len(msg.split())
    rec["buckets"]["length"] = ("1-5" if words <= 5 else "sentence" if words <= 30
                                else "paragraph" if words <= 150 else "300+" if words > 300 else "long")
    return rec


# ── sessions ────────────────────────────────────────────────────────────


async def sessions(split: str, interviewer_name: str, n: int, *, concurrency: int, cap: float,
                   sim_model: str, seed: int, only_personas=None, only_styles=None, only_qids=None,
                   draw_rate: float = 0.2) -> None:
    conn = store.connect()
    bank = load()
    rng = random.Random(seed)
    personas = list(PERSONAS)
    plans = []
    for i in range(n):
        track = INTERVIEW_TRACKS[i % len(INTERVIEW_TRACKS)]
        pool = questions(track, split)
        if only_qids:
            pool = [q for q in pool if q in only_qids] or ([] if any(track_of(q) == track for q in only_qids) else [])
            if not pool:
                track = INTERVIEW_TRACKS[(i + 1) % len(INTERVIEW_TRACKS)]
                pool = [q for q in questions(track, split) if q in only_qids]
        if not pool:
            continue
        k = 2 if (len(pool) > 1 and rng.random() < 0.3) else 1
        qids = rng.sample(pool, k)
        minutes = sum(bank.get(q).duration or 10 for q in qids)
        minutes = max(5, int(round(minutes * rng.choice([0.5, 0.8, 1.0, 1.0, 1.4]))))
        cycle = only_personas or personas
        persona = cycle[(i // len(INTERVIEW_TRACKS)) % len(cycle)] if i < len(cycle) * 4 \
            else rng.choice(cycle)
        style = rng.choice(only_styles) if only_styles else None
        plans.append({"qids": qids, "track": track, "minutes": minutes,
                      "profile": Profile.draw(rng, persona=persona, style=style),
                      "draw": rng.random() < draw_rate})

    sem = asyncio.Semaphore(concurrency)
    interviewer = completer(interviewer_name)
    provider = MODELS[interviewer_name].provider
    done = Counter()

    async def one(plan):
        async with sem:
            if store.spent(conn, provider) >= cap:
                done["skipped_cap"] += 1
                return
            sim = Simulator(plan["profile"], model=sim_model, may_draw=plan["draw"])
            tags = {"track": plan["track"], "minutes": plan["minutes"],
                    "persona": plan["profile"].persona, "style": plan["profile"].style}
            try:
                out = await run_session(plan["qids"], bank=bank, interviewer=interviewer,
                                        candidate=sim, total_minutes=plan["minutes"],
                                        max_turns=18, session_tags=tags)
            except Exception:
                done["failed"] += 1
                traceback.print_exc(limit=3)
                return
            finally:
                _log(conn, sim_model, "simulator", sim.usage["in"], sim.usage["out"])
            for rec in out["records"]:
                t = rec["interviewer"].get("tokens") or {}
                _log(conn, interviewer_name, f"interviewer:{split}", t.get("in", 0), t.get("out", 0))
                store.put_record(conn, _annotate(rec), task="interview", split=split, source="sim",
                                 track=track_of(rec["question_id"]) if rec.get("question_id") else plan["track"])
            store.put_session(conn, out, split=split, task="interview")
            done["sessions"] += 1
            done["records"] += len(out["records"])
            print(f"[{done['sessions']}/{len(plans)}] {plan['track']} q{plan['qids']} "
                  f"{plan['profile'].persona}/{plan['profile'].style} turns={out['turns']} "
                  f"spent={store.spent(conn):.2f}", flush=True)

    await asyncio.gather(*(one(p) for p in plans))
    print(dict(done), "total spent", round(store.spent(conn), 2))


# ── edits: the same state at another clock ──────────────────────────────

#: (stage seconds, question seconds) edge cases the sessions under-produce.
CLOCK_EDGES = [(0, 400), (25, 400), (50, 90), (200, 60), (300, 900), (700, 1200), (500, 30), (0, 0), (40, 15), (420, 600)]


async def edits(split: str, interviewer_name: str, n: int, *, concurrency: int, cap: float, seed: int):
    conn = store.connect()
    rng = random.Random(seed)
    base = [r for r in store.records(conn, split=split, task="interview") if r["source"] == "sim"]
    rng.shuffle(base)
    interviewer = completer(interviewer_name)
    provider = MODELS[interviewer_name].provider
    sem = asyncio.Semaphore(concurrency)
    made = Counter()

    async def one(rec, edge):
        async with sem:
            if store.spent(conn, provider) >= cap:
                return
            ai = store.agent_input(conn, rec["id"])
            if ai is None:
                return
            # Any edge is a reachable state: the question clock is set with the stage
            # clock, and a stage never has more time than its question.
            q = edge[1]
            clock = dataclasses.replace(ai.current_state, time_left_in_stage_s=min(edge[0], q) if q else 0,
                                        time_left_in_question_s=q)
            new = dataclasses.replace(ai, current_state=clock)
            try:
                out = await replay(new, interviewer)
            except Exception:
                traceback.print_exc(limit=2)
                return
            t = out["interviewer"].get("tokens") or {}
            _log(conn, interviewer_name, f"interviewer:{split}:edit", t.get("in", 0), t.get("out", 0))
            out.update({"id": f"{rec['id']}-clk{edge[0]}_{edge[1]}", "session_id": rec["session_id"],
                        "turn": rec["turn"], "question_id": rec["question_id"], "tags": rec.get("tags"),
                        "candidate": rec.get("candidate"), "edit_of": rec["id"], "edit": {"clock": edge}})
            store.put_record(conn, _annotate(out), task="interview", split=split, source="edit",
                             track=rec.get("track"))
            made["edits"] += 1

    jobs = [one(rec, CLOCK_EDGES[i % len(CLOCK_EDGES)]) for i, rec in enumerate(base[:n])]
    await asyncio.gather(*jobs)
    print(dict(made), "total spent", round(store.spent(conn), 2))


# ── pairs: the same answer, written messily ─────────────────────────────


async def pairs(split: str, interviewer_name: str, n: int, *, concurrency: int, cap: float, seed: int):
    conn = store.connect()
    rng = random.Random(seed)
    base = [r for r in store.records(conn, split=split, task="interview")
            if r["source"] == "sim" and r["buckets"].get("style") == "clean"
            and len((r.get("message") or "").split()) >= 8]
    rng.shuffle(base)
    interviewer = completer(interviewer_name)
    provider = MODELS[interviewer_name].provider
    sem = asyncio.Semaphore(concurrency)
    made = Counter()

    async def one(rec, i):
        async with sem:
            if store.spent(conn, provider) >= cap:
                return
            ai = store.agent_input(conn, rec["id"])
            if ai is None:
                return
            kind = ["typos", "sms", "dictated", "mixed"][i % 4]
            messy = corrupt(ai.candidate_response.text, seed=i, kind=kind)
            new = dataclasses.replace(ai, candidate_response=dataclasses.replace(ai.candidate_response, text=messy))
            try:
                out = await replay(new, interviewer)
            except Exception:
                traceback.print_exc(limit=2)
                return
            t = out["interviewer"].get("tokens") or {}
            _log(conn, interviewer_name, f"interviewer:{split}:pair", t.get("in", 0), t.get("out", 0))
            pair_id = f"pair-{rec['id']}"
            cand = dict(rec.get("candidate") or {}, style=f"messy:{kind}")
            out.update({"id": f"{rec['id']}-messy", "session_id": rec["session_id"], "turn": rec["turn"],
                        "question_id": rec["question_id"], "tags": rec.get("tags"), "candidate": cand,
                        "pair_id": pair_id, "pair_of": rec["id"]})
            store.put_record(conn, _annotate(out), task="interview", split=split, source="pair",
                             track=rec.get("track"))
            orig = dict(rec, pair_id=pair_id)
            conn.execute("update records set doc=? where id=?", (json.dumps(orig, default=str), rec["id"]))
            made["pairs"] += 1

    await asyncio.gather(*(one(rec, i) for i, rec in enumerate(base[:n])))
    print(dict(made), "total spent", round(store.spent(conn), 2))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["sessions", "edits", "pairs"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--interviewer", default="opus-5")
    ap.add_argument("--sessions", type=int, default=8)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--cap", type=float, default=60.0, help="stop when this provider's spend reaches $")
    ap.add_argument("--sim-model", default="gpt-5.4-mini")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--personas", nargs="*")
    ap.add_argument("--styles", nargs="*")
    ap.add_argument("--qids", nargs="*", type=int)
    ap.add_argument("--draw-rate", type=float, default=0.2)
    a = ap.parse_args(argv)
    if a.what == "sessions":
        asyncio.run(sessions(a.split, a.interviewer, a.sessions, concurrency=a.concurrency, cap=a.cap,
                             sim_model=a.sim_model, seed=a.seed, only_personas=a.personas,
                             only_styles=a.styles, only_qids=a.qids, draw_rate=a.draw_rate))
    elif a.what == "edits":
        asyncio.run(edits(a.split, a.interviewer, a.n, concurrency=a.concurrency, cap=a.cap, seed=a.seed))
    else:
        asyncio.run(pairs(a.split, a.interviewer, a.n, concurrency=a.concurrency, cap=a.cap, seed=a.seed))


if __name__ == "__main__":
    main()
