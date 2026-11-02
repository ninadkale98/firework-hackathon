"""Run models on the frozen test set, score them, judge them, rank them.

    python -m distill.runs run   --models haiku-5.5 gpt-5.4 kimi-k3 ... --split test
    python -m distill.runs score --split test
    python -m distill.runs judge --split test --sample 400 --models ...
    python -m distill.runs board --split test

Every model gets the exact system blocks, user message and schema the reference
interviewer got (``records.doc``). The reference itself (Opus on the test set) is
scored as model ``reference`` so it sits on the same board.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import traceback
from collections import Counter, defaultdict
from typing import Any, Dict, List

from distill import store
from distill.models import MODELS, completer, cost
from distill.score import headline, score_output

REFERENCE = "reference"


def _reply_doc(reply) -> Dict[str, Any]:
    return {
        "model": reply.model, "stop_reason": reply.stop_reason, "raw": reply.raw_text,
        "move": reply.move, "latency_ms": reply.latency_ms,
        "tokens": {"in": reply.input_tokens, "out": reply.output_tokens,
                   "cache_read": reply.cache_read_input_tokens},
    }


def stratified(recs: List[Dict[str, Any]], n: int, seed: int = 0) -> List[Dict[str, Any]]:
    """``n`` records spread evenly over (task, persona), the same ``n`` for every
    model given the same records and seed."""
    rng = random.Random(seed)
    by = defaultdict(list)
    for r in sorted(recs, key=lambda r: r["id"]):
        by[(r["task"], (r.get("buckets") or {}).get("persona", "-"))].append(r)
    for v in by.values():
        rng.shuffle(v)
    picked: List[Dict[str, Any]] = []
    while len(picked) < min(n, len(recs)):
        for key in sorted(by):
            if by[key] and len(picked) < n:
                picked.append(by[key].pop())
    return picked


async def run(models: List[str], split: str, *, concurrency: int, caps: Dict[str, float],
              limit: int = 0, sample: int = 0) -> None:
    conn = store.connect()
    recs = list(store.records(conn, split=split))
    if sample:
        recs = stratified(recs, sample)
    if limit:
        recs = recs[:limit]
    print(f"{len(recs)} records x {len(models)} models")

    async def per_model(name: str):
        spec = MODELS[name]
        comp = completer(name)
        sem = asyncio.Semaphore(concurrency)
        done = Counter()

        async def one(rec):
            async with sem:
                if store.has_run(conn, rec["id"], name):
                    done["cached"] += 1
                    return
                if store.spent(conn, spec.provider) >= caps.get(spec.provider, 1e9):
                    done["cap"] += 1
                    return
                try:
                    reply = await comp.complete(system=rec["system"], message=rec["message"],
                                                schema=rec["schema"])
                    doc = _reply_doc(reply)
                    store.log_spend(conn, spec.provider, name, f"baseline:{split}",
                                    reply.input_tokens, reply.output_tokens,
                                    cost(name, reply.input_tokens, reply.output_tokens))
                except Exception as exc:
                    doc = {"error": repr(exc)[:500], "move": None, "raw": None}
                    done["error"] += 1
                store.put_run(conn, rec["id"], name, doc)
                done["ok"] += 1
                if done["ok"] % 50 == 0:
                    print(f"  {name}: {done['ok']}/{len(recs)} errors={done['error']}", flush=True)

        await asyncio.gather(*(one(r) for r in recs))
        print(f"{name}: {dict(done)}", flush=True)

    await asyncio.gather(*(per_model(m) for m in models))
    print("spend:", {p: round(store.spent(conn, p), 2) for p in ("anthropic", "openai", "fireworks")})


def _runs(conn, split: str) -> Dict[str, Dict[str, Dict[str, Any]]]:
    out: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    ids = {r["id"] for r in store.records(conn, split=split)}
    for rid, model, doc in conn.execute("select record_id, model, doc from runs"):
        if rid in ids:
            out[model][rid] = json.loads(doc)
    return out


def score(split: str) -> None:
    conn = store.connect()
    recs = {r["id"]: r for r in store.records(conn, split=split)}
    for rid, rec in recs.items():
        ref = rec["interviewer"]
        store.put_score(conn, rid, REFERENCE, "rules", score_output(rec, ref, reference=None))
    for model, runs in _runs(conn, split).items():
        for rid, doc in runs.items():
            rec = recs[rid]
            store.put_score(conn, rid, model, "rules", score_output(rec, doc, reference=rec["interviewer"]))
    print("scored", len(recs), "records")


async def judge(split: str, models: List[str], sample: int, *, concurrency: int, cap: float,
                pairwise: bool, seed: int) -> None:
    from distill import judge as J

    conn = store.connect()
    recs = list(store.records(conn, split=split))
    # The same stratified sample ``run --run-sample`` uses, so the models that only
    # answered the sample are judged on exactly the turns they answered.
    picked = stratified(recs, sample)
    runs = _runs(conn, split)
    sem = asyncio.Semaphore(concurrency)
    done = Counter()
    prov = _judge_provider(J)

    def judged(rid, model, kind):
        return conn.execute("select 1 from scores where record_id=? and model=? and kind=?",
                            (rid, model, J.kind(kind))).fetchone() is not None

    async def one(rec, model, out):
        async with sem:
            if store.spent(conn, prov) >= cap:
                done["cap"] += 1
                return
            try:
                if not judged(rec["id"], model, "judge"):
                    res = await J.criteria_task(rec, out)
                    store.put_score(conn, rec["id"], model, J.kind("judge"), res)
                    u = res.get("usage") or {}
                    store.log_spend(conn, prov, J.DEFAULT_MODEL, "judge", u.get("in", 0), u.get("out", 0),
                                    cost(J.DEFAULT_MODEL, u.get("in", 0), u.get("out", 0)))
                if pairwise and model != REFERENCE and not judged(rec["id"], model, "pairwise"):
                    res = await J.pairwise_task(rec, out, rec["interviewer"])
                    store.put_score(conn, rec["id"], model, J.kind("pairwise"), res)
                    u = res.get("usage") or {}
                    store.log_spend(conn, prov, J.DEFAULT_MODEL, "judge", u.get("in", 0), u.get("out", 0),
                                    cost(J.DEFAULT_MODEL, u.get("in", 0), u.get("out", 0)))
                done["ok"] += 1
            except Exception:
                done["error"] += 1
                traceback.print_exc(limit=2)

    jobs = []
    for rec in picked:
        jobs.append(one(rec, REFERENCE, rec["interviewer"]))
        for m in models:
            out = runs.get(m, {}).get(rec["id"])
            if out and out.get("move"):
                jobs.append(one(rec, m, out))
    await asyncio.gather(*jobs)
    print("judge:", dict(done), f"{prov} spend", round(store.spent(conn, prov), 2))


def _judge_provider(J) -> str:
    return MODELS[J.DEFAULT_MODEL].provider if J.DEFAULT_MODEL in MODELS else "openai"


def attempt_turn(rec: Dict[str, Any]) -> bool:
    """A turn where the candidate tried manipulation or asked for the answer: the
    simulator's own flag, its persona, or the Study message kind."""
    from distill.score import is_injection_attempt

    cand = rec.get("candidate") or {}
    if cand.get("injection_attempt"):
        return True
    if cand.get("persona") in ("injection", "asks_answer") or cand.get("kind") in ("injection", "asks_answer"):
        return True
    return is_injection_attempt(rec.get("message") or "")


async def safety(split: str, models: List[str], sample: int, *, concurrency: int, cap: float) -> None:
    """The safety judge on the sample's manipulation and answer-fishing turns, for
    the reference and every model that answered them."""
    from distill import judge as J

    conn = store.connect()
    picked = [r for r in stratified(list(store.records(conn, split=split)), sample) if attempt_turn(r)]
    runs = _runs(conn, split)
    sem = asyncio.Semaphore(concurrency)
    done = Counter()
    prov = _judge_provider(J)

    async def one(rec, model, out):
        async with sem:
            if conn.execute("select 1 from scores where record_id=? and model=? and kind=?",
                            (rec["id"], model, J.kind("safety"))).fetchone() or store.spent(conn, prov) >= cap:
                return
            try:
                res = await J.safety_task(rec, out)
                store.put_score(conn, rec["id"], model, J.kind("safety"), res)
                u = res.get("usage") or {}
                store.log_spend(conn, prov, J.DEFAULT_MODEL, "judge:safety", u.get("in", 0), u.get("out", 0),
                                cost(J.DEFAULT_MODEL, u.get("in", 0), u.get("out", 0)))
                done["ok"] += 1
            except Exception:
                done["error"] += 1
                traceback.print_exc(limit=2)

    jobs = []
    for rec in picked:
        jobs.append(one(rec, REFERENCE, rec["interviewer"]))
        for m in models:
            out = runs.get(m, {}).get(rec["id"])
            if out and out.get("move"):
                jobs.append(one(rec, m, out))
    print(f"{len(picked)} attempt turns in the sample, {len(jobs)} judge calls")
    await asyncio.gather(*jobs)
    print("safety:", dict(done), f"{prov} spend", round(store.spent(conn, prov), 2))


def board(split: str) -> Dict[str, Any]:
    from distill import judge as J

    conn = store.connect()
    recs = {r["id"]: r for r in store.records(conn, split=split)}
    rows: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    extra: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(lambda: defaultdict(dict))
    raw: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(lambda: defaultdict(dict))
    for rid, model, kind, doc in conn.execute("select record_id, model, kind, doc from scores"):
        if rid in recs:
            raw[model][rid][kind] = json.loads(doc)
    for model, by_rid in raw.items():
        for rid, kinds in by_rid.items():
            kinds = J.judge_kinds(kinds)
            if "rules" in kinds:
                rows[model][recs[rid]["task"]].append(kinds.pop("rules"))
            extra[model][rid].update(kinds)
    out: Dict[str, Any] = {}
    for model, by_task in rows.items():
        out[model] = {}
        for task, scores in by_task.items():
            for s in scores:
                j = extra[model].get(s.get("record_id"), {}).get("judge")
                if j and j.get("overall") is not None:
                    s["judge_overall"] = j["overall"]
            h = headline(scores)
            pw = [extra[model][s["record_id"]]["pairwise"]["winner"] for s in scores
                  if "pairwise" in extra[model].get(s.get("record_id"), {})]
            if pw:
                h["win_or_tie_vs_reference"] = sum(w in ("a", "tie") for w in pw) / len(pw)
                h["pairwise_n"] = len(pw)
            spec = MODELS.get(model)
            tin = [s.get("tokens_in") or 0 for s in scores]
            tout = [s.get("tokens_out") or 0 for s in scores]
            if spec and spec.price:
                h["usd_per_turn"] = cost(model, int(statistics.mean(tin)), int(statistics.mean(tout)))
            out[model][task] = h
    path = store.ROOT / "data" / f"board_{split}.json"
    path.write_text(json.dumps(out, indent=2, default=str))
    _print(out)
    return out


def _print(out: Dict[str, Any]) -> None:
    keys = ["format_ok", "rules_ok", "leak", "same_move", "judge_overall", "win_or_tie_vs_reference"]
    for task in sorted({t for m in out.values() for t in m}):
        print(f"\n== {task}")
        print(f"{'model':<22}" + "".join(f"{k:>18}" for k in keys) + f"{'p50 ms':>10}{'$/turn':>10}")
        for model, by_task in sorted(out.items()):
            h = by_task.get(task)
            if not h:
                continue
            cells = []
            for k in keys:
                v = h.get(k)
                if isinstance(v, dict):
                    v = v.get("rate", v.get("mean"))
                cells.append(f"{v:>18.3f}" if isinstance(v, (int, float)) else f"{'-':>18}")
            lat = h.get("latency_ms") or {}
            p50 = lat.get("p50") if isinstance(lat, dict) else None
            usd = h.get("usd_per_turn")
            print(f"{model:<22}" + "".join(cells) + (f"{p50:>10.0f}" if p50 else f"{'-':>10}")
                  + (f"{usd:>10.4f}" if usd is not None else f"{'-':>10}"))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["run", "score", "judge", "safety", "board"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--models", nargs="*", default=[])
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sample", type=int, default=400, help="judge: records to judge")
    ap.add_argument("--run-sample", type=int, default=0, help="run: stratified sample size (0 = all)")
    ap.add_argument("--pairwise", action="store_true")
    ap.add_argument("--cap-anthropic", type=float, default=80)
    ap.add_argument("--cap-openai", type=float, default=150)
    ap.add_argument("--cap-fireworks", type=float, default=40)
    ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args(argv)
    caps = {"anthropic": a.cap_anthropic, "openai": a.cap_openai, "fireworks": a.cap_fireworks}
    if a.what == "run":
        asyncio.run(run(a.models, a.split, concurrency=a.concurrency, caps=caps, limit=a.limit,
                        sample=a.run_sample))
    elif a.what == "score":
        score(a.split)
    elif a.what in ("judge", "safety"):
        from distill import judge as J

        cap = caps[_judge_provider(J)] if _judge_provider(J) in caps else a.cap_openai
        if a.what == "judge":
            asyncio.run(judge(a.split, a.models, a.sample, concurrency=a.concurrency,
                              cap=cap, pairwise=a.pairwise, seed=a.seed))
        else:
            asyncio.run(safety(a.split, a.models, a.sample, concurrency=a.concurrency, cap=cap))
    else:
        board(a.split)


if __name__ == "__main__":
    main()
