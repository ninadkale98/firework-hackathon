"""Re-run the judge with DeepSeek V4.1 Flash on Fireworks (no OpenAI credit needed).

Same prompts, criteria and schemas as ``distill.judge``; only the model and endpoint
change. Verdicts are stored under their own kinds (``judge_ds``, ``pairwise_ds``,
``safety_ds``) so the GPT-5.4 verdicts stay untouched and the two judges can be
compared on the same turns.

Caveat: DeepSeek V4.1 Flash is the teacher that wrote the training data, so it may
favour the tuned student. Read its numbers relative to the other models it judged,
and check its agreement with GPT-5.4 (``scripts/judge_ds.py --report``).

    python scripts/judge_ds.py --models reference qwen3.5-9b deepseek-v4.1-flash muse-glimmer-tuned
    python scripts/judge_ds.py --report
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import traceback
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import AsyncOpenAI  # noqa: E402

from distill import judge as J  # noqa: E402
from distill import store  # noqa: E402
from distill.models import FIREWORKS_URL, MODELS, cost  # noqa: E402
from distill.runs import REFERENCE, _runs, attempt_turn, stratified  # noqa: E402

JUDGE = "deepseek-v4.1-flash"
KINDS = {"judge": "judge_ds", "pairwise": "pairwise_ds", "safety": "safety_ds"}


async def judge(models, sample, concurrency, limit):
    conn = store.connect()
    api = AsyncOpenAI(api_key=os.environ["FIREWORKS_API_KEY"], base_url=FIREWORKS_URL, timeout=300.0)
    jm = MODELS[JUDGE].model
    picked = stratified(list(store.records(conn, split="test")), sample)
    if limit:
        picked = picked[:limit]
    runs = _runs(conn, "test")
    sem, done = asyncio.Semaphore(concurrency), Counter()

    def has(rid, model, kind):
        return conn.execute("select 1 from scores where record_id=? and model=? and kind=?",
                            (rid, model, kind)).fetchone() is not None

    def log(res):
        u = res.get("usage") or {}
        store.log_spend(conn, "fireworks", JUDGE, "judge_ds", u.get("in", 0), u.get("out", 0),
                        cost(JUDGE, u.get("in", 0), u.get("out", 0)))

    async def one(rec, model, out):
        async with sem:
            for attempt in range(5):
                try:
                    if not has(rec["id"], model, KINDS["judge"]):
                        res = await J.criteria_task(rec, out, model=jm, api=api)
                        store.put_score(conn, rec["id"], model, KINDS["judge"], res); log(res)
                    if model != REFERENCE and not has(rec["id"], model, KINDS["pairwise"]):
                        res = await J.pairwise_task(rec, out, rec["interviewer"], model=jm, api=api)
                        store.put_score(conn, rec["id"], model, KINDS["pairwise"], res); log(res)
                    if attempt_turn(rec) and not has(rec["id"], model, KINDS["safety"]):
                        res = await J.safety_task(rec, out, model=jm, api=api)
                        store.put_score(conn, rec["id"], model, KINDS["safety"], res); log(res)
                    done["ok"] += 1
                    break
                except Exception:
                    if attempt == 4:
                        done["error"] += 1
                        traceback.print_exc(limit=2)
                    else:
                        await asyncio.sleep(5 * 2 ** attempt)
            n = done["ok"] + done["error"]
            if n % 100 == 0:
                print(f"  {n}/{len(jobs)} errors={done['error']}", flush=True)

    jobs = []
    for rec in picked:
        for m in models:
            out = rec["interviewer"] if m == REFERENCE else runs.get(m, {}).get(rec["id"])
            if out and (m == REFERENCE or out.get("move")):
                jobs.append((rec, m, out))
    print(f"{len(picked)} turns, {len(jobs)} (turn, model) jobs, judge {jm}", flush=True)
    await asyncio.gather(*(one(*j) for j in jobs))
    print("done:", dict(done), "deepseek judge spend $",
          round(sum(u or 0 for (u,) in conn.execute(
              "select usd from spend where role='judge_ds'")), 2), flush=True)


def _leak_ok(j):
    return bool(j["leak_ok"]) if "leak_ok" in j else bool((j.get("no_spoon_feeding") or {}).get("pass"))


def report(models, sample):
    conn = store.connect()
    recs = {r["id"]: r for r in store.records(conn, split="test")}
    ids = {r["id"] for r in stratified(list(recs.values()), sample)}
    S = defaultdict(lambda: defaultdict(dict))
    for rid, model, kind, doc in conn.execute("select record_id, model, kind, doc from scores"):
        if rid in ids:
            S[model][rid][kind] = json.loads(doc)

    def pct(xs):
        return f"{100 * statistics.mean(xs):5.1f}%" if xs else "   – "

    print(f"\n300-turn sample. Judge columns: DeepSeek V4.1 Flash (GPT-5.4 in brackets where it exists).\n")
    hdr = f"{'model':<22}{'valid':>8}{'same move':>11}{'judge 1-5':>16}{'gives away':>18}{'win/tie Opus':>18}{'resists':>16}"
    print(hdr); print("-" * len(hdr))
    for m in models:
        rows = [S[m][i] for i in ids if "rules" in S[m].get(i, {})]
        if not rows:
            print(f"{m:<22} (no scored runs)"); continue
        valid = [float(bool(r["rules"].get("format_ok"))) for r in rows]
        same = [float(bool(r["rules"]["same_move"])) for r in rows if r["rules"].get("same_move") is not None]

        def cols(k_judge, k_pw, k_safe):
            jd = [r[k_judge] for r in rows if k_judge in r and r[k_judge].get("overall") is not None]
            ov = f"{statistics.mean(j['overall'] for j in jd):.2f}" if jd else "–"
            leak = pct([float(not _leak_ok(j)) for j in jd]).strip()
            pw = [r[k_pw]["winner"] for r in rows if k_pw in r]
            win = pct([float(w in ("a", "tie")) for w in pw]).strip()
            sf = [float(bool(r[k_safe]["ok"])) for r in rows if k_safe in r and r[k_safe].get("ok") is not None]
            return ov, leak, win, pct(sf).strip()
        d, g = cols("judge_ds", "pairwise_ds", "safety_ds"), cols("judge", "pairwise", "safety")
        cell = lambda a, b: f"{a} ({b})" if b not in ("–",) else a
        print(f"{m:<22}{pct(valid):>8}{pct(same):>11}{cell(d[0], g[0]):>16}{cell(d[1], g[1]):>18}"
              f"{cell(d[2], g[2]) if m != REFERENCE else '–':>18}{cell(d[3], g[3]):>16}")

    # Agreement between the two judges on the same (turn, model) pairs.
    both = [(S[m][i]["judge"], S[m][i]["judge_ds"]) for m in models for i in ids
            if "judge" in S[m].get(i, {}) and "judge_ds" in S[m].get(i, {})
            and S[m][i]["judge"].get("overall") is not None and S[m][i]["judge_ds"].get("overall") is not None]
    if both:
        exact = statistics.mean(float(a["overall"] == b["overall"]) for a, b in both)
        within1 = statistics.mean(float(abs(a["overall"] - b["overall"]) <= 1) for a, b in both)
        leak = statistics.mean(float(_leak_ok(a) == _leak_ok(b)) for a, b in both)
        print(f"\nJudge agreement, DeepSeek vs GPT-5.4, n={len(both)}: overall exact {exact:.0%}, "
              f"within 1 point {within1:.0%}, gives-away verdict {leak:.0%}")
        for m in models:
            pm = [(a, b) for mm in [m] for i in ids if "judge" in S[mm].get(i, {}) and "judge_ds" in S[mm].get(i, {})
                  for a, b in [(S[mm][i]["judge"], S[mm][i]["judge_ds"])]
                  if a.get("overall") is not None and b.get("overall") is not None]
            if pm:
                print(f"  {m:<22} GPT-5.4 mean {statistics.mean(a['overall'] for a, _ in pm):.2f}"
                      f"  DeepSeek mean {statistics.mean(b['overall'] for _, b in pm):.2f}  n={len(pm)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*",
                    default=[REFERENCE, "qwen3.5-9b", "deepseek-v4.1-flash", "muse-glimmer-tuned"])
    ap.add_argument("--sample", type=int, default=300)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.report:
        report(a.models, a.sample)
    else:
        asyncio.run(judge(a.models, a.sample, a.concurrency, a.limit))
