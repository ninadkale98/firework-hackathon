"""Judge sanity check: does the fixed judge score Opus (the reference) high, and
still catch the base 9B's real problems? Stores verdicts as kind ``judge_v2``."""
import asyncio, json, random, sys
from collections import Counter, defaultdict
from distill import store, judge as J
from distill.runs import stratified, attempt_turn
from distill.models import cost

EXAMPLES = ["4b5dda505aee-t5-c0-messy", "e8b7bc0acbbf-t1-c0", "9d6426037745-t4-c0", "4f2d46cc0573-t3-c0",
            "5d3ce25f336c-t11-c0", "e47ba700783c-t4-c0", "d71c82b9766e-t5-c0", "83523bb5f916-t3-c0"]
PER_MODE = int(sys.argv[1]) if len(sys.argv) > 1 else 30


async def main():
    conn = store.connect()
    recs = {r["id"]: r for r in store.records(conn, split="test")}
    sample = stratified(list(recs.values()), 300)
    rng = random.Random(42)
    picked = list(EXAMPLES)
    for t in ("interview", "assess", "study"):
        pool = [r["id"] for r in sample if r["task"] == t and r["id"] not in picked]
        picked += rng.sample(pool, min(PER_MODE, len(pool)))
    qwen = {rid: json.loads(d) for rid, d in conn.execute("select record_id, doc from runs where model='qwen3.5-9b'")}
    old = {rid: json.loads(d) for rid, d in conn.execute("select record_id, doc from scores where model='reference' and kind='judge'")}
    sem = asyncio.Semaphore(12)
    res = defaultdict(dict)
    spend = [0.0]

    async def one(rid, model, out):
        async with sem:
            v = await J.criteria_task(recs[rid], out)
            if attempt_turn(recs[rid]):
                v["safety"] = await J.safety_task(recs[rid], out)
            for u in (v.get("usage") or {}, (v.get("safety") or {}).get("usage") or {}):
                spend[0] += cost("gpt-5.4", u.get("in", 0), u.get("out", 0)) or 0
            res[model][rid] = v
            store.put_score(conn, rid, model, "judge_v2", v)

    jobs = [one(rid, "reference", recs[rid]["interviewer"]) for rid in picked]
    jobs += [one(rid, "qwen3.5-9b", qwen[rid]) for rid in picked if rid in qwen and qwen[rid].get("move")]
    await asyncio.gather(*jobs)

    print(f"judged {len(jobs)} outputs, ~${spend[0]:.2f}\n")
    print("== the 8 examples (Opus): old -> new")
    for rid in EXAMPLES:
        n, o = res["reference"].get(rid, {}), old.get(rid, {})
        fails = [k for k, v in n.items() if isinstance(v, dict) and v.get("pass") is False]
        print(f"  {recs[rid]['task']:<9} overall {o.get('overall')} -> {n.get('overall')}  now failing: {fails or 'none'}"
              f"{'  (a leak verdict was discarded: quote not in reply)' if n.get('leak_unverified') else ''}")
    for model in ("reference", "qwen3.5-9b"):
        print(f"\n== {model}")
        for t in ("interview", "assess", "study"):
            vs = [v for rid, v in res[model].items() if recs[rid]["task"] == t and rid not in EXAMPLES]
            if not vs:
                continue
            ov = Counter(v.get("overall") for v in vs)
            fails = Counter(k for v in vs for k, x in v.items() if isinstance(x, dict) and x.get("pass") is False)
            leak = sum(1 for v in vs if v.get("leak_ok") is False)
            saf = [v["safety"]["ok"] for v in vs if "safety" in v and "ok" in v["safety"]]
            mean = sum(v.get("overall") or 0 for v in vs) / len(vs)
            print(f"  {t:<9} n={len(vs)} mean={mean:.2f} overall={dict(sorted(ov.items()))} gives-away={leak}/{len(vs)} "
                  f"safety ok={sum(saf)}/{len(saf)} unverified-leaks-dropped={sum(1 for v in vs if v.get('leak_unverified'))}")
            print(f"            failed criteria: {dict(fails.most_common())}")

asyncio.run(main())
