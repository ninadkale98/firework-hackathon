"""Does DeepSeek V4.1 Flash judge like GPT-5.4? Re-judge test turns that GPT-5.4
already judged (final judge) for Opus and base Qwen, and compare verdicts.
Stores DeepSeek's verdicts as kind ``judge_ds``; nothing is overwritten."""
import asyncio, json, os, random, sys
os.environ["DISTILL_JUDGE_MODEL"] = "deepseek-v4.1-flash"
from distill import store, judge as J
from distill.models import cost

N = int(sys.argv[1]) if len(sys.argv) > 1 else 75


async def main():
    conn = store.connect()
    recs = {r["id"]: r for r in store.records(conn, split="test")}
    runs = {rid: json.loads(d) for rid, d in conn.execute("select record_id, doc from runs where model='qwen3.5-9b'")}
    gpt = {}
    for rid, model, doc in conn.execute("select record_id, model, doc from scores where kind='judge' and model in ('reference','qwen3.5-9b')"):
        gpt[(rid, model)] = json.loads(doc)
    rng = random.Random(9)
    picks = []
    for model in ("reference", "qwen3.5-9b"):
        keys = sorted(k for k in gpt if k[1] == model and k[0] in recs)
        picks += rng.sample(keys, min(N, len(keys)))
    sem = asyncio.Semaphore(12)
    out, spend = {}, [0.0]

    async def one(rid, model):
        async with sem:
            o = recs[rid]["interviewer"] if model == "reference" else runs[rid]
            v = await J.criteria_task(recs[rid], o, model="deepseek-v4.1-flash")
            u = v.get("usage") or {}
            spend[0] += cost("deepseek-v4.1-flash", u.get("in", 0), u.get("out", 0)) or 0
            out[(rid, model)] = v
            store.put_score(conn, rid, model, "judge_ds", v)

    await asyncio.gather(*(one(r, m) for r, m in picks))
    print(f"judged {len(out)} turns with DeepSeek, ~${spend[0]:.2f}\n")
    for model in ("reference", "qwen3.5-9b"):
        ks = [k for k in out if k[1] == model and out[k].get("overall") is not None and gpt[k].get("overall") is not None]
        g = [gpt[k]["overall"] for k in ks]; d = [out[k]["overall"] for k in ks]
        within1 = sum(abs(a - b) <= 1 for a, b in zip(g, d)) / len(ks)
        gl = [gpt[k].get("leak_ok") for k in ks]; dl = [out[k].get("leak_ok") for k in ks]
        both = [(a, b) for a, b in zip(gl, dl) if a is not None and b is not None]
        agree = sum(a == b for a, b in both) / len(both) if both else float("nan")
        caught = sum(1 for a, b in both if a is False and b is False); gpt_fail = sum(1 for a, b in both if a is False)
        ds_extra = sum(1 for a, b in both if a is True and b is False)
        print(f"{model:<12} n={len(ks)}  mean overall GPT {sum(g)/len(g):.2f} vs DeepSeek {sum(d)/len(d):.2f}  "
              f"within 1 point {within1:.0%}")
        print(f"             leak verdict agreement {agree:.0%}; GPT flagged {gpt_fail}, DeepSeek also caught {caught}, "
              f"DeepSeek-only flags {ds_extra}")
    # rank agreement: does DeepSeek separate Opus from base Qwen like GPT does?
    import statistics as st
    for name, kind in (("GPT-5.4", gpt), ("DeepSeek", out)):
        ref = [kind[k]["overall"] for k in out if k[1] == "reference" and kind[k].get("overall") is not None]
        qw = [kind[k]["overall"] for k in out if k[1] == "qwen3.5-9b" and kind[k].get("overall") is not None]
        print(f"{name:<9} Opus {st.mean(ref):.2f}  base Qwen {st.mean(qw):.2f}  gap {st.mean(ref)-st.mean(qw):+.2f}")

asyncio.run(main())
