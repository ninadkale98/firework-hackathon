"""Leaderboard metrics (EVALUATION.md section 3) for the Muse Glimmer runs, from either judge.

Computes the same per-model numbers as ``distill.site.collect`` (same sample, same
definitions, same bootstrap intervals) but lets you pick which judge's verdicts to
read: the original GPT-5.4 kinds (``judge`` / ``pairwise`` / ``safety``) or the
DeepSeek V4.1 Flash re-judge (``judge_ds`` / ``pairwise_ds`` / ``safety_ds``,
written by ``scripts/judge_ds.py``). Also reports judge agreement and paired
base-vs-tuned tests.

    python scripts/eval_report.py                  # DeepSeek judge, markdown to stdout
    python scripts/eval_report.py --judge gpt      # GPT-5.4 verdicts where they exist
    python scripts/eval_report.py --json results/muse_eval/board_ds.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from distill import store  # noqa: E402
from distill.runs import REFERENCE, stratified  # noqa: E402
from distill.score import bootstrap_ci  # noqa: E402

SAMPLE = 300
TASKS = ["interview", "assess", "study"]
MODELS = [REFERENCE, "deepseek-v4.1-flash", "qwen3.5-9b", "muse-glimmer-base", "muse-glimmer-tuned"]
LABEL = {
    REFERENCE: "Opus 5 (reference)",
    "deepseek-v4.1-flash": "DeepSeek V4.1 Flash (teacher)",
    "qwen3.5-9b": "Qwen3.5-9B base",
    "muse-glimmer-base": "Muse Glimmer 30B base",
    "muse-glimmer-tuned": "Muse Glimmer 30B tuned",
}


def _rate(vals: List[float]) -> Optional[Dict[str, Any]]:
    if not vals:
        return None
    lo, hi = bootstrap_ci(vals, n=1000, seed=0)
    return {"v": sum(vals) / len(vals), "lo": lo, "hi": hi, "n": len(vals)}


def _leak_ok(j: Dict[str, Any]) -> bool:
    if "leak_ok" in j:
        return bool(j["leak_ok"])
    return bool((j.get("no_spoon_feeding") or {}).get("pass"))


def load(suffix: str):
    conn = store.connect()
    recs = {r["id"]: r for r in store.records(conn, split="test")}
    ids = {r["id"] for r in stratified(list(recs.values()), SAMPLE)}
    want = {"rules": "rules", f"judge{suffix}": "judge", f"pairwise{suffix}": "pairwise",
            f"safety{suffix}": "safety", "judge": "judge_gpt", "judge_ds": "judge_ds"}
    S: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(lambda: defaultdict(dict))
    for rid, model, kind, doc in conn.execute("select record_id, model, kind, doc from scores"):
        if rid in ids and kind in want:
            S[model][rid][want[kind]] = json.loads(doc)
            if kind == f"judge{suffix}":
                S[model][rid]["judge"] = json.loads(doc)
    lat: Dict[str, Dict[str, float]] = defaultdict(dict)
    toks: Dict[str, Dict[str, int]] = defaultdict(dict)
    for rid, model, doc in conn.execute("select record_id, model, doc from runs"):
        if rid in ids:
            d = json.loads(doc)
            if d.get("latency_ms"):
                lat[model][rid] = d["latency_ms"]
            toks[model][rid] = (d.get("tokens") or {}).get("out") or 0
    for rid in ids:
        if recs[rid]["interviewer"].get("latency_ms"):
            lat[REFERENCE][rid] = recs[rid]["interviewer"]["latency_ms"]
    return recs, ids, S, lat, toks


def summarise(S, recs, lat, toks, model: str, ids) -> Optional[Dict[str, Any]]:
    """``distill.site.collect.summarise``, unchanged in its definitions."""
    rows = [dict(S[model][i], _task=recs[i]["task"]) for i in ids if "rules" in S[model].get(i, {})]
    if not rows:
        return None
    rules = [r["rules"] for r in rows]
    judged = [r for r in rows if "judge" in r and r["judge"].get("overall") is not None]
    pw = [r["pairwise"]["winner"] for r in rows if "pairwise" in r]

    def kept(r):
        ok = _leak_ok(r["judge"])
        if "safety" in r and r["safety"].get("ok") is not None:
            ok = ok and bool(r["safety"]["ok"])
        return ok

    closing = [r for r in judged if r["rules"].get("move") in ("PROBE", "QUESTION_DONE")
               and r["_task"] in ("interview", "assess")]
    l = sorted(lat[model][i] for i in ids if i in lat[model])
    t = [toks[model][i] for i in ids if i in toks[model]]
    return {
        "n": len(rows),
        "quality": _rate([(float(bool(r["rules"].get("format_ok"))) + float(kept(r)) + r["judge"]["overall"] / 5.0) / 3.0
                          for r in judged]),
        "format": _rate([float(bool(s.get("format_ok"))) for s in rules]),
        "rules": _rate([float(kept(r)) for r in judged]),
        "leak": _rate([float(not _leak_ok(r["judge"])) for r in judged]),
        "safety": _rate([float(bool(r["safety"]["ok"])) for r in rows
                         if "safety" in r and r["safety"].get("ok") is not None]),
        "reveal_closing": _rate([float(bool(any(isinstance(v, dict) and v.get("closed_stage")
                                                 for v in r["judge"].values()))) for r in closing]),
        "same_move": _rate([float(s["same_move"]) for s in rules if s.get("same_move") is not None]),
        "same_grade": _rate([float(s["same_level"]) for s in rules if s.get("same_level") is not None]),
        "judge": _rate([r["judge"]["overall"] for r in judged]) if judged else None,
        "win_tie": _rate([1.0 if w in ("a", "tie") else 0.0 for w in pw]) if pw else None,
        "p50_ms": l[len(l) // 2] if l else None,
        "median_out_tokens": statistics.median(t) if t else None,
    }


def mcnemar(a: List[bool], b: List[bool]) -> Dict[str, Any]:
    only_a = sum(1 for x, y in zip(a, b) if x and not y)
    only_b = sum(1 for x, y in zip(a, b) if y and not x)
    d = only_a + only_b
    tail = sum(math.comb(d, k) for k in range(min(only_a, only_b) + 1)) / 2 ** d if d else 1.0
    return {"n": len(a), "only_a": only_a, "only_b": only_b, "p_value": min(1.0, 2 * tail)}


COLS = [("quality", "Quality", "pct"), ("format", "Valid output", "pct"), ("rules", "Rules kept", "pct"),
        ("leak", "Gives answer away", "pct"), ("safety", "Resists fishing / injection", "pct"),
        ("reveal_closing", "Reveals answer when moving on", "pct"), ("same_move", "Same move as Opus", "pct"),
        ("same_grade", "Same grade as Opus (Assess)", "pct"), ("judge", "Judge 1–5", "num"),
        ("win_tie", "Judge prefers it or ties vs Opus", "pct"), ("p50_ms", "Latency p50", "ms"),
        ("median_out_tokens", "Median output tokens", "int")]


def cell(h, key, kind):
    v = h.get(key) if h else None
    if v is None:
        return "–"
    if kind == "ms":
        return f"{v / 1000:.1f} s"
    if kind == "int":
        return f"{v:,.0f}"
    if kind == "num":
        return f"{v['v']:.2f} [{v['lo']:.2f}–{v['hi']:.2f}]"
    return f"{100 * v['v']:.0f}% [{100 * v['lo']:.0f}–{100 * v['hi']:.0f}] n={v['n']}"


def table(out, models, scope):
    lines = ["| Metric | " + " | ".join(LABEL[m] for m in models) + " |",
             "|---|" + "---|" * len(models)]
    for key, name, kind in COLS:
        lines.append(f"| {name} | " + " | ".join(cell(out[m][scope], key, kind) for m in models) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", choices=["ds", "gpt"], default="ds")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    recs, ids, S, lat, toks = load("_ds" if a.judge == "ds" else "")
    models = [m for m in MODELS if S.get(m)]
    out = {m: {"overall": summarise(S, recs, lat, toks, m, ids),
               **{t: summarise(S, recs, lat, toks, m, [i for i in ids if recs[i]["task"] == t]) for t in TASKS}}
           for m in models}
    judge_name = "DeepSeek V4.1 Flash" if a.judge == "ds" else "GPT-5.4"
    print(f"# Leaderboard metrics, 300-turn shared sample, judge: {judge_name}\n")
    print("Values are rates with 95% bootstrap intervals and n. Definitions: EVALUATION.md section 3.\n")
    print("## All modes\n"); print(table(out, models, "overall"))
    for t in TASKS:
        print(f"\n## {t.title()}\n"); print(table(out, models, t))

    # Paired base vs tuned on identical turns.
    b, tu = "muse-glimmer-base", "muse-glimmer-tuned"
    if b in S and tu in S:
        print("\n## Base vs tuned Muse Glimmer, paired on the same turns (exact McNemar)\n")
        print("| Outcome | base only | tuned only | n | p |\n|---|---|---|---|---|")
        for name, f in [("Valid output", lambda r: bool(r["rules"].get("format_ok"))),
                        ("Same move as Opus", lambda r: r["rules"].get("same_move")),
                        ("Did not give answer away", lambda r: _leak_ok(r["judge"]) if "judge" in r else None),
                        ("Judge score ≥ 4", lambda r: r["judge"]["overall"] >= 4 if r.get("judge", {}).get("overall") is not None else None)]:
            pairs = [(f(S[b][i]), f(S[tu][i])) for i in ids if "rules" in S[b].get(i, {}) and "rules" in S[tu].get(i, {})]
            pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
            r = mcnemar([x for x, _ in pairs], [y for _, y in pairs])
            print(f"| {name} | {r['only_a']} | {r['only_b']} | {r['n']} | {r['p_value']:.2g} |")

    # Agreement between the two judges where both scored the same reply.
    pairs = [(S[m][i]["judge_gpt"], S[m][i]["judge_ds"]) for m in models for i in ids
             if "judge_gpt" in S[m].get(i, {}) and "judge_ds" in S[m].get(i, {})
             and S[m][i]["judge_gpt"].get("overall") is not None and S[m][i]["judge_ds"].get("overall") is not None]
    if pairs:
        print(f"\n## Judge agreement: DeepSeek V4.1 Flash vs GPT-5.4 on the same replies (n={len(pairs)})\n")
        print(f"- Overall score identical: {statistics.mean(float(g['overall'] == d['overall']) for g, d in pairs):.0%}; "
              f"within 1 point: {statistics.mean(float(abs(g['overall'] - d['overall']) <= 1) for g, d in pairs):.0%}")
        print(f"- 'Gives answer away' verdict identical: {statistics.mean(float(_leak_ok(g) == _leak_ok(d)) for g, d in pairs):.0%}")
        print("\n| Model | GPT-5.4 mean | DeepSeek mean | n |\n|---|---|---|---|")
        for m in models:
            pm = [(S[m][i]["judge_gpt"]["overall"], S[m][i]["judge_ds"]["overall"]) for i in ids
                  if "judge_gpt" in S[m].get(i, {}) and "judge_ds" in S[m].get(i, {})
                  and S[m][i]["judge_gpt"].get("overall") is not None and S[m][i]["judge_ds"].get("overall") is not None]
            if pm:
                print(f"| {LABEL[m]} | {statistics.mean(g for g, _ in pm):.2f} | {statistics.mean(d for _, d in pm):.2f} | {len(pm)} |")
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps({"judge": judge_name, "sample": SAMPLE, "models": out}, indent=2, default=str))


if __name__ == "__main__":
    main()
