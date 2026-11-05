"""A/B the prompt variants in ``prompts/variants.py`` on the same test turns.

Default backend is a local Ollama model (free, nothing leaves the machine). The
Fireworks backend (v1 deployment) costs credit and only runs with ``--allow-fireworks``.

    python scripts/prompt_ab.py --per-mode 5                               # local qwen3:8b
    python scripts/prompt_ab.py --backend fireworks --allow-fireworks --per-mode 20   # v1

Scores with ``distill.score.score_output`` (the leaderboard's code checks): valid output,
legal move, same move and same grade as Opus. No AI judge.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import requests  # noqa: E402

from distill import store  # noqa: E402
from distill.runs import stratified  # noqa: E402
from distill.score import score_output  # noqa: E402
from prompts.variants import VARIANTS, build, parse  # noqa: E402

V1 = ("accounts/vishakha-d933-0nmp64/models/muse-starter-sl-codepraxis-muse-v1"
      "#accounts/vishakha-d933-0nmp64/deployments/eqb76xt9")


def wrapped(schema):
    return {"type": "object", "properties": {"move_object": schema},
            "required": ["move_object"], "additionalProperties": False}


def call_ollama(model, msgs, schema, max_tokens):
    body = {"model": model, "messages": msgs, "stream": False, "think": False,
            "options": {"temperature": 0, "num_ctx": 32768, "num_predict": max_tokens}}
    if schema is not None:
        body["format"] = schema
    r = requests.post("http://localhost:11434/api/chat", json=body, timeout=1800)
    r.raise_for_status()
    d = r.json()
    return d["message"]["content"], d.get("prompt_eval_count", 0), d.get("eval_count", 0), d.get("done_reason")


def call_fireworks(model, msgs, schema, max_tokens):
    from openai import OpenAI
    cl = OpenAI(base_url="https://api.fireworks.ai/inference/v1", api_key=os.environ["FIREWORKS_API_KEY"])
    kw = dict(model=model, messages=msgs, max_tokens=max_tokens, temperature=0)
    if schema is not None:
        kw["response_format"] = {"type": "json_schema", "json_schema": {"name": "move", "strict": True, "schema": schema}}
    o = cl.chat.completions.create(**kw)
    return o.choices[0].message.content or "", o.usage.prompt_tokens, o.usage.completion_tokens, o.choices[0].finish_reason


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["ollama", "fireworks"], default="ollama")
    ap.add_argument("--model", default=None)
    ap.add_argument("--allow-fireworks", action="store_true", help="Required to send traffic to Fireworks (costs credit).")
    ap.add_argument("--per-mode", type=int, default=5)
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS))
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.backend == "fireworks" and not a.allow_fireworks:
        sys.exit("Refusing to call Fireworks without --allow-fireworks (it costs credit).")
    model = a.model or ("qwen3:8b" if a.backend == "ollama" else V1)
    call = call_ollama if a.backend == "ollama" else call_fireworks
    if a.backend == "fireworks":
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")

    conn = store.connect()
    sample = stratified(list(store.records(conn, split="test")), 300)  # the leaderboard's sample
    picked = []
    for t in ("interview", "assess", "study"):
        picked += [r for r in sample if r["task"] == t][: a.per_mode]
    out = Path(a.out or ROOT / "prompts" / "results" / f"{a.backend}_{model.split('/')[-1].replace(':', '_').split('#')[0]}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {(d["record_id"], d["variant"]) for d in map(json.loads, out.open())}
    print(f"{len(picked)} turns x {len(a.variants)} variants on {a.backend}:{model} -> {out}", flush=True)
    with out.open("a") as f:
        for rec in picked:
            for v in a.variants:
                if (rec["id"], v) in done:
                    continue
                msgs, strict = build(rec, v)
                t0 = time.time()
                try:
                    text, tin, tout, stop = call(model, msgs, wrapped(rec["schema"]) if strict else None, a.max_tokens)
                    err = None
                except Exception as exc:
                    text, tin, tout, stop, err = "", 0, 0, None, repr(exc)[:300]
                move = parse(text, v)
                doc = {"raw": json.dumps({"move_object": move}) if move else text, "move": move,
                       "stop_reason": "end_turn" if stop in ("stop", "end_turn") else stop,
                       "tokens": {"in": tin, "out": tout}}
                s = score_output(rec, doc, reference=rec["interviewer"])
                row = {"record_id": rec["id"], "task": rec["task"], "variant": v, "secs": round(time.time() - t0, 1),
                       "tokens_in": tin, "tokens_out": tout, "error": err, "text": text[:4000],
                       **{k: s.get(k) for k in ("move", "format_ok", "legal_move", "same_move", "same_level")}}
                f.write(json.dumps(row) + "\n"); f.flush()
                print(f"  {rec['task']:<9} {v:<9} move={row['move']} ref={(rec['interviewer'].get('move') or {}).get('move')} "
                      f"ok={row['format_ok']} same={row['same_move']} {row['secs']}s{' ERR' if err else ''}", flush=True)
    report(out)


def report(path):
    rows = [json.loads(l) for l in open(path)]
    by = defaultdict(list)
    for r in rows:
        by[(r["variant"], r["task"])].append(r); by[(r["variant"], "all")].append(r)
    rate = lambda xs, k: (f"{sum(bool(x[k]) for x in xs if x[k] is not None)}/{sum(x[k] is not None for x in xs)}")
    print(f"\n{'variant':<10}{'mode':<11}{'valid':>8}{'legal':>8}{'same move':>11}{'same grade':>12}{'median s':>10}")
    for v in VARIANTS:
        for t in ("interview", "assess", "study", "all"):
            xs = by.get((v, t))
            if xs:
                print(f"{v:<10}{t:<11}{rate(xs,'format_ok'):>8}{rate(xs,'legal_move'):>8}{rate(xs,'same_move'):>11}"
                      f"{rate(xs,'same_level'):>12}{statistics.median(x['secs'] for x in xs):>10.1f}")


if __name__ == "__main__":
    main()
