"""Run the tuned Muse Glimmer deployment on test records without the CodePraxis backend.

Replicates the backend's ``OpenAISchemaCompleter`` call as far as the stored runs show it:
the system blocks joined with blank lines (as ``distill.export`` does for training), the
user message, and strict JSON-schema output with the turn's schema wrapped as
``{"move_object": ...}``. Writes ``runs`` rows in the same shape as ``distill.runs``.

    python scripts/run_tuned.py --sample 300 --concurrency 8
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import AsyncOpenAI  # noqa: E402

from distill import store  # noqa: E402
from distill.models import FIREWORKS_URL, MODELS  # noqa: E402
from distill.runs import stratified  # noqa: E402
from prompts.variants import build, parse  # noqa: E402

DEFAULT_NAME = "muse-glimmer-tuned"


def wrapped(schema):
    return {"type": "object", "properties": {"move_object": schema},
            "required": ["move_object"], "additionalProperties": False}


async def main(a) -> None:
    conn = store.connect()
    recs = list(store.records(conn, split=a.split))
    if a.sample:
        recs = stratified(recs, a.sample)
    if a.limit:
        recs = recs[: a.limit]
    NAME = a.name
    spec = MODELS[NAME]
    client = AsyncOpenAI(api_key=os.environ["FIREWORKS_API_KEY"], base_url=FIREWORKS_URL, timeout=300.0)
    sem, done = asyncio.Semaphore(a.concurrency), Counter()
    print(f"{len(recs)} records -> {spec.model}", flush=True)

    async def one(rec):
        async with sem:
            if store.has_run(conn, rec["id"], NAME):
                done["cached"] += 1
                return
            msgs, strict = build(rec, a.variant)
            kw = dict(model=spec.model, max_tokens=a.max_tokens, temperature=0, messages=msgs)
            if strict and not a.no_schema:
                kw["response_format"] = {"type": "json_schema", "json_schema": {
                    "name": "move", "strict": True, "schema": wrapped(rec["schema"])}}
            t0 = time.time()
            for attempt in range(6):
                try:
                    out = await client.chat.completions.create(**kw)
                    break
                except Exception as exc:  # 429 / 503 scaling up: back off
                    if attempt == 5:
                        store.put_run(conn, rec["id"], NAME, {"error": repr(exc)[:500], "move": None, "raw": None})
                        done["error"] += 1
                        return
                    await asyncio.sleep(5 * 2 ** attempt)
            raw = out.choices[0].message.content or ""
            move = parse(raw, a.variant)
            finish = out.choices[0].finish_reason
            store.put_run(conn, rec["id"], NAME, {
                "model": spec.model, "stop_reason": "end_turn" if finish == "stop" else finish,
                "raw": json.dumps({"move_object": move}) if (move and a.variant == "cot") else raw, "move": move, "latency_ms": int((time.time() - t0) * 1000),
                "tokens": {"in": out.usage.prompt_tokens, "out": out.usage.completion_tokens,
                           "cache_read": 0},
                "schema_mode": "json_schema_strict" if (strict and not a.no_schema) else "none",
                "prompt_variant": a.variant,
            })
            store.log_spend(conn, "fireworks", NAME, f"baseline:{a.split}",
                            out.usage.prompt_tokens, out.usage.completion_tokens, None)
            done["ok"] += 1
            if done["ok"] % 25 == 0:
                print(f"  {done['ok']}/{len(recs)} errors={done['error']}", flush=True)

    await asyncio.gather(*(one(r) for r in recs))
    print(f"{NAME}: {dict(done)}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default=DEFAULT_NAME, help="model name in distill.models.MODELS")
    ap.add_argument("--split", default="test")
    ap.add_argument("--sample", type=int, default=300)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--no-schema", action="store_true")
    ap.add_argument("--variant", default="original", choices=["original", "optimized", "cot"], help="prompts/variants.py")
    asyncio.run(main(ap.parse_args()))
