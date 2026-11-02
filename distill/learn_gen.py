"""Make Study and Assess records: simulated Learn sessions with varied setups.

    python -m distill.learn_gen --task study  --split test --sessions 40
    python -m distill.learn_gen --task assess --split test --sessions 40

Each session draws a setup over the knobs ``learn_engine`` exposes (notes, gaps,
memory, ticks, Fix my gaps, code-run outcomes, earlier chat, Assess start stage and
turns already spent), a persona and a style, and for Study the first message kind,
cycled so every kind leads some sessions. The Tutor for the test split is Opus at
production's per-phase effort (Study low, Assess medium).
"""

from __future__ import annotations

import argparse
import asyncio
import random
import traceback
from collections import Counter
from typing import Any, Dict

from distill import store
from distill.bank import SPLITS, load
from distill.learn_engine import STUDY_KINDS, LearnLearner, run_assess, run_study
from distill.models import MODELS, completer, cost
from distill.simulator import PERSONAS, Profile

OUTCOMES = ["pass", "fail", "compile_error", "timeout", "runtime_error"]


def tutor_pair(name: str):
    """(study, assess) completers: production's effort per phase for Claude."""
    spec = MODELS[name]
    if spec.provider == "anthropic":
        from app.services.interview.ai.anthropic_client import AnthropicSchemaCompleter

        mk = lambda effort: AnthropicSchemaCompleter(model=spec.model, effort=effort, effort_env=None,
                                                      timeout_seconds=240.0)
        return mk("low"), mk("medium")
    c = completer(name)
    return c, c


def draw_setup(rng: random.Random, task: str) -> Dict[str, Any]:
    setup: Dict[str, Any] = {
        "notes_preset": rng.choices(["", "short", "long", "injection"], [3, 3, 2, 1])[0],
        "done_submodules": rng.choice([0, 0, 1, 3, "all_but_one"]),
        "prior_gaps": rng.choice([0, 0, 2, 5, 10]),
        "memory": rng.choice([{}, {}, {"doubts": 2}, {"doubts": 3, "confusions": 2},
                              {"doubts": 5, "confusions": 4}]),
        "ticks": rng.choice(["none", "some", "all"]),
        "study_prefix": rng.choice([0, 2, 5, 10]),
        "check_outcomes": rng.sample(OUTCOMES, rng.choice([0, 1, 2])),
    }
    if rng.random() < 0.15:
        setup["gap_fix"] = True
    if task == "assess":
        setup["assess_start_stage"] = rng.choice([0, 0, 2, 4, 6])
        setup["assess_stage_turns"] = rng.choice([0, 0, 2, 5])
        setup["check_outcomes"] = setup["check_outcomes"] + rng.sample(["submit_pass", "submit_fail"], rng.choice([0, 1]))
    return {k: v for k, v in setup.items() if v not in ("", None)}


async def generate(task: str, split: str, tutor_name: str, n: int, *, concurrency: int, cap: float,
                   sim_model: str, seed: int) -> None:
    conn = store.connect()
    bank = load()
    qids = list(SPLITS["learn"][split])
    rng = random.Random(seed)
    tutors = tutor_pair(tutor_name)
    provider = MODELS[tutor_name].provider
    kinds = list(STUDY_KINDS)
    personas = list(PERSONAS)
    sem = asyncio.Semaphore(concurrency)
    done = Counter()

    async def one(i: int):
        async with sem:
            if store.spent(conn, provider) >= cap:
                done["cap"] += 1
                return
            r = random.Random(rng.randrange(1 << 30) + i)
            qid = qids[i % len(qids)]
            profile = Profile.draw(r, persona=personas[i % len(personas)])
            first = [kinds[i % len(kinds)]] if task == "study" else []
            learner = LearnLearner(profile, model=sim_model, kinds=first)
            setup = draw_setup(r, task)
            tags = {"persona": profile.persona, "style": profile.style, "setup": setup}
            try:
                if task == "study":
                    out = await run_study(qid, bank=bank, tutor=tutors, learner=learner,
                                          max_turns=r.choice([5, 7, 9]), tags=tags, setup=setup, seed=i)
                else:
                    out = await run_assess(qid, bank=bank, tutor=tutors, learner=learner,
                                           max_turns=12, tags=tags, setup=setup, seed=i)
            except Exception:
                done["failed"] += 1
                traceback.print_exc(limit=3)
                return
            finally:
                store.log_spend(conn, "openai", sim_model, "simulator:learn", learner.usage["in"],
                                learner.usage["out"], cost(sim_model, learner.usage["in"], learner.usage["out"]))
            for rec in out["records"]:
                t = rec["interviewer"].get("tokens") or {}
                store.log_spend(conn, provider, tutor_name, f"tutor:{split}:{task}", t.get("in", 0),
                                t.get("out", 0), cost(tutor_name, t.get("in", 0), t.get("out", 0)))
                cand = rec.get("candidate") or {}
                b = rec.setdefault("buckets", {})
                b.setdefault("persona", cand.get("persona", profile.persona))
                b.setdefault("style", cand.get("style", profile.style))
                if task == "study":
                    b.setdefault("kind", cand.get("kind", "-"))
                rec.setdefault("question_id", qid)
                store.put_record(conn, rec, task=rec.get("task", task), split=split, source="sim", track="learn")
            store.put_session(conn, out, split=split, task=task)
            done["sessions"] += 1
            done["records"] += len(out["records"])
            print(f"[{done['sessions']}/{n}] {task} q{qid} {profile.persona}/{profile.style} "
                  f"turns={out['turns']} records={len(out['records'])} "
                  f"spent={store.spent(conn, provider):.2f}", flush=True)

    await asyncio.gather(*(one(i) for i in range(n)))
    print(dict(done), "spend", {p: round(store.spent(conn, p), 2) for p in ("anthropic", "openai")})


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["study", "assess"], required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--tutor", default="opus-5")
    ap.add_argument("--sessions", type=int, default=10)
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--cap", type=float, default=100.0)
    ap.add_argument("--sim-model", default="gpt-5.4-mini")
    ap.add_argument("--seed", type=int, default=5)
    a = ap.parse_args(argv)
    asyncio.run(generate(a.task, a.split, a.tutor, a.sessions, concurrency=a.concurrency, cap=a.cap,
                         sim_model=a.sim_model, seed=a.seed))


if __name__ == "__main__":
    main()
