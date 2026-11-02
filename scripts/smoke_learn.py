"""Smoke test: one short Study and one short Assess session on question 615, run
through the production Learn code with gpt-5.4-mini as Tutor and as learner.

    cd hackathon/fireworks-distill
    PYTHONPATH=../../backend:. ../../backend/.venv/bin/python scripts/smoke_learn.py
"""

from __future__ import annotations

import asyncio
import json
import random
import sys
from pathlib import Path

from distill import bank as local_bank
from distill.learn_engine import LearnLearner, run_assess, run_study
from distill.models import completer, cost
from distill.simulator import Profile

QID = 615
TUTOR = "gpt-5.4-mini"
OUT = Path(__file__).resolve().parent.parent / "data" / "smoke_learn.json"


def show(result) -> int:
    print(f"\n=== {result['task'].upper()} session {result['session_id']}: {result['turns']} turns, "
          f"completed={result['completed']}, phase={result['phase']}")
    tin = tout = 0
    for r in result["records"]:
        it = r["interviewer"]
        tok = it.get("tokens") or {}
        tin += tok.get("in") or 0
        tout += tok.get("out") or 0
        move = (it.get("move") or {}).get("move")
        print(f"\n-- {r['id']} task={r['task']} kind={r['candidate'].get('kind')} runs={r['candidate'].get('runs')}")
        print(f"   offered: {r['offered']}")
        print(f"   chosen:  {move}   served: {(r.get('served') or {}).get('move')} "
              f"{(r.get('served') or {}).get('downgraded') or ''}")
        print(f"   tokens:  in={tok.get('in')} out={tok.get('out')}   system chars={r['system_chars']} "
              f"(~{r['system_chars'] // 4} tok)   blocks={[len(b['text']) for b in r['system']]}")
        print(f"   buckets: {json.dumps(r['buckets'])}")
        print(f"   message: {r['message'][:160]!r}")
        print(f"   reply:   {(it.get('move') or {}).get('response', '')[:200]!r}")
    print(f"\n   tutor tokens: in={tin} out={tout} cost=${cost(TUTOR, tin, tout):.4f}")
    return cost(TUTOR, tin, tout)


async def main() -> None:
    bank = local_bank.load()
    tutor = completer(TUTOR)
    rng = random.Random(7)

    study_learner = LearnLearner(Profile(persona="partial", style="clean", seed=rng.randrange(1 << 30)),
                                 kinds=["quote", "check_code", "summarise_notes"])
    study = await run_study(
        QID, bank=bank, tutor=tutor, learner=study_learner, max_turns=3, tags={"smoke": True}, seed=1,
        setup={"notes_preset": "short", "done_submodules": 1, "prior_gaps": 2,
               "memory": {"doubts": 3, "confusions": 1}, "ticks": "some", "check_outcomes": ["fail"]},
    )
    total = show(study)

    assess_learner = LearnLearner(Profile(persona="partial", style="sms", seed=rng.randrange(1 << 30)))
    assess = await run_assess(
        QID, bank=bank, tutor=tutor, learner=assess_learner, max_turns=3, tags={"smoke": True}, seed=2,
        setup={"notes_preset": "injection", "study_prefix": 2, "assess_start_stage": 6,
               "check_outcomes": ["compile_error", "submit_pass"]},
    )
    total += show(assess)
    print(f"\n   grades: {json.dumps(assess['grades'])[:400]}")

    lu = {k: study_learner.usage[k] + assess_learner.usage[k] for k in ("in", "out", "calls")}
    total += cost("gpt-5.4-mini", lu["in"], lu["out"])
    print(f"\nlearner: {lu}   total cost ~${total:.4f}")
    study_sys = [r["system_chars"] for r in study["records"]]
    study_tok = [r["interviewer"].get("tokens", {}).get("in") or 0 for r in study["records"]]
    print(f"study system chars max={max(study_sys, default=0)}   study call input tokens max={max(study_tok, default=0)}")
    print(f"notes after study:\n{study['notes_after'][-600:]}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"study": study, "assess": assess}, indent=1, default=str))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
