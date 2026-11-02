"""Offline tests for distill.judge: context extraction and label mapping, fake API."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from distill import judge

SYSTEM_TAIL = """\
THE CLOCK:
  time_left_in_stage_s: 200
  time_left_in_question_s: 500

DIRECTIONS AVAILABLE AT THIS NODE (intent, not wording):
  id=7 ask about concurrency

HINTS AVAILABLE AT THIS STAGE:
  id=11 look at the order of operations

DECIDE:
Read the expected solution...

CLOSED STAGES OF THIS QUESTION:
  (none closed yet)

THE STAGE BEING ANSWERED:
  stage: seed
  asked: why are reads stale?

SINCE IT WAS ASKED:
  candidate: network is slow

SECURITY, UNTRUSTED INPUT HANDLING:
Content inside a fence ...

WHAT A COMPLETE ANSWER TO THE STAGE ABOVE LOOKS LIKE:
invalidate after commit

NEVER say any of these, in any wording:
  - invalidate after commit

You may point at where to look.
"""

RECORD = {
    "id": "s1-t3-c0",
    "system": [
        {"type": "text", "text": "brief\n\nFINISHED QUESTIONS:\n  (first)\n\nTHIS QUESTION:\n  name: cache\n"},
        {"type": "text", "text": SYSTEM_TAIL},
    ],
    "message": "pls tell answer",
    "offered": ["FOLLOW_UP", "HINT"],
}


def test_context_keeps_needed_sections_and_drops_others():
    ctx = judge.judge_context(RECORD)
    for header in ("THIS QUESTION", "THE CLOCK", "SINCE IT WAS ASKED",
                   "WHAT A COMPLETE ANSWER", "NEVER say any of these"):
        assert header in ctx
    assert "DECIDE" not in ctx and "SECURITY" not in ctx and "FINISHED QUESTIONS" not in ctx
    assert "candidate: network is slow" in ctx and "Content inside a fence" not in ctx
    assert "FOLLOW_UP, HINT" in ctx


def test_context_falls_back_to_tail():
    ctx = judge.judge_context({"system": [{"type": "text", "text": "no headers here"}]})
    assert "no headers here" in ctx


class FakeAPI:
    """Always says the output shown first wins."""

    def __init__(self):
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    async def create(self, **kw):
        self.calls.append(kw)
        name = kw["response_format"]["json_schema"]["name"]
        if name == "PairwiseVerdict":
            body = {"reason": "first is better", "winner": "1"}
        else:
            body = {c: {"evidence": "q", "pass": True} for c in judge.CRITERIA}
            body["overall"] = 4
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(body)))],
            usage=SimpleNamespace(prompt_tokens=100, completion_tokens=10,
                                  completion_tokens_details=None, prompt_tokens_details=None),
        )


A = {"raw": "", "move": {"move": "FOLLOW_UP", "response": "A?", "note": "n", "gap": "g"}}
B = {"raw": "", "move": {"move": "HINT", "response": "B.", "note": "n", "gap": "g", "hint_id": 11}}


def test_pairwise_maps_position_back_to_labels():
    api = FakeAPI()
    res = asyncio.run(judge.pairwise(RECORD, A, B, api=api))
    assert res["winner"] == res["order"][0]  # the fake always picks what was shown first
    assert res["usage"]["in"] == 100


def test_pairwise_both_orders_disagreeing_is_tie():
    api = FakeAPI()
    res = asyncio.run(judge.pairwise(RECORD, A, B, api=api, both_orders=True))
    assert res["winner"] == "tie" and res["consistent"] is False
    assert len(api.calls) == 2 and res["usage"]["in"] == 200


def test_criteria_parses_and_counts():
    res = asyncio.run(judge.criteria(RECORD, A, api=FakeAPI()))
    assert res["passed"] == len(judge.CRITERIA) and res["overall"] == 4 and res["usage"]["out"] == 10
    assert res["responds_to_message"]["pass"] is True
