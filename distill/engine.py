"""One interview, run through the production engine, every model call recorded.

The session service, driver, cursor, agent, schema builder and check are the ones
production runs. Only storage is in memory and the clock is ours, so a whole
interview's worth of time passes instantly and any clock value can be reached.

Each model call becomes one record: the exact system blocks, user message and
schema that were sent, what came back, and the ``AgentInput`` fields the scorer and
the coverage report need (clocks, options, secrets).
"""

from __future__ import annotations

import json
import pickle
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence

from app.services.interview import InterviewSessionService
from app.services.interview.models.dialogue import FORMAT_FREE_TEXT, FORMAT_WHITEBOARD, Answer
from app.services.interview.plan_builder import build_planned_segments
from app.services.interview.runtime import InterviewAgent, QuestionCursor, RuntimeDriver
from app.services.interview.runtime.shapes import offered_moves
from app.services.interview.session.serialization import state_from_dict


class Clock:
    """Time that only moves when we say so."""

    def __init__(self, t: float = 1_700_000_000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += max(0.0, seconds)


# ── in-memory storage, the repository contract the session service needs ─────


class _Repo:
    def __init__(self):
        self._rows, self._by_assignment, self._next = {}, {}, 1

    def insert_if_absent(self, session):
        key = (session.assignment_id, session.round)
        sid = self._by_assignment.get(key)
        if sid is not None:
            return self._rows[sid], False
        sid = self._next
        self._next += 1
        row = session.model_copy(update={"interview_session_id": sid, "version": 0})
        self._rows[sid] = row
        self._by_assignment[key] = sid
        return row, True

    def get_by_assignment(self, assignment_id, *, round="ai_interview"):
        sid = self._by_assignment.get((assignment_id, round))
        return self._rows.get(sid) if sid else None

    def get_by_id(self, sid):
        return self._rows.get(sid)

    def update_state(self, sid, *, state, status, llm_call_count, expected_version):
        row = self._rows[sid]
        if row.version != expected_version:
            return None
        stored = json.loads(json.dumps(state, default=str))
        row = row.model_copy(update={
            "state": stored, "status": status, "llm_call_count": llm_call_count,
            "version": row.version + 1,
        })
        self._rows[sid] = row
        return row


class _Rows:
    def __init__(self):
        self.rows = []

    def append_many(self, rows):
        self.rows.extend(rows)
        return list(rows)

    def list_for_session(self, interview_session_id):
        return [r for r in self.rows if r.interview_session_id == interview_session_id]


# ── recording ───────────────────────────────────────────────────────────


class _Sink:
    def __init__(self):
        self.current = None
        self.calls: List[Dict[str, Any]] = []


class _RecordingCompleter:
    """The interviewer's completer, with every call kept exactly as sent."""

    def __init__(self, inner, sink: _Sink):
        self._inner = inner
        self._sink = sink

    async def complete(self, *, system, message, schema, **kw):
        started = time.monotonic()
        error = None
        reply = None
        try:
            reply = await self._inner.complete(system=system, message=message, schema=schema, **kw)
        except Exception as exc:  # recorded, then re-raised: the turn fails as in production
            error = repr(exc)
            raise
        finally:
            self._sink.calls.append({
                "agent_input": self._sink.current,
                "system": [dict(b) for b in system],
                "message": message,
                "schema": schema,
                "reply": reply,
                "error": error,
                "wall_ms": int((time.monotonic() - started) * 1000),
            })
        return reply


class _RecordingAgent(InterviewAgent):
    def __init__(self, completer, sink: _Sink):
        super().__init__(_RecordingCompleter(completer, sink))
        self._sink = sink

    async def decide(self, agent_input):
        self._sink.current = agent_input
        return await super().decide(agent_input)


# ── what a record keeps ─────────────────────────────────────────────────


def _bucket_clock(seconds: int) -> str:
    if seconds <= 0:
        return "0"
    if seconds < 60:
        return "1-59"
    if seconds < 240:
        return "60-239"
    if seconds <= 600:
        return "240-600"
    return ">600"


def _bucket_count(n: int, edges=(0, 1, 2, 3, 7)) -> str:
    if n == 0:
        return "0"
    if n == 1:
        return "1"
    if n <= 2:
        return "2"
    if n <= 6:
        return "3-6"
    return "7+"


def facts(agent_input) -> Dict[str, Any]:
    """What the scorer and the coverage report read off the engine's own input."""
    clock = agent_input.current_state
    current = agent_input.current_question
    entities = list(getattr(current, "entities", ()) or ())
    answer_entities = list(agent_input.candidate_response.entities or ())
    return {
        "question_name": agent_input.name,
        "seniority": agent_input.seniority or "unspecified",
        "time_left_in_stage_s": clock.time_left_in_stage_s,
        "time_left_in_question_s": clock.time_left_in_question_s,
        "question_position": [clock.question_position.n, clock.question_position.of],
        "stage_position": [clock.stage_position.n, clock.stage_position.of],
        "level": current.level,
        "stage": current.stage,
        "probe_option_ids": [int(p.id) for p in agent_input.probe_options],
        "hint_ids": [int(h.id) for h in agent_input.available_hints],
        "finished_questions": len(agent_input.global_interview_context),
        "closed_stages": len(agent_input.stages_so_far),
        "history_len": len(agent_input.stage_history),
        "stage_entity_types": sorted({str(e.get("type")) for e in entities}),
        "answer_entity_types": sorted({str(e.get("type")) for e in answer_entities}),
        "runnable": bool(agent_input.runnable_files),
        "code_runs": len(agent_input.code_runs),
        "has_expected": bool(agent_input.expected_current_solution),
        "primary_topic": agent_input.primary_topic,
        "secrets": {
            "withheld": list(agent_input.withheld_solution_parts),
            "expected": agent_input.expected_current_solution or "",
        },
    }


def buckets(f: Dict[str, Any], offered: Sequence[str]) -> Dict[str, Any]:
    return {
        "stage_clock": _bucket_clock(f["time_left_in_stage_s"]),
        "question_clock": "0" if f["time_left_in_question_s"] <= 0
        else ("<=120" if f["time_left_in_question_s"] <= 120 else "plenty"),
        "level": str(f["level"]),
        "probe_options": _bucket_count(len(f["probe_option_ids"])),
        "hints_left": _bucket_count(len(f["hint_ids"])),
        "offered_set": ",".join(sorted(offered)),
        "finished_questions": _bucket_count(f["finished_questions"]),
        "closed_stages": _bucket_count(f["closed_stages"]),
        "history_len": _bucket_count(f["history_len"]),
        "stage_files": ",".join(f["stage_entity_types"]) or "none",
        "answer_files": ",".join(f["answer_entity_types"]) or "none",
        "runnable": "yes" if f["runnable"] else "no",
        "code_runs": _bucket_count(f["code_runs"]),
        "withheld": _bucket_count(len(f["secrets"]["withheld"])),
        "expected": "authored" if f["has_expected"] else "none",
        "seniority": f["seniority"],
    }


def _reply_fields(reply) -> Dict[str, Any]:
    if reply is None:
        return {}
    return {
        "model": reply.model,
        "effort": getattr(reply, "effort", None),
        "stop_reason": reply.stop_reason,
        "raw": reply.raw_text,
        "move": reply.move,
        "latency_ms": reply.latency_ms,
        "tokens": {"in": reply.input_tokens, "out": reply.output_tokens,
                   "cache_read": reply.cache_read_input_tokens},
    }


# ── the candidate side ──────────────────────────────────────────────────


@dataclass
class CandidateTurn:
    """What the simulated candidate does next: an answer, and how long it took."""

    answer: Answer
    think_seconds: float
    meta: Dict[str, Any] = field(default_factory=dict)


#: (question shown, transcript so far, session info) -> the candidate's turn.
Candidate = Callable[[Any, List[Dict[str, str]], Dict[str, Any]], Awaitable[CandidateTurn]]


async def run_session(
    question_ids: Sequence[int],
    *,
    bank,
    interviewer,
    candidate: Candidate,
    total_minutes: int = 30,
    max_turns: int = 40,
    session_tags: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Play one interview to the end; return its records and transcript."""
    sink = _Sink()
    clock = Clock()
    service = InterviewSessionService(
        _Repo(),
        RuntimeDriver(QuestionCursor(bank, allow_unpublished=True),
                      _RecordingAgent(interviewer, sink), now=clock),
        turns=_Rows(), traces=_Rows(), auto_run=False,
    )
    segments = build_planned_segments(
        total_minutes=total_minutes, question_ids=list(question_ids),
        available_formats=(FORMAT_FREE_TEXT, FORMAT_WHITEBOARD),
    )
    view = await service.start(1, segments=segments)
    row = service._repo.get_by_assignment(1)
    state = state_from_dict(row.state)
    started = clock()
    state.cursor.interview_ends_at = started + sum(s.time_budget_s for s in state.segments)
    state.cursor.segment_ends_at = started + state.segments[state.cursor.segment_index].time_budget_s
    service._persist(row, state)

    session_id = uuid.uuid4().hex[:12]
    transcript: List[Dict[str, Any]] = []
    records: List[Dict[str, Any]] = []
    info = dict(session_tags or {})
    turns = 0
    while view.question is not None and turns < max_turns:
        shown = view.question
        transcript.append({"who": "ai", "text": shown.text})
        info.update({
            "time_left_in_stage_s": view.time_left_in_stage_s,
            "time_left_in_question_s": view.time_left_in_question_s,
            "format": shown.format,
        })
        turn = await candidate(shown, transcript, info)
        clock.advance(turn.think_seconds)
        transcript.append({"who": "candidate", "text": turn.answer.text, **turn.meta})
        live = state_from_dict(service._repo.get_by_assignment(1).state).runtime
        live_qid = int(live.question_id) if live is not None and live.question_id is not None else None
        before = len(sink.calls)
        view = await service.submit(
            1, question_id=view.question_id, answer=turn.answer, segments=segments,
        )
        for n, call in enumerate(sink.calls[before:]):
            ai = call["agent_input"]
            f = facts(ai) if ai is not None else {}
            offered = offered_moves(call["schema"])
            records.append({
                "id": f"{session_id}-t{turns}-c{n}",
                "session_id": session_id,
                "turn": turns,
                "attempt": n,
                "question_id": live_qid,
                "tags": dict(session_tags or {}),
                "candidate": turn.meta,
                "think_seconds": turn.think_seconds,
                "system": call["system"],
                "message": call["message"],
                "schema": call["schema"],
                "offered": offered,
                "facts": f,
                "buckets": buckets(f, offered) if f else {},
                "interviewer": _reply_fields(call["reply"]),
                "error": call["error"],
                "wall_ms": call["wall_ms"],
                # The engine's own input, kept so an edit (another clock, a messier
                # message) is rebuilt by the production prompt builder, not by hand.
                "_agent_input": pickle.dumps(ai) if ai is not None else None,
            })
        turns += 1
    return {
        "session_id": session_id,
        "question_ids": list(question_ids),
        "tags": dict(session_tags or {}),
        "turns": turns,
        "completed": view.question is None,
        "transcript": transcript,
        "records": records,
    }


async def replay(agent_input, completer):
    """One call on an ``AgentInput`` through the production agent, recorded the same
    way a session turn is. Used for clock edits and clean/messy pairs."""
    sink = _Sink()
    agent = _RecordingAgent(completer, sink)
    await agent.decide(agent_input)
    call = sink.calls[-1]
    f = facts(agent_input)
    offered = offered_moves(call["schema"])
    return {
        "system": call["system"], "message": call["message"], "schema": call["schema"],
        "offered": offered, "facts": f, "buckets": buckets(f, offered),
        "interviewer": _reply_fields(call["reply"]), "error": call["error"],
        "wall_ms": call["wall_ms"], "_agent_input": pickle.dumps(agent_input),
    }
