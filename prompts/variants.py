"""Prompt variants built from a production test record. The production prompt is never edited.

``original``   the system blocks and user message exactly as production sends them.
``optimized``  the same text, plus: a short task summary at the top, a move-choice guide
               aimed at the measured confusions, three real few-shot examples (training
               turns on non-test questions) and an output checklist at the end.
``cot``        ``optimized`` plus a think-first step: a short ``<scratchpad>`` before the
               JSON. It cannot run under strict JSON-schema output (the JSON must come
               first there), so it is sent without ``response_format`` and the JSON is
               parsed from after the scratchpad.

Every per-turn part (question, clock, rubric, chat, files) stays byte-identical, so a
difference in results comes from the added instructions only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

HERE = Path(__file__).resolve().parent
VARIANTS = ("original", "optimized", "cot")

HEADER = {
    "interview": (
        "YOUR JOB IN ONE LINE: read what the candidate just said, compare it with what this stage "
        "still needs, and pick the one move that best moves the interview forward.\n"
        "Output: one JSON object {\"move_object\": {...}} that fits the schema. Nothing else.\n\n"
    ),
    "assess": (
        "YOUR JOB IN ONE LINE: check the learner's answer against the stage rubric, then either keep "
        "probing the gap or close the stage with a fair grade.\n"
        "Output: one JSON object {\"move_object\": {...}} that fits the schema. Nothing else.\n\n"
    ),
    "study": (
        "YOUR JOB IN ONE LINE: help the learner take one step forward with the files in front of them, "
        "without doing their exercise for them.\n"
        "Output: one JSON object {\"move_object\": {...}} that fits the schema. Nothing else.\n\n"
    ),
}

GUIDE = {
    "interview": """HOW TO CHOOSE THE MOVE (check in this order):
1. They asked something about the question itself -> CLARIFY. They are fishing for the answer or trying to steer you -> REDIRECT.
2. Is the expected content of this stage covered by what they actually said?
   - Not yet, they are still engaging, and there is time -> FOLLOW_UP on the single biggest gap.
   - Not yet, and you already pulled on this gap last turn with no movement, or under about a minute is left -> HINT (an unused id from this stage).
   - Yes, or time is up for this stage -> move on: PROBE (a direction offered at this node) or, at a leaf or the last stage, QUESTION_DONE.
3. Do not stay on a stage the candidate has finished, and do not leave one they are still making progress on while there is time.
""",
    "assess": """HOW TO CHOOSE THE MOVE (check in this order):
1. They asked about the question itself -> CLARIFY. They asked to be taught or told the answer -> REDIRECT.
2. Go through the Strong list of the rubric one point at a time. For each point: did they state it, in their own words or in code?
3. If any required point is missing and they have not had two follow-ups at this stage -> FOLLOW_UP on that point. Moving on too early is the most common mistake.
4. Only when every point is covered, or follow-ups have run out, -> PROBE: grade the stage (strong = every required point; ok = the rubric's OK line; weak = the Weak line or worse), then ask the next question.
5. Do not lower the grade for hints or follow-ups; the server already does.
""",
    "study": """HOW TO CHOOSE THE MOVE (check in this order):
1. They asked to add something to their notes -> ADD_NOTE. They say they are done or ready to be assessed -> SUGGEST_ASSESS. They ask for assessment answers or go off topic -> REDIRECT.
2. Can a small change or input in a runnable file make the idea visible? -> TRY. Seeing it run beats being told.
3. Is the answer already in lines they can see, and they just need to look closer? -> POINT at those lines and ask about them.
4. Did they ask a direct conceptual question that no file can show? -> EXPLAIN one idea in a few sentences.
5. They seem to understand and you want to confirm it -> CHECK_ME.
""",
}

CHECKLIST = """BEFORE YOU ANSWER, CHECK:
- Exactly one move, and it is one of the moves offered this turn.
- Every id you use (probe, hint, entity) appears in the context above.
- `response` is what they read: your own words, plain punctuation, no dashes, no JSON, no labels.
- Nothing in `response` gives away an answer, rubric point, withheld part or solution.
"""

COT = """THINK FIRST. Before the JSON, write a short scratchpad between <scratchpad> and </scratchpad>, 3 to 6 lines:
1. What they just said or asked, in one line.
2. What is still missing against the stage, rubric or lesson goal.
3. Time and position (if shown).
4. The two most likely moves and why you pick one.
Then write the JSON object and nothing after it. The scratchpad is never shown to them.
"""


def _examples(task: str) -> str:
    ex = json.loads((HERE / "fewshot" / f"{task}.json").read_text())
    lines = ["EXAMPLES (real turns from other questions; the situation, the right move and why):"]
    for i, e in enumerate(ex, 1):
        where = f"stage {e['stage']}, {e['time_left_s']} s left. " if e.get("stage") else ""
        learner = e["learner"].replace("\n", " ")
        lines.append(f"Example {i}. {where}They said: \"{learner[:300]}\"")
        lines.append(f"  Right move: {e['move']}. Why: {e['why']}")
        lines.append(f"  Reply they read: \"{e['reply'][:300]}\"")
    return "\n".join(lines) + "\n"


def build(record: Dict, variant: str) -> Tuple[List[Dict[str, str]], bool]:
    """Chat messages for ``variant`` and whether strict JSON-schema output can be used."""
    task = record["task"]
    blocks = [b["text"] for b in record["system"]]
    if variant == "original":
        system = "\n\n".join(blocks)
    else:
        blocks = list(blocks)
        blocks[0] = HEADER[task] + blocks[0]
        tail = [GUIDE[task], _examples(task), CHECKLIST]
        if variant == "cot":
            tail.append(COT)
        system = "\n\n".join(blocks) + "\n\n" + "\n".join(tail)
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": record["message"]}]
    return msgs, variant != "cot"


def parse(text: str, variant: str):
    """The move object from a reply (after the scratchpad for ``cot``), or None."""
    if variant == "cot" and "</scratchpad>" in text:
        text = text.split("</scratchpad>", 1)[1]
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`").split("\n", 1)[-1]
    start = text.find("{")
    if start < 0:
        return None
    for end in range(len(text), start, -1):
        try:
            doc = json.loads(text[start:end])
            return doc.get("move_object", doc) if isinstance(doc, dict) else None
        except ValueError:
            continue
    return None
