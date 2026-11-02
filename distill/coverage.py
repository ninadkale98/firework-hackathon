"""Coverage of a split: how many records fall in each bucket of each field, per
task, and which pairs of main fields never meet.

    python -m distill.coverage --split test          # prints, and writes data/coverage_test.html
"""

from __future__ import annotations

import argparse
import html
import itertools
from collections import Counter, defaultdict
from typing import Dict, List

from distill import store

#: Main fields: every pair of their values should appear at least once.
MAIN = ["stage_clock", "offered_set", "persona", "style", "level", "history_len"]
MIN_MAIN, MIN_MINOR = 30, 10


def counts(split: str):
    conn = store.connect()
    by_task: Dict[str, Dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    pairs: Dict[str, Counter] = defaultdict(Counter)
    totals = Counter()
    sources = defaultdict(Counter)
    questions = defaultdict(Counter)
    for rec in store.records(conn, split=split):
        task = rec["task"]
        totals[task] += 1
        sources[task][rec["source"]] += 1
        questions[task][rec.get("question_id")] += 1
        b = rec.get("buckets") or {}
        for k, v in b.items():
            by_task[task][k][str(v)] += 1
        for a, c in itertools.combinations([m for m in MAIN if m in b], 2):
            pairs[task][(a, str(b[a]), c, str(b[c]))] += 1
    return totals, by_task, pairs, sources, questions


def report(split: str) -> str:
    totals, by_task, pairs, sources, questions = counts(split)
    lines: List[str] = []
    page: List[str] = [f"<h1>Coverage: {split}</h1>"]
    for task in sorted(totals):
        lines.append(f"\n=== {task}: {totals[task]} records  sources={dict(sources[task])}  "
                     f"questions={len(questions[task])}")
        page.append(f"<h2>{task} — {totals[task]} records</h2><p>sources {dict(sources[task])} · "
                    f"{len(questions[task])} questions</p>")
        for field in sorted(by_task[task]):
            need = MIN_MAIN if field in MAIN else MIN_MINOR
            c = by_task[task][field]
            low = [k for k, v in c.items() if v < need]
            lines.append(f"  {field:<20} " + "  ".join(f"{k}={v}" for k, v in c.most_common())
                         + (f"   << under {need}: {low}" if low else ""))
            cells = "".join(
                f"<td style='background:{'#fde2e2' if v < need else '#e3f6e8'}'>{html.escape(k)}<br><b>{v}</b></td>"
                for k, v in c.most_common())
            page.append(f"<table><tr><th>{field}</th>{cells}</tr></table>")
        # Pairs of main-field values that never meet.
        values = {f: list(by_task[task][f]) for f in MAIN if f in by_task[task]}
        missing = []
        for a, c in itertools.combinations(values, 2):
            for va in values[a]:
                for vc in values[c]:
                    if pairs[task][(a, va, c, vc)] == 0:
                        missing.append(f"{a}={va} × {c}={vc}")
        possible = sum(len(values[a]) * len(values[c]) for a, c in itertools.combinations(values, 2))
        lines.append(f"  pairs covered: {possible - len(missing)}/{possible}")
        page.append(f"<p>Pairs of main-field values covered: <b>{possible - len(missing)}/{possible}</b></p>")
    out = store.ROOT / "data" / f"coverage_{split}.html"
    out.write_text("<meta charset=utf-8><style>body{font:14px system-ui;margin:24px}table{border-collapse:collapse;margin:4px 0}"
                   "td,th{border:1px solid #ddd;padding:4px 8px;text-align:center}th{text-align:left;min-width:160px}</style>"
                   + "\n".join(page))
    lines.append(f"\nwrote {out}")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    print(report(ap.parse_args().split))
