"""The public leaderboard page: ``index.html`` at the repo root (GitHub Pages).

    python -m distill.site            # reads data/distill.db, writes index.html

Only aggregates leave the database: no question text, rubric, prompt or reply is
written to the page, because the repository is public and the questions carry
interview answer keys.

The headline comparison is on the stratified sample every model answered and the
judge scored (``runs.stratified``), so no model is compared on easier turns.
"""

from __future__ import annotations

import html
import json
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from distill import judge as J
from distill import store
from distill.models import MODELS, cost
from distill.runs import REFERENCE, stratified
from distill.score import bootstrap_ci

SAMPLE = 300

#: Display name, group. Groups are the only colours on the charts (three, the
#: palette's all-pairs-safe slots); every point is labelled by name as well.
SHOW = {
    REFERENCE: ("Claude Opus 5 (reference)", "frontier"),
    "deepseek-v4.1-flash": ("DeepSeek V4.1 Flash (teacher)", "open"),
    "qwen3.5-9b": ("Qwen3.5-9B base (student)", "student"),
    "qwen3.5-9b-tuned": ("Qwen3.5-9B tuned (ours)", "student"),
    "qwen3.5-9b-dpo": ("Qwen3.5-9B tuned + DPO (ours)", "student"),
}
GROUPS = {"frontier": "Frontier reference", "open": "Open teacher (writes the training data)", "student": "Our student, 9B"}
TASKS = ["interview", "assess", "study"]


def _rate(vals: List[float]) -> Optional[Dict[str, Any]]:
    if not vals:
        return None
    lo, hi = bootstrap_ci(vals, n=1000, seed=0)
    return {"v": sum(vals) / len(vals), "lo": lo, "hi": hi, "n": len(vals)}


def collect() -> Dict[str, Any]:
    conn = store.connect()
    recs = {r["id"]: r for r in store.records(conn, split="test")}
    sample_ids = {r["id"] for r in stratified(list(recs.values()), SAMPLE)}
    scores: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(lambda: defaultdict(dict))
    for rid, model, kind, doc in conn.execute("select record_id, model, kind, doc from scores"):
        if rid in recs:
            scores[model][rid][kind] = json.loads(doc)
    for by_rid in scores.values():
        for rid in list(by_rid):
            by_rid[rid] = J.judge_kinds(by_rid[rid])
    latency: Dict[str, Dict[str, float]] = defaultdict(dict)
    for rid, model, doc in conn.execute("select record_id, model, doc from runs"):
        d = json.loads(doc)
        if d.get("latency_ms"):
            latency[model][rid] = d["latency_ms"]
    for rid, rec in recs.items():
        if rec["interviewer"].get("latency_ms"):
            latency[REFERENCE][rid] = rec["interviewer"]["latency_ms"]

    def summarise(model: str, ids: List[str]) -> Optional[Dict[str, Any]]:
        rows = [dict(scores[model][i], _task=recs[i]["task"]) for i in ids if "rules" in scores[model].get(i, {})]
        if not rows:
            return None
        rules = [r["rules"] for r in rows]
        judged = [r for r in rows if "judge" in r and r["judge"].get("overall") is not None]
        pw = [r["pairwise"]["winner"] for r in rows if "pairwise" in r]
        # Rules, judged by the AI judge rather than word overlap: no spoon-feeding on
        # every judged turn, and on manipulation / answer-fishing turns the safety
        # verdict (stayed in role, revealed nothing, redirected) as well.
        def leak_ok(r):
            # Each mode's own "did not give the answer away" verdict: interview
            # spoon-feeding, Assess teaching or revealing, Study handing over an
            # exercise or reference solution (explaining is a tutor's job).
            j = r["judge"]
            if "leak_ok" in j:
                return bool(j["leak_ok"])
            return bool(j.get("no_spoon_feeding", {}).get("pass"))

        def kept(r):
            ok = leak_ok(r)
            if "safety" in r and r["safety"].get("ok") is not None:
                ok = ok and bool(r["safety"]["ok"])
            return ok
        rules_ai = [float(kept(r)) for r in judged]
        leak_ai = [float(not leak_ok(r)) for r in judged]
        safety_ok = [float(bool(r["safety"]["ok"])) for r in rows if "safety" in r and r["safety"].get("ok") is not None]
        # Explaining a stage's answer while moving on from it: not a leak (the stage
        # is graded), but measured, because a weak model skips ahead and explains.
        closing = [r for r in judged if (r["rules"].get("move") in ("PROBE", "QUESTION_DONE"))
                   and r.get("_task") in ("interview", "assess")]
        reveal_closing = [float(bool(any(isinstance(v, dict) and v.get("closed_stage") for v in r["judge"].values())))
                          for r in closing]
        quality = [(float(bool(r["rules"].get("format_ok"))) + float(kept(r)) + r["judge"]["overall"] / 5.0) / 3.0
                   for r in judged]
        lat = sorted(latency[model][i] for i in ids if i in latency[model])
        tin = [s.get("tokens_in") or 0 for s in rules]
        tout = [s.get("tokens_out") or 0 for s in rules]
        spec = MODELS.get(model) or (MODELS["opus-5"] if model == REFERENCE else None)
        usd = cost(spec.name, int(statistics.mean(tin)), int(statistics.mean(tout))) \
            if spec and spec.price and tin else None
        return {
            "n": len(rows),
            "quality": _rate(quality),
            "format": _rate([float(bool(s.get("format_ok"))) for s in rules]),
            "rules": _rate(rules_ai),
            "leak": _rate(leak_ai),
            "safety": _rate(safety_ok),
            "reveal_closing": _rate(reveal_closing),
            "same_move": _rate([float(s["same_move"]) for s in rules if s.get("same_move") is not None]),
            "same_grade": _rate([float(s["same_level"]) for s in rules if s.get("same_level") is not None]),
            "judge": (_rate([r["judge"]["overall"] for r in judged]) if judged else None),
            "win_tie": _rate([1.0 if w in ("a", "tie") else 0.0 for w in pw]) if pw else None,
            "p50_ms": lat[len(lat) // 2] if lat else None,
            "usd_turn": usd,
        }

    by_task_sample = {t: [i for i in sample_ids if recs[i]["task"] == t] for t in TASKS}
    out: Dict[str, Any] = {"sample": {}, "overall": {}, "full": {}, "robust": {}}
    models = [m for m in SHOW if m in scores]
    for m in models:
        out["overall"][m] = summarise(m, list(sample_ids))
        out["sample"][m] = {t: summarise(m, ids) for t, ids in by_task_sample.items()}
        full = [i for i in recs if i in scores[m]]
        if len(full) > SAMPLE * 2:
            out["full"][m] = {t: summarise(m, [i for i in full if recs[i]["task"] == t]) for t in TASKS}
    # Robustness: same move on a clean answer and its messy rewrite.
    pairs = [(r["pair_of"], r["id"]) for r in recs.values() if r.get("pair_of")]
    for m in models:
        same = []
        for a, b in pairs:
            sa = scores[m].get(a, {}).get("rules")
            sb = scores[m].get(b, {}).get("rules")
            if sa and sb and sa.get("move") and sb.get("move"):
                same.append(float(sa["move"] == sb["move"]))
        out["robust"][m] = _rate(same)
    # The teacher was chosen earlier among five open models (Kimi K3, Qwen 3.8 Max,
    # DeepSeek V4.1 Flash, gpt-oss-120b, GLM 5.3) by the rule fixed before results.
    out["teacher"] = {t: {"model": "deepseek-v4.1-flash"} for t in TASKS}

    counts = defaultdict(int)
    for r in recs.values():
        counts[r["task"]] += 1
    out["meta"] = {
        "built": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "test_turns": dict(counts), "sample": len(sample_ids),
        "sample_by_task": {t: len(v) for t, v in by_task_sample.items()},
        "questions": len({r.get("question_id") for r in recs.values()}),
    }
    out["show"] = {m: {"name": SHOW[m][0], "group": SHOW[m][1]} for m in models}
    out["groups"] = GROUPS
    return out


def render(data: Dict[str, Any]) -> str:
    payload = json.dumps(data, default=str).replace("</", "<\\/")
    return PAGE.replace("__DATA__", payload)


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Interviewer Model Leaderboard</title>
<meta name="description" content="An open 9B model fine-tuned to run AI technical interviews, scored against Claude Opus on a frozen test set.">
<style>
:root {
  color-scheme: light;
  --surface: #fcfcfb; --surface-2: #f3f2ef; --border: #e2e0da;
  --ink: #0b0b0b; --ink-2: #52514e; --ink-3: #7a7974;
  --g-frontier: #2a78d6; --g-open: #eb6834; --g-student: #1baf7a;
  --grid: #e9e7e1; --accent: #2a78d6;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) {
    color-scheme: dark;
    --surface: #1a1a19; --surface-2: #232321; --border: #34332f;
    --ink: #f0efec; --ink-2: #c3c2b7; --ink-3: #8f8e86;
    --g-frontier: #3987e5; --g-open: #d95926; --g-student: #199e70;
    --grid: #2c2b28; --accent: #3987e5;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --surface: #1a1a19; --surface-2: #232321; --border: #34332f;
  --ink: #f0efec; --ink-2: #c3c2b7; --ink-3: #8f8e86;
  --g-frontier: #3987e5; --g-open: #d95926; --g-student: #199e70;
  --grid: #2c2b28; --accent: #3987e5;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--surface); color: var(--ink);
  font: 15px/1.5 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1080px; margin: 0 auto; padding: 32px 16px 64px; }
h1 { font-size: 28px; line-height: 1.2; margin: 0 0 8px; letter-spacing: -0.01em; }
h2 { font-size: 19px; margin: 40px 0 6px; }
p.lede { color: var(--ink-2); margin: 0 0 20px; max-width: 760px; }
.muted { color: var(--ink-3); font-size: 13px; }
a { color: var(--accent); }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; margin: 20px 0 8px; }
.tile { background: var(--surface-2); border: 1px solid var(--border); border-radius: 10px; padding: 14px 16px; }
.tile .k { color: var(--ink-2); font-size: 13px; }
.tile .v { font-size: 26px; font-weight: 650; font-variant-numeric: tabular-nums; }
.tile .s { color: var(--ink-3); font-size: 12px; }
.tabs { display: flex; gap: 6px; flex-wrap: wrap; margin: 12px 0; }
.tabs button { font: inherit; font-size: 13px; padding: 6px 12px; border-radius: 999px; cursor: pointer;
  border: 1px solid var(--border); background: transparent; color: var(--ink-2); }
.tabs button[aria-selected="true"] { background: var(--ink); color: var(--surface); border-color: var(--ink); }
.scroll { overflow-x: auto; border: 1px solid var(--border); border-radius: 10px; }
table { border-collapse: collapse; width: 100%; font-size: 13.5px; font-variant-numeric: tabular-nums; }
th, td { padding: 9px 12px; text-align: right; white-space: nowrap; border-bottom: 1px solid var(--border); }
th:first-child, td:first-child { text-align: left; }
th { color: var(--ink-2); font-weight: 600; background: var(--surface-2); position: sticky; top: 0; }
tr:last-child td { border-bottom: 0; }
td .ci { color: var(--ink-3); font-size: 11.5px; margin-left: 4px; }
.dot { display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 8px; vertical-align: 1px; }
.charts { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px; }
.chart { border: 1px solid var(--border); border-radius: 10px; padding: 12px 12px 4px; position: relative; }
.chart h3 { font-size: 14px; margin: 2px 4px 6px; }
.legend { display: flex; gap: 16px; flex-wrap: wrap; font-size: 13px; color: var(--ink-2); margin: 8px 0 0; }
svg text { fill: var(--ink-2); font-size: 11px; }
svg .lab { fill: var(--ink); font-size: 11.5px; }
.tip { position: absolute; pointer-events: none; background: var(--ink); color: var(--surface); font-size: 12px;
  padding: 6px 8px; border-radius: 6px; opacity: 0; transition: opacity .1s; white-space: nowrap; }
ul.notes { color: var(--ink-2); padding-left: 18px; }
ul.notes li { margin: 4px 0; }
.theme { position: fixed; top: 12px; right: 12px; font: inherit; font-size: 12px; padding: 5px 10px; border-radius: 999px;
  border: 1px solid var(--border); background: var(--surface-2); color: var(--ink-2); cursor: pointer; }
</style>
</head>
<body>
<button class="theme" id="theme" type="button" aria-label="Switch colour theme">Theme</button>
<main>
  <h1>Can a 9B open model run a technical interview like Claude Opus?</h1>
  <p class="lede">Every turn of a CodePraxis AI interview, and of its Learn tutor (Study) and Learn assessment (Assess), is one model call. The call reads the question, the clocks, the files, the conversation and a rubric it must never reveal, and returns one JSON move. We froze a test set of turns where <b>Claude Opus 5</b> was the interviewer, then had every other model answer the exact same prompts.</p>
  <div class="tiles" id="tiles"></div>
  <p class="muted" id="meta"></p>
  <p><a href="presentation.html">How the three agents work, the baseline and the evaluation &rarr;</a></p>

  <div class="tabs" role="tablist" id="tabs"></div>
  <h2>Quality against cost and speed</h2>
  <p class="muted">Quality is the per-turn mean of three things: valid output, rules kept (the AI judge finds no spoon-feeding, and on turns where the candidate fished for the answer or tried to manipulate the interviewer, it stayed in role, revealed nothing and redirected), and the judge's 1–5 score scaled to 1. It is measured on the shared sample, with 95% bootstrap intervals in the table. The x axes are log scale.</p>
  <div class="charts">
    <div class="chart" id="c-cost"><h3>Quality vs cost per turn (US$)</h3></div>
    <div class="chart" id="c-lat"><h3>Quality vs median latency per turn (s)</h3></div>
  </div>
  <h2 id="gap-h">Where the 9B falls short</h2>
  <p class="muted" id="gap-note"></p>
  <div class="chart" id="c-gap"></div>
  <div class="legend" id="legend"></div>
  <p class="muted">Qwen3.5-9B runs on its own rented GPU, billed per hour rather than per token, so it has no point on the cost chart until the cost is measured under load.</p>

  <h2>Leaderboard</h2>
  <p class="muted">The shared sample is stratified by task and candidate persona, and every model answered all of it. "Same move" is agreement with Opus's move. "Win/tie" is the judge's head-to-head verdict against Opus's reply, in randomised order. "Gives answer away" is each mode's own rule, judged by AI: the interviewer must not spoon-feed; the examiner must not teach or reveal a rubric point; the tutor may explain but must not hand over an exercise or reference solution or assessment answers. "Same grade" compares the strong / ok / weak grade on the Assess turns where both Opus and the model closed a stage. "Resists" is the safety judge on turns where the candidate fished for the answer or tried to manipulate the interviewer. The judge is GPT-5.4 and strict: it marks Opus too, so read these columns relative to Opus.</p>
  <div class="scroll"><table id="board"></table></div>

  <h2>Robustness to messy English</h2>
  <p class="muted">The same candidate answer was given clean and rewritten messily (typos, SMS spelling, dictation fillers). Each column is the share of pairs where the model chose the same move both times.</p>
  <div class="scroll"><table id="robust"></table></div>

  <h2>How the test set was made</h2>
  <ul class="notes">
    <li>Sessions run through the <b>production engine</b>: session service, driver, prompt and schema builders, checks. Only storage is in memory and the clock is fake. Every prompt is byte-identical to production.</li>
    <li>The candidates are simulated by GPT: 10 personas (strong, partial, confidently wrong, "I don't know", fishing for the answer, off-topic, prompt injection, terse, code-pasting, slow) and 8 writing styles, Hinglish included. Drawings go through the production Excalidraw normaliser.</li>
    <li>Coverage is measured per prompt field: clocks, depth, hints, probe options, the set of moves offered, history length, files, Learn notes, gaps and memory. Clock edge cases and messy rewrites were replayed through the production prompt builder.</li>
    <li>The test questions are held out from training, two per interview track (embedded, frontend, backend, AI engineering), plus three Learn sub-modules.</li>
    <li>Neither Claude's nor GPT's outputs are used as training targets. The fine-tuning data comes from DeepSeek V4.1 Flash, chosen among five open models (Kimi K3, Qwen 3.8 Max, DeepSeek V4.1 Flash, gpt-oss-120b, GLM 5.3) by a rule fixed before the results: licence allows training on outputs, no worse than Opus at giving answers away, best quality, cheaper on a tie.</li>
  </ul>
  <p class="muted">Code: <a href="https://github.com/ninadkale98/firework-hackathon">github.com/ninadkale98/firework-hackathon</a>. The questions, prompts and generated data are not published, because they contain interview answer keys.</p>
</main>
<script>
const D = __DATA__;
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const gcol = g => `var(--g-${g})`;
const pct = x => x == null ? "–" : (x * 100).toFixed(1) + "%";
const ci = c => c ? `<span class="ci">${(c.lo*100).toFixed(0)}–${(c.hi*100).toFixed(0)}</span>` : "";
const models = Object.keys(D.show);

// theme toggle: light / dark / system
(() => {
  const b = document.getElementById("theme"); const k = "lb-theme";
  const apply = v => { if (v) document.documentElement.dataset.theme = v; else delete document.documentElement.dataset.theme; b.textContent = v ? (v === "dark" ? "Dark" : "Light") : "Theme: system"; };
  let v = null; try { v = localStorage.getItem(k); } catch (e) {}
  apply(v);
  b.onclick = () => { v = v === null ? "dark" : v === "dark" ? "light" : null; try { v ? localStorage.setItem(k, v) : localStorage.removeItem(k); } catch (e) {} apply(v); draw(); };
})();

// tiles
(() => {
  const o = D.overall, ref = o.reference, base = o["qwen3.5-9b"], tuned = o["qwen3.5-9b-tuned"];
  const t = [];
  if (ref && ref.quality) t.push(["Claude Opus 5 quality", pct(ref.quality.v), `$${(ref.usd_turn||0).toFixed(3)} / turn`]);
  if (base && base.same_move) t.push(["Qwen3.5-9B base: same move as Opus", pct(base.same_move.v), base.win_tie ? `judge prefers it or ties on ${pct(base.win_tie.v)} of turns` : ""]);
  if (tuned && tuned.same_move) t.push(["Qwen3.5-9B tuned: same move as Opus", pct(tuned.same_move.v), tuned.win_tie ? `judge prefers it or ties on ${pct(tuned.win_tie.v)} of turns` : ""]);
  const tt = D.meta.test_turns; const total = Object.values(tt).reduce((a,b)=>a+b,0);
  t.push(["Frozen test set", total.toLocaleString() + " turns", `interview ${tt.interview||0} · assess ${tt.assess||0} · study ${tt.study||0}`]);
  document.getElementById("tiles").innerHTML = t.map(([k,v,s]) => `<div class="tile"><div class="k">${k}</div><div class="v">${v}</div><div class="s">${s}</div></div>`).join("");
  document.getElementById("meta").textContent = `Shared sample: ${D.meta.sample} turns (interview ${D.meta.sample_by_task.interview}, assess ${D.meta.sample_by_task.assess}, study ${D.meta.sample_by_task.study}) · ${D.meta.questions} held-out questions · built ${D.meta.built}`;
})();

// table
const COLS = [
  ["quality", "Quality", r => pct(r.quality && r.quality.v) + ci(r.quality)],
  ["format", "Valid output", r => pct(r.format && r.format.v)],
  ["rules", "Rules kept", r => pct(r.rules && r.rules.v)],
  ["leak", "Gives answer away", r => r.leak ? pct(r.leak.v) : "–"],
  ["reveal_closing", "Reveals answer when moving on", r => r.reveal_closing ? pct(r.reveal_closing.v) + ` <span class="ci">n=${r.reveal_closing.n}</span>` : "–"],
  ["safety", "Resists fishing / injection", r => r.safety ? pct(r.safety.v) + ` <span class="ci">n=${r.safety.n}</span>` : "–"],
  ["same_move", "Same move as Opus", r => r.same_move ? pct(r.same_move.v) + ci(r.same_move) : "–"],
  ["same_grade", "Same grade as Opus", r => r.same_grade ? pct(r.same_grade.v) + ` <span class="ci">n=${r.same_grade.n}</span>` : "–"],
  ["judge", "Judge 1–5", r => r.judge ? r.judge.v.toFixed(2) : "–"],
  ["win_tie", "Win/tie vs Opus", r => r.win_tie ? pct(r.win_tie.v) : "–"],
  ["p50_ms", "Median latency", r => r.p50_ms ? (r.p50_ms/1000).toFixed(1) + " s" : "–"],
  ["usd_turn", "$ / turn", r => r.usd_turn != null ? "$" + r.usd_turn.toFixed(4) : (r.n ? "GPU-hour" : "–")],
  ["n", "Turns", r => r.n],
];
const TABS = [["overall", "All tasks"], ["interview", "Interview"], ["assess", "Assess"], ["study", "Study"]];
let tab = "overall";
function rowsFor(t) {
  return models.map(m => [m, t === "overall" ? D.overall[m] : (D.sample[m] || {})[t]]).filter(([,r]) => r)
    .sort((a, b) => ((b[1].quality||{}).v||0) - ((a[1].quality||{}).v||0));
}
function table() {
  const head = `<tr><th>Model</th>${COLS.map(c => `<th>${c[1]}</th>`).join("")}</tr>`;
  const body = rowsFor(tab).map(([m, r]) => `<tr><td><span class="dot" style="background:${gcol(D.show[m].group)}"></span>${D.show[m].name}</td>${COLS.map(c => `<td>${c[2](r)}</td>`).join("")}</tr>`).join("");
  document.getElementById("board").innerHTML = head + body;
}
const tabsEl = document.getElementById("tabs");
tabsEl.innerHTML = TABS.map(([k, l]) => `<button role="tab" aria-selected="${k===tab}" data-k="${k}">${l}</button>`).join("");
tabsEl.onclick = e => { const k = e.target.dataset.k; if (!k) return; tab = k; [...tabsEl.children].forEach(b => b.setAttribute("aria-selected", b.dataset.k === k)); table(); draw(); };
const cur = m => tab === "overall" ? D.overall[m] : (D.sample[m] || {})[tab];
table();

// robustness
(() => {
  const rows = models.filter(m => D.robust[m]).map(m => `<tr><td><span class="dot" style="background:${gcol(D.show[m].group)}"></span>${D.show[m].name}</td><td>${pct(D.robust[m].v)}${ci(D.robust[m])}</td><td>${D.robust[m].n}</td></tr>`).join("");
  document.getElementById("robust").innerHTML = `<tr><th>Model</th><th>Same move, clean vs messy</th><th>Pairs</th></tr>` + rows;
})();

// legend
document.getElementById("legend").innerHTML = Object.entries(D.groups).map(([g, l]) => `<span><span class="dot" style="background:${gcol(g)}"></span>${l}</span>`).join("");

// scatter charts (log x), labelled points, hover tooltip
function scatter(id, xKey, xLabel, xFmt) {
  const host = document.getElementById(id);
  host.querySelectorAll("svg,.tip").forEach(n => n.remove());
  const pts = models.map(m => ({ m, r: cur(m) })).filter(p => p.r && p.r.quality && p.r[xKey] != null && p.r[xKey] > 0)
    .map(p => ({ ...p, x: xKey === "p50_ms" ? p.r.p50_ms / 1000 : p.r[xKey], y: p.r.quality.v }));
  const W = Math.max(320, host.clientWidth - 24), H = 300, L = 44, R = 16, T = 12, B = 40;
  if (!pts.length) { host.insertAdjacentHTML("beforeend", `<p class="muted" style="margin:8px 4px 12px">No data yet.</p>`); return; }
  const xs = pts.map(p => Math.log10(p.x)), x0 = Math.floor(Math.min(...xs) * 2) / 2, x1 = Math.ceil(Math.max(...xs) * 2) / 2 || x0 + 1;
  const ys = pts.map(p => p.y), y0 = Math.max(0, Math.floor((Math.min(...ys) - 0.05) * 10) / 10), y1 = Math.min(1, Math.ceil((Math.max(...ys) + 0.02) * 10) / 10);
  const sx = v => L + (Math.log10(v) - x0) / (x1 - x0 || 1) * (W - L - R);
  const sy = v => T + (1 - (v - y0) / (y1 - y0 || 1)) * (H - T - B);
  let g = "";
  for (let e = Math.floor(x0); e <= Math.ceil(x1); e++) for (const k of [1, 2, 5]) {
    const v = k * Math.pow(10, e), lv = Math.log10(v); if (lv < x0 - 1e-9 || lv > x1 + 1e-9) continue;
    g += `<line x1="${sx(v)}" x2="${sx(v)}" y1="${T}" y2="${H-B}" stroke="var(--grid)"/><text x="${sx(v)}" y="${H-B+16}" text-anchor="middle">${xFmt(v)}</text>`; }
  for (let v = y0; v <= y1 + 1e-9; v += 0.1) { g += `<line x1="${L}" x2="${W-R}" y1="${sy(v)}" y2="${sy(v)}" stroke="var(--grid)"/><text x="${L-6}" y="${sy(v)+4}" text-anchor="end">${Math.round(v*100)}%</text>`; }
  g += `<text x="${(L + W - R)/2}" y="${H-6}" text-anchor="middle">${xLabel}</text>`;
  // Labels: above the point, nudged down past any earlier label they would overlap.
  const placed = [];
  const marks = pts.map((p, i) => ({ p, i, cx: sx(p.x), cy: sy(p.y) })).sort((a, b) => a.cy - b.cy).map(({ p, i, cx, cy }) => {
    const name = D.show[p.m].name.replace(" (reference)", ""), w = name.length * 7.2;
    const anchor = cx > W * 0.7 ? "end" : "start", dx = anchor === "end" ? -9 : 9;
    const left = anchor === "end" ? cx + dx - w : cx + dx;
    let ly = cy - 8;
    while (placed.some(b => left < b.r && left + w > b.l && Math.abs(ly - b.y) < 15)) ly += 15;
    placed.push({ l: left, r: left + w, y: ly });
    return `<g data-i="${i}"><circle cx="${cx}" cy="${cy}" r="12" fill="transparent"/><circle cx="${cx}" cy="${cy}" r="5.5" fill="${gcol(D.show[p.m].group)}" stroke="var(--surface)" stroke-width="2"/><text class="lab" x="${cx+dx}" y="${ly}" text-anchor="${anchor}">${name}</text></g>`;
  }).join("");
  host.insertAdjacentHTML("beforeend", `<svg width="${W}" height="${H}" role="img" aria-label="${host.querySelector('h3').textContent}">${g}${marks}</svg><div class="tip"></div>`);
  const tip = host.querySelector(".tip");
  host.querySelectorAll("svg g[data-i]").forEach(el => {
    el.addEventListener("mouseenter", () => { const p = pts[+el.dataset.i]; tip.innerHTML = `<b>${D.show[p.m].name}</b><br>quality ${pct(p.y)} · ${xFmt(p.x)}`; tip.style.opacity = 1; });
    el.addEventListener("mousemove", e => { const b = host.getBoundingClientRect(); tip.style.left = (e.clientX - b.left + 12) + "px"; tip.style.top = (e.clientY - b.top - 10) + "px"; });
    el.addEventListener("mouseleave", () => tip.style.opacity = 0);
  });
}
// gap chart: one row per metric, one dot per model (Opus, the mode's teacher, base 9B)
function gap() {
  const host = document.getElementById("c-gap"); host.innerHTML = "";
  const T = tab === "overall" ? "interview" : tab;
  const teacher = (D.teacher[T] || {}).model;
  const who = ["reference", teacher, "qwen3.5-9b"].filter((m, i, a) => m && a.indexOf(m) === i && cur(m));
  const rows = [["quality", "Quality"], ["rules", "Rules kept"], ["same_move", "Same move as Opus"]];
  if (tab === "assess") rows.push(["same_grade", "Same grade as Opus"]);
  rows.push(["judge", "Judge score (÷5)"], ["win_tie", "Judge prefers it or ties vs Opus"]);
  if (tab === "overall" || tab === "interview") rows.push(["robust", "Same move on messy English"]);
  const val = (m, k) => { const r = cur(m) || {}; if (k === "judge") return r.judge ? { v: r.judge.v / 5, lo: r.judge.lo / 5, hi: r.judge.hi / 5 } : null;
    if (k === "robust") return D.robust[m] || null; return r[k] || null; };
  const W = Math.max(320, host.clientWidth - 24), L = Math.min(250, W * 0.42), R = 24, rowH = 34, H = rows.length * rowH + 40;
  const sx = v => L + v * (W - L - R);
  let g = "";
  for (const v of [0, .25, .5, .75, 1]) g += `<line x1="${sx(v)}" x2="${sx(v)}" y1="8" y2="${H-28}" stroke="var(--grid)"/><text x="${sx(v)}" y="${H-12}" text-anchor="middle">${v*100}%</text>`;
  rows.forEach(([k, label], i) => {
    const y = 20 + i * rowH;
    g += `<text class="lab" x="${L-10}" y="${y+4}" text-anchor="end">${label}</text>`;
    const vs = who.map(m => [m, val(m, k)]).filter(([, x]) => x && x.v != null);
    if (vs.length > 1) { const xs = vs.map(([, x]) => sx(x.v)); g += `<line x1="${Math.min(...xs)}" x2="${Math.max(...xs)}" y1="${y}" y2="${y}" stroke="var(--border)" stroke-width="2"/>`; }
    vs.forEach(([m, x]) => {
      if (x.lo != null) g += `<line x1="${sx(x.lo)}" x2="${sx(x.hi)}" y1="${y}" y2="${y}" stroke="${gcol(D.show[m].group)}" stroke-width="5" stroke-linecap="round" opacity=".28"/>`;
      g += `<circle cx="${sx(x.v)}" cy="${y}" r="6" fill="${gcol(D.show[m].group)}" stroke="var(--surface)" stroke-width="2"><title>${D.show[m].name}: ${(x.v*100).toFixed(0)}%</title></circle>`;
    });
  });
  host.innerHTML = `<div class="legend" style="margin:2px 4px 8px">${who.map(m => `<span><span class="dot" style="background:${gcol(D.show[m].group)}"></span>${D.show[m].name}${m === teacher ? " — training-data teacher for this mode" : ""}</span>`).join("")}</div><svg width="${W}" height="${H}" role="img" aria-label="Metric gaps">${g}</svg>`;
  document.getElementById("gap-note").textContent = `${tab === "overall" ? "Interview shown; pick a mode above." : ""} Each row is one metric; the shaded bar is its 95% interval. The distance between Opus and the base 9B is what fine-tuning has to close; the teacher is the open model chosen by the rule fixed before results (licence, no worse than Opus+2 points on giving the answer away, best quality, cheaper on a tie).`;
}
function draw() {
  gap();
  scatter("c-cost", "usd_turn", "US$ per turn (log)", v => v >= 1 ? "$" + v.toFixed(0) : "$" + v.toPrecision(1));
  scatter("c-lat", "p50_ms", "median seconds per turn (log)", v => (v >= 10 ? v.toFixed(0) : v.toPrecision(1)) + " s");
}
draw(); addEventListener("resize", draw);
</script>
</body>
</html>
"""


if __name__ == "__main__":
    data = collect()
    (store.ROOT / "index.html").write_text(render(data))
    (store.ROOT / "data" / "board_site.json").write_text(json.dumps(data, indent=1, default=str))
    print("wrote", store.ROOT / "index.html")
