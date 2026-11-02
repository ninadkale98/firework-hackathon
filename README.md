# Make our own frontier: an open 9B model as an AI technical interviewer

Built for the Fireworks × Microsoft × NVIDIA hackathon, SF, 2026-11-02.

CodePraxis runs AI technical interviews and a Learn tutor with Claude. Each turn is one LLM call that reads a long, structured prompt and returns one JSON move. The prompt holds:

- the question and the current stage;
- the clocks;
- the files;
- the conversation so far;
- a rubric the model must never reveal.

The model picks one move — follow up, hint, clarify, redirect, probe deeper, or close the question — and writes what the candidate reads.

**The goal:** fine-tune small open models (**Qwen3.5-9B** and **Muse Glimmer 30B**) on Fireworks to make the same decisions as Claude Opus. The dimension we push is **cost and latency at equal quality**: a 2-day screening round for 50K students costs ~$50K on Opus and ~$2K on a tuned 9B.

**Write-up (results, architecture, baseline, training, evaluation):** https://ninadkale98.github.io/firework-hackathon/ · [leaderboard](https://ninadkale98.github.io/firework-hackathon/leaderboard.html)

**Result (300 held-out test turns, one judge for every model):** Muse Glimmer 30B tuned scores 4.25 against Opus's 4.32 (judge, 1–5), makes Opus's move 65% of the time (teacher 64%), and gives the answer away as rarely as Opus (1%). Qwen3.5-9B tuned cuts giving the answer away from 10% to 1%, raises resisting answer-fishing from 73% to 97%, and answers in 2.9 s against Opus's 9.2 s. DPO on the 9B is training.

## How it works

```
 TRAINING DATA                                         TEST SET (frozen)
 GPT-mini plays the candidate                          GPT-mini plays the candidate
   ↕ the production interview / Learn engine             ↕ the same engine
 an open teacher model interviews                      Claude Opus interviews (the reference)
   → GPT judge drops bad turns                           → ~1,500 turns: interview, Study, Assess
   → SFT LoRA on Qwen3.5-9B (Fireworks)

 Every model answers the same frozen test turns → code checks + LLM judge → leaderboard
```

- **The real engine, not a copy.** Sessions run through the production session service, driver, prompt builder, schema builder and checks. Only storage is in memory, and the clock is fake. So every recorded prompt is byte-identical to production, and a whole interview's worth of time passes instantly.
- **Variety on every prompt field.** Coverage is measured per field:
  - clocks, depth, hints, probe options, the set of moves offered, history length;
  - files: code, repo, image, PDF, video, drawings;
  - code-run results and Learn notes, gaps and memory;
  - 10 candidate personas, 8 writing styles including Hinglish, and drawings run through the production Excalidraw normaliser.

  Generation tops up until every bucket reaches its minimum.
- **Edits and pairs.** The same engine input is replayed with edge clocks, and with a messier rewrite of the candidate's answer for a robustness check. Both are rebuilt through the production prompt builder.
- **Split by question.** Test questions are never seen in training.

## Scoring

| Check | How |
|---|---|
| Format | valid JSON for the turn's schema, legal move, ids exist |
| Rules | no leak of the rubric or withheld answer parts; time-up behaviour; refuses injection and answer-fishing |
| Agreement | same move / probe / hint / grade as Opus |
| Quality | GPT judge: six criteria pass/fail with quoted evidence, overall 1–5, head-to-head win rate vs Opus |
| Robustness | same move on clean vs messy versions of an answer |
| Cost | latency p50/p95, $ per turn and per interview |

Every score comes with 95% bootstrap confidence intervals, broken down by task and by coverage bucket.

## Layout

| Path | What |
|---|---|
| `distill/models.py` | every model behind one completer interface: Claude, GPT, Fireworks serverless and deployments, local Ollama |
| `distill/bank.py` | the question bank, loaded into the production `QuestionManager` |
| `distill/engine.py` | runs one interview through the production engine and records every call; `replay` for edits |
| `distill/learn_engine.py` | the same for Learn (Study tutor, Assess) |
| `distill/simulator.py` | the GPT candidate: personas, writing styles, drawings |
| `distill/gen.py`, `distill/learn_gen.py` | sessions, clock edits, clean/messy pairs, with spend caps |
| `distill/corrupt.py` | deterministic messy-English rewrites |
| `distill/coverage.py` | the coverage report per field and per pair of fields |
| `distill/runs.py` | run models on the test set, score, judge, leaderboard |
| `distill/score.py`, `distill/judge.py` | code checks and the LLM judge |
| `distill/export.py` | records → Fireworks SFT JSONL |
| `STUDY.md` | the methods used: distillation, SFT, LoRA, DPO, RFT, LLM-as-judge, quantization |

The question content and generated data are in the data repo (below); API keys are never committed.

## Running

It needs the CodePraxis backend on `PYTHONPATH` (the production engine) and keys in a local `.env`:

```bash
PYTHONPATH=../../backend:. python -m distill.gen sessions --split test --interviewer opus-5 --sessions 60
PYTHONPATH=../../backend:. python -m distill.learn_gen --task study --split test --sessions 40
PYTHONPATH=../../backend:. python -m distill.runs run --split test --models qwen3.5-9b haiku-5.5 gpt-5.4
PYTHONPATH=../../backend:. python -m distill.runs score && python -m distill.runs board
PYTHONPATH=../../backend:. python -m distill.export --split train
```

## Continuing on another machine

The data (test set, training set, question bank, plan, logs) lives in [`ninadkale98/firework-hackathon-data`](https://github.com/ninadkale98/firework-hackathon-data). Its README has the full restore steps. To evaluate another model, use `data/export/test_eval.jsonl` there. In short:

1. Clone this repo and `codepraxis-org/backend` (private) at commit `ccee1c5`, which the data was made with. Create the backend venv.
2. Clone the data repo and unpack it into this folder with `zstd -d …`: `data/distill.db`, `data/export/*.jsonl`, `data/bank/`, `questions/`, `PLAN.md`.
3. Create `.env` with `FIREWORKS_API_KEY` and `DISTILL_OPENAI_API_KEY` (never committed). Claude runs read `ANTHROPIC_API_KEY` from the backend's `.env`.
4. Download `firectl` into `bin/` and recreate the `bin/fw` wrapper (commands in the data repo's README).

Status and next steps: `MORNING.md`.
