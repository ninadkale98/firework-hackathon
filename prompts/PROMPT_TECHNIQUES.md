# Prompt techniques for the CodePraxis agent

Techniques to apply to the interviewer / examiner / tutor prompts, why each matters for this system, and how to check it. Ordered by expected value. The production prompt is unchanged; the copies in `prompts/variants.py` apply techniques 2–6 so they can be tested first.

| # | Technique | What it means here | Status |
|---|---|---|---|
| 1 | **Measure every change** | Fixed sample of test turns, the same code metrics (valid output, legal move, same move, same grade), then the judge on winners only. Never ship a prompt edit on intuition. | `scripts/prompt_ab.py` |
| 2 | **One-line job statement first** | Each mode opens with what the turn is for and what to output, before the long rules. | in `optimized`, `cot` |
| 3 | **Ordered decision procedure** | Turn the move list into "check in this order" steps, written against the failures we measured (assess: moving on before the rubric is covered; study: POINT instead of TRY; interview: FOLLOW_UP vs PROBE / HINT). | in `optimized`, `cot` |
| 4 | **Contrastive definitions for confusable moves** | Say how two similar moves differ and when to pick each (TRY = make it visible by running; POINT = it is already in lines they can see). Definitions alone don't separate them. | in the guide |
| 5 | **Few-shot examples from real turns** | Short situation → right move → why → reply, taken from training turns on questions that are not in the test set. Cover the confusable pairs. | 3 per mode, `prompts/fewshot/` |
| 6 | **Output contract and self-check at the end** | Restate the output rules last (recency): one move, an offered move, ids that exist, what `response` may contain. | in `optimized`, `cot` |
| 7 | **Reason before deciding (chain of thought)** | A short scratchpad before the JSON. Under strict JSON-schema output the JSON must come first, so this needs either no schema plus parsing, or a `reasoning` field placed *first* in the schema. For Muse Glimmer, the native place is its `to=self` thinking channel, which the tuned model must be trained to use (it currently skips it). | `cot` (no schema) |
| 8 | **Dynamic few-shot** | Pick the examples per turn by similarity (same mode, same move set, similar stage and time left) instead of a fixed three. | next |
| 9 | **Stable prefix, then per-turn data** | Keep all fixed instructions together at the top (block 0, already cache-marked) and move the per-mode `DECIDE` list out of block 1 into it. Better caching, and the rules are not buried in context. | next |
| 10 | **Concrete thresholds over adjectives** | "Under about a minute left → HINT" instead of "time is short". Numbers are followed more consistently. | partly in the guide |
| 11 | **Keep untrusted input fenced** | Learner text stays inside the `CANDIDATE_DATA` fences with the instruction to treat it as data. Already in production; keep it in every variant. | kept |
| 12 | **Remove redundancy and conflicts** | One place per rule; delete restatements that drift apart over edits. | next |
| 13 | **For the fine-tuned model, train on the new prompt** | v1 learned the exact production prompt. A better prompt helps a fine-tuned model most when its training data is regenerated with that prompt; at inference alone the change is off-distribution and can be ignored or even hurt. | next |

## How to run the comparison

```bash
python scripts/prompt_ab.py --per-mode 5                                         # free, local qwen3:8b via Ollama
python scripts/prompt_ab.py --backend fireworks --allow-fireworks --per-mode 20  # v1 deployment (costs credit)
```

Results land in `prompts/results/<backend>_<model>.jsonl`; the script prints valid / legal / same move / same grade per variant and mode.

## Caveats
- A local 8B model is a proxy. It shows whether the instructions are clear to a model that was not trained on them; it does not predict v1, which was trained on the original prompt.
- Same move as Opus has a ceiling below 100% (several moves can be reasonable), especially in Study. Use the judge before claiming a prompt is better.
