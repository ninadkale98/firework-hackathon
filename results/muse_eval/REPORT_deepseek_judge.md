# Leaderboard metrics, 300-turn shared sample, judge: DeepSeek V4.1 Flash

Values are rates with 95% bootstrap intervals and n. Definitions: EVALUATION.md section 3.

## All modes

| Metric | Opus 5 (reference) | DeepSeek V4.1 Flash (teacher) | Qwen3.5-9B base | Muse Glimmer 30B base | Muse Glimmer 30B tuned |
|---|---|---|---|---|---|
| Quality | 95% [94–96] n=300 | 95% [95–96] n=299 | 82% [80–84] n=300 | 92% [91–93] n=284 | 94% [94–95] n=300 |
| Valid output | 99% [98–100] n=300 | 100% [99–100] n=300 | 99% [98–100] n=300 | 95% [92–97] n=300 | 100% [100–100] n=300 |
| Rules kept | 98% [97–100] n=300 | 99% [98–100] n=299 | 85% [80–88] n=300 | 96% [94–98] n=284 | 98% [97–100] n=300 |
| Gives answer away | 1% [0–3] n=300 | 0% [0–1] n=299 | 10% [7–13] n=300 | 0% [0–1] n=284 | 1% [0–3] n=300 |
| Resists fishing / injection | 97% [93–100] n=71 | 97% [93–100] n=70 | 73% [62–83] n=71 | 85% [76–93] n=67 | 97% [93–100] n=71 |
| Reveals answer when moving on | 0% [0–0] n=87 | 0% [0–0] n=111 | 11% [6–16] n=141 | 3% [0–5] n=150 | 0% [0–0] n=93 |
| Same move as Opus | – | 64% [58–70] n=298 | 42% [37–48] n=298 | 45% [39–50] n=298 | 65% [59–70] n=298 |
| Same grade as Opus (Assess) | – | 97% [92–100] n=38 | 68% [51–84] n=37 | 94% [84–100] n=31 | 82% [70–95] n=40 |
| Judge 1–5 | 4.32 [4.24–4.39] | 4.36 [4.29–4.43] | 3.16 [3.04–3.28] | 4.00 [3.90–4.08] | 4.25 [4.18–4.32] |
| Judge prefers it or ties vs Opus | – | 58% [53–64] n=299 | 7% [4–10] n=300 | 24% [19–29] n=284 | 41% [35–47] n=300 |
| Latency p50 | 9.2 s | 12.0 s | 5.1 s | 18.0 s | 8.0 s |
| Median output tokens | – | 1,354 | 1,434 | 1,720 | 278 |

## Interview

| Metric | Opus 5 (reference) | DeepSeek V4.1 Flash (teacher) | Qwen3.5-9B base | Muse Glimmer 30B base | Muse Glimmer 30B tuned |
|---|---|---|---|---|---|
| Quality | 94% [93–95] n=100 | 95% [94–96] n=100 | 79% [75–82] n=100 | 91% [89–92] n=100 | 94% [93–95] n=100 |
| Valid output | 99% [97–100] n=100 | 100% [100–100] n=100 | 97% [93–100] n=100 | 100% [100–100] n=100 | 100% [100–100] n=100 |
| Rules kept | 100% [100–100] n=100 | 99% [97–100] n=100 | 80% [72–88] n=100 | 99% [97–100] n=100 | 99% [97–100] n=100 |
| Gives answer away | 0% [0–0] n=100 | 0% [0–0] n=100 | 18% [11–26] n=100 | 0% [0–0] n=100 | 1% [0–3] n=100 |
| Resists fishing / injection | 100% [100–100] n=21 | 95% [86–100] n=21 | 86% [67–100] n=21 | 95% [86–100] n=21 | 100% [100–100] n=21 |
| Reveals answer when moving on | 0% [0–0] n=46 | 0% [0–0] n=49 | 9% [2–17] n=58 | 3% [0–7] n=95 | 0% [0–0] n=31 |
| Same move as Opus | – | 65% [55–74] n=99 | 38% [29–47] n=99 | 34% [24–43] n=99 | 65% [55–74] n=99 |
| Same grade as Opus (Assess) | – | – | – | – | – |
| Judge 1–5 | 4.19 [4.07–4.31] | 4.29 [4.19–4.39] | 2.99 [2.81–3.16] | 3.64 [3.51–3.77] | 4.22 [4.13–4.32] |
| Judge prefers it or ties vs Opus | – | 55% [45–65] n=100 | 7% [2–13] n=100 | 13% [7–20] n=100 | 38% [28–48] n=100 |
| Latency p50 | 13.2 s | 13.6 s | 15.9 s | 19.6 s | 7.9 s |
| Median output tokens | – | 1,524 | 6,142 | 1,758 | 251 |

## Assess

| Metric | Opus 5 (reference) | DeepSeek V4.1 Flash (teacher) | Qwen3.5-9B base | Muse Glimmer 30B base | Muse Glimmer 30B tuned |
|---|---|---|---|---|---|
| Quality | 94% [92–96] n=100 | 95% [94–96] n=100 | 83% [80–86] n=100 | 94% [92–95] n=84 | 94% [93–95] n=100 |
| Valid output | 99% [97–100] n=100 | 100% [100–100] n=100 | 100% [100–100] n=100 | 84% [77–91] n=100 | 100% [100–100] n=100 |
| Rules kept | 98% [95–100] n=100 | 100% [100–100] n=100 | 89% [83–95] n=100 | 99% [96–100] n=84 | 99% [97–100] n=100 |
| Gives answer away | 1% [0–3] n=100 | 0% [0–0] n=100 | 3% [0–6] n=100 | 1% [0–4] n=84 | 1% [0–3] n=100 |
| Resists fishing / injection | 95% [85–100] n=20 | 100% [100–100] n=20 | 60% [40–80] n=20 | 100% [100–100] n=16 | 100% [100–100] n=20 |
| Reveals answer when moving on | 0% [0–0] n=41 | 0% [0–0] n=62 | 13% [6–22] n=83 | 2% [0–7] n=55 | 0% [0–0] n=62 |
| Same move as Opus | – | 74% [66–82] n=99 | 47% [37–57] n=99 | 54% [43–64] n=99 | 78% [70–86] n=99 |
| Same grade as Opus (Assess) | – | 97% [92–100] n=38 | 68% [51–84] n=37 | 94% [84–100] n=31 | 82% [70–95] n=40 |
| Judge 1–5 | 4.29 [4.17–4.42] | 4.22 [4.10–4.33] | 3.03 [2.81–3.26] | 4.14 [3.96–4.30] | 4.14 [4.02–4.27] |
| Judge prefers it or ties vs Opus | – | 46% [36–56] n=100 | 4% [1–8] n=100 | 26% [18–36] n=84 | 43% [34–53] n=100 |
| Latency p50 | 9.6 s | 15.0 s | 13.1 s | 20.1 s | 9.6 s |
| Median output tokens | – | 1,748 | 4,948 | 2,294 | 385 |

## Study

| Metric | Opus 5 (reference) | DeepSeek V4.1 Flash (teacher) | Qwen3.5-9B base | Muse Glimmer 30B base | Muse Glimmer 30B tuned |
|---|---|---|---|---|---|
| Quality | 96% [94–97] n=100 | 97% [95–98] n=99 | 85% [82–88] n=100 | 92% [90–94] n=100 | 95% [93–96] n=100 |
| Valid output | 100% [100–100] n=100 | 99% [97–100] n=100 | 100% [100–100] n=100 | 100% [100–100] n=100 | 100% [100–100] n=100 |
| Rules kept | 97% [93–100] n=100 | 98% [95–100] n=99 | 85% [78–92] n=100 | 91% [85–96] n=100 | 97% [93–100] n=100 |
| Gives answer away | 3% [0–7] n=100 | 1% [0–3] n=99 | 8% [3–14] n=100 | 0% [0–0] n=100 | 2% [0–5] n=100 |
| Resists fishing / injection | 97% [90–100] n=30 | 97% [90–100] n=29 | 73% [57–87] n=30 | 70% [53–87] n=30 | 93% [83–100] n=30 |
| Reveals answer when moving on | – | – | – | – | – |
| Same move as Opus | – | 55% [46–64] n=100 | 41% [31–51] n=100 | 46% [36–54] n=100 | 53% [43–63] n=100 |
| Same grade as Opus (Assess) | – | – | – | – | – |
| Judge 1–5 | 4.48 [4.35–4.60] | 4.59 [4.47–4.70] | 3.47 [3.27–3.66] | 4.23 [4.06–4.40] | 4.39 [4.25–4.53] |
| Judge prefers it or ties vs Opus | – | 74% [65–82] n=99 | 9% [4–15] n=100 | 33% [24–43] n=100 | 42% [32–52] n=100 |
| Latency p50 | 6.2 s | 8.0 s | 3.0 s | 15.4 s | 7.7 s |
| Median output tokens | – | 860 | 600 | 1,162 | 284 |

## Base vs tuned Muse Glimmer, paired on the same turns (exact McNemar)

| Outcome | base only | tuned only | n | p |
|---|---|---|---|---|
| Valid output | 0 | 16 | 300 | 3.1e-05 |
| Same move as Opus | 38 | 99 | 298 | 1.9e-07 |
| Did not give answer away | 4 | 1 | 284 | 0.38 |
| Judge score ≥ 4 | 13 | 55 | 284 | 2.7e-07 |

## Judge agreement: DeepSeek V4.1 Flash vs GPT-5.4 on the same replies (n=899)

- Overall score identical: 49%; within 1 point: 95%
- 'Gives answer away' verdict identical: 97%

| Model | GPT-5.4 mean | DeepSeek mean | n |
|---|---|---|---|
| Opus 5 (reference) | 4.54 | 4.32 | 300 |
| DeepSeek V4.1 Flash (teacher) | 4.62 | 4.36 | 299 |
| Qwen3.5-9B base | 3.16 | 3.16 | 300 |
