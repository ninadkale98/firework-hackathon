# Leaderboard metrics, 300-turn shared sample, judge: DeepSeek V4.1 Flash

Values are rates with 95% bootstrap intervals and n. Definitions: EVALUATION.md section 3.

## All modes

| Metric | Opus 5 (reference) | DeepSeek V4.1 Flash (teacher) | Qwen3.5-9B base | Muse Glimmer 30B base | Muse Glimmer 30B tuned |
|---|---|---|---|---|---|
| Quality | 95% [94–96] n=294 | 96% [95–96] n=291 | 83% [81–84] n=292 | 92% [91–93] n=277 | 94% [94–95] n=300 |
| Valid output | 99% [98–100] n=300 | 100% [99–100] n=300 | 99% [98–100] n=300 | 95% [92–97] n=300 | 100% [100–100] n=300 |
| Rules kept | 98% [97–100] n=294 | 100% [99–100] n=291 | 85% [81–89] n=292 | 96% [94–99] n=277 | 98% [97–100] n=300 |
| Gives answer away | 1% [0–3] n=294 | 0% [0–1] n=291 | 10% [7–13] n=292 | 0% [0–1] n=277 | 1% [0–3] n=300 |
| Resists fishing / injection | 97% [93–100] n=68 | 100% [100–100] n=55 | 72% [61–82] n=61 | 84% [74–93] n=58 | 97% [93–100] n=71 |
| Reveals answer when moving on | 0% [0–0] n=86 | 0% [0–0] n=110 | 11% [6–16] n=137 | 3% [1–6] n=144 | 0% [0–0] n=93 |
| Same move as Opus | – | 64% [59–70] n=298 | 42% [37–48] n=298 | 45% [39–50] n=298 | 65% [59–70] n=298 |
| Same grade as Opus (Assess) | – | 97% [92–100] n=38 | 68% [51–81] n=37 | 94% [84–100] n=31 | 82% [70–92] n=40 |
| Judge 1–5 | 4.34 [4.27–4.40] | 4.35 [4.28–4.42] | 3.16 [3.03–3.30] | 4.00 [3.90–4.09] | 4.25 [4.18–4.32] |
| Judge prefers it or ties vs Opus | – | 57% [51–63] n=270 | 7% [4–10] n=268 | 24% [19–29] n=263 | 41% [36–47] n=300 |
| Latency p50 | 9.2 s | 12.0 s | 5.1 s | 18.0 s | 8.0 s |
| Median output tokens | – | 1,354 | 1,434 | 1,720 | 278 |

## Interview

| Metric | Opus 5 (reference) | DeepSeek V4.1 Flash (teacher) | Qwen3.5-9B base | Muse Glimmer 30B base | Muse Glimmer 30B tuned |
|---|---|---|---|---|---|
| Quality | 95% [94–96] n=96 | 95% [95–96] n=99 | 79% [75–82] n=96 | 90% [89–92] n=94 | 94% [93–95] n=100 |
| Valid output | 99% [97–100] n=100 | 100% [100–100] n=100 | 97% [93–100] n=100 | 100% [100–100] n=100 | 100% [100–100] n=100 |
| Rules kept | 100% [100–100] n=96 | 100% [100–100] n=99 | 80% [72–88] n=96 | 99% [97–100] n=94 | 99% [97–100] n=100 |
| Gives answer away | 0% [0–0] n=96 | 0% [0–0] n=99 | 18% [10–26] n=96 | 0% [0–0] n=94 | 1% [0–3] n=100 |
| Resists fishing / injection | 100% [100–100] n=21 | 100% [100–100] n=16 | 83% [67–100] n=18 | 95% [84–100] n=19 | 100% [100–100] n=21 |
| Reveals answer when moving on | 0% [0–0] n=45 | 0% [0–0] n=48 | 7% [2–14] n=56 | 3% [0–8] n=89 | 0% [0–0] n=31 |
| Same move as Opus | – | 65% [55–74] n=99 | 38% [28–47] n=99 | 34% [25–43] n=99 | 65% [56–74] n=99 |
| Same grade as Opus (Assess) | – | – | – | – | – |
| Judge 1–5 | 4.25 [4.16–4.33] | 4.28 [4.19–4.38] | 2.99 [2.79–3.19] | 3.63 [3.49–3.76] | 4.22 [4.12–4.31] |
| Judge prefers it or ties vs Opus | – | 54% [44–64] n=91 | 8% [3–13] n=91 | 12% [6–20] n=88 | 38% [28–47] n=100 |
| Latency p50 | 13.2 s | 13.6 s | 15.9 s | 19.6 s | 7.9 s |
| Median output tokens | – | 1,524 | 6,142 | 1,758 | 251 |

## Assess

| Metric | Opus 5 (reference) | DeepSeek V4.1 Flash (teacher) | Qwen3.5-9B base | Muse Glimmer 30B base | Muse Glimmer 30B tuned |
|---|---|---|---|---|---|
| Quality | 94% [92–96] n=98 | 95% [94–96] n=99 | 84% [81–86] n=98 | 94% [92–95] n=84 | 94% [93–95] n=100 |
| Valid output | 99% [97–100] n=100 | 100% [100–100] n=100 | 100% [100–100] n=100 | 84% [76–91] n=100 | 100% [100–100] n=100 |
| Rules kept | 98% [95–100] n=98 | 100% [100–100] n=99 | 90% [84–95] n=98 | 99% [96–100] n=84 | 99% [97–100] n=100 |
| Gives answer away | 1% [0–3] n=98 | 0% [0–0] n=99 | 3% [0–7] n=98 | 1% [0–4] n=84 | 1% [0–3] n=100 |
| Resists fishing / injection | 95% [84–100] n=19 | 100% [100–100] n=18 | 61% [39–83] n=18 | 100% [100–100] n=13 | 100% [100–100] n=20 |
| Reveals answer when moving on | 0% [0–0] n=41 | 0% [0–0] n=62 | 14% [7–22] n=81 | 2% [0–5] n=55 | 0% [0–0] n=62 |
| Same move as Opus | – | 74% [66–82] n=99 | 47% [38–58] n=99 | 54% [43–63] n=99 | 78% [70–86] n=99 |
| Same grade as Opus (Assess) | – | 97% [92–100] n=38 | 68% [51–81] n=37 | 94% [84–100] n=31 | 82% [70–92] n=40 |
| Judge 1–5 | 4.29 [4.15–4.42] | 4.22 [4.10–4.33] | 3.04 [2.82–3.27] | 4.14 [3.98–4.31] | 4.14 [4.02–4.26] |
| Judge prefers it or ties vs Opus | – | 46% [36–56] n=91 | 5% [1–9] n=88 | 28% [18–38] n=80 | 43% [34–53] n=100 |
| Latency p50 | 9.6 s | 15.0 s | 13.1 s | 20.1 s | 9.6 s |
| Median output tokens | – | 1,748 | 4,948 | 2,294 | 385 |

## Study

| Metric | Opus 5 (reference) | DeepSeek V4.1 Flash (teacher) | Qwen3.5-9B base | Muse Glimmer 30B base | Muse Glimmer 30B tuned |
|---|---|---|---|---|---|
| Quality | 96% [94–97] n=100 | 97% [96–98] n=93 | 85% [82–88] n=98 | 92% [90–94] n=99 | 95% [93–97] n=100 |
| Valid output | 100% [100–100] n=100 | 99% [97–100] n=100 | 100% [100–100] n=100 | 100% [100–100] n=100 | 100% [100–100] n=100 |
| Rules kept | 97% [93–100] n=100 | 99% [97–100] n=93 | 86% [79–92] n=98 | 92% [86–97] n=99 | 97% [93–100] n=100 |
| Gives answer away | 3% [0–7] n=100 | 1% [0–3] n=93 | 8% [3–14] n=98 | 0% [0–0] n=99 | 2% [0–5] n=100 |
| Resists fishing / injection | 96% [89–100] n=28 | 100% [100–100] n=21 | 72% [56–88] n=25 | 69% [50–85] n=26 | 93% [83–100] n=30 |
| Reveals answer when moving on | – | – | – | – | – |
| Same move as Opus | – | 55% [45–65] n=100 | 41% [32–51] n=100 | 46% [36–56] n=100 | 53% [43–63] n=100 |
| Same grade as Opus (Assess) | – | – | – | – | – |
| Judge 1–5 | 4.48 [4.34–4.60] | 4.57 [4.44–4.69] | 3.46 [3.27–3.65] | 4.23 [4.05–4.39] | 4.39 [4.25–4.52] |
| Judge prefers it or ties vs Opus | – | 72% [62–81] n=88 | 10% [4–17] n=89 | 32% [22–40] n=95 | 42% [33–52] n=100 |
| Latency p50 | 6.2 s | 8.0 s | 3.0 s | 15.4 s | 7.7 s |
| Median output tokens | – | 860 | 600 | 1,162 | 284 |

## Base vs tuned Muse Glimmer, paired on the same turns (exact McNemar)

| Outcome | base only | tuned only | n | p |
|---|---|---|---|---|
| Valid output | 0 | 16 | 300 | 3.1e-05 |
| Same move as Opus | 38 | 99 | 298 | 1.9e-07 |
| Did not give answer away | 4 | 1 | 277 | 0.38 |
| Judge score ≥ 4 | 13 | 54 | 277 | 4.5e-07 |

## Judge agreement: DeepSeek V4.1 Flash vs GPT-5.4 on the same replies (n=877)

- Overall score identical: 49%; within 1 point: 95%
- 'Gives answer away' verdict identical: 97%

| Model | GPT-5.4 mean | DeepSeek mean | n |
|---|---|---|---|
| Opus 5 (reference) | 4.56 | 4.34 | 294 |
| DeepSeek V4.1 Flash (teacher) | 4.62 | 4.35 | 291 |
| Qwen3.5-9B base | 3.16 | 3.16 | 292 |
