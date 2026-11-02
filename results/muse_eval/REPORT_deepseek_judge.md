# Leaderboard metrics, 300-turn shared sample, judge: DeepSeek V4.1 Flash

Values are rates with 95% bootstrap intervals and n. Definitions: EVALUATION.md section 3.

## All modes

| Metric | Opus 5 (reference) | DeepSeek V4.1 Flash (teacher) | Qwen3.5-9B base | Muse Glimmer 30B base | Muse Glimmer 30B tuned |
|---|---|---|---|---|---|
| Quality | 95% [94–96] n=258 | 95% [95–96] n=254 | 83% [81–85] n=259 | 93% [92–94] n=168 | 94% [94–95] n=300 |
| Valid output | 99% [98–100] n=300 | 100% [99–100] n=300 | 99% [98–100] n=300 | 95% [92–97] n=300 | 100% [100–100] n=300 |
| Rules kept | 98% [97–100] n=258 | 100% [99–100] n=254 | 85% [81–89] n=259 | 99% [98–100] n=168 | 98% [97–100] n=300 |
| Gives answer away | 1% [0–3] n=258 | 0% [0–1] n=254 | 9% [6–13] n=259 | 1% [0–2] n=168 | 1% [0–3] n=300 |
| Resists fishing / injection | 97% [92–100] n=59 | 100% [100–100] n=48 | 71% [58–83] n=52 | 100% [100–100] n=15 | 97% [93–100] n=71 |
| Reveals answer when moving on | 0% [0–0] n=77 | 0% [0–0] n=100 | 10% [5–16] n=124 | 2% [0–6] n=86 | 0% [0–0] n=93 |
| Same move as Opus | – | 64% [59–70] n=298 | 42% [37–48] n=298 | 45% [39–51] n=298 | 65% [60–70] n=298 |
| Same grade as Opus (Assess) | – | 97% [92–100] n=38 | 68% [54–81] n=37 | 94% [84–100] n=31 | 82% [70–92] n=40 |
| Judge 1–5 | 4.34 [4.26–4.41] | 4.34 [4.27–4.41] | 3.20 [3.08–3.33] | 4.02 [3.90–4.14] | 4.25 [4.18–4.31] |
| Judge prefers it or ties vs Opus | – | 57% [51–63] n=237 | 8% [5–12] n=236 | 24% [17–31] n=118 | 41% [35–46] n=300 |
| Latency p50 | 9.2 s | 12.0 s | 5.1 s | 18.0 s | 8.0 s |
| Median output tokens | – | 1,354 | 1,434 | 1,720 | 278 |

## Interview

| Metric | Opus 5 (reference) | DeepSeek V4.1 Flash (teacher) | Qwen3.5-9B base | Muse Glimmer 30B base | Muse Glimmer 30B tuned |
|---|---|---|---|---|---|
| Quality | 95% [94–96] n=85 | 95% [94–96] n=88 | 79% [76–83] n=86 | 91% [90–92] n=59 | 94% [94–95] n=100 |
| Valid output | 99% [97–100] n=100 | 100% [100–100] n=100 | 97% [93–100] n=100 | 100% [100–100] n=100 | 100% [100–100] n=100 |
| Rules kept | 100% [100–100] n=85 | 100% [100–100] n=88 | 80% [72–88] n=86 | 100% [100–100] n=59 | 99% [97–100] n=100 |
| Gives answer away | 0% [0–0] n=85 | 0% [0–0] n=88 | 17% [9–26] n=86 | 0% [0–0] n=59 | 1% [0–3] n=100 |
| Resists fishing / injection | 100% [100–100] n=18 | 100% [100–100] n=14 | 81% [62–100] n=16 | 100% [100–100] n=5 | 100% [100–100] n=21 |
| Reveals answer when moving on | 0% [0–0] n=41 | 0% [0–0] n=44 | 6% [0–12] n=49 | 2% [0–5] n=56 | 0% [0–0] n=31 |
| Same move as Opus | – | 65% [55–74] n=99 | 38% [29–48] n=99 | 34% [25–43] n=99 | 65% [55–74] n=99 |
| Same grade as Opus (Assess) | – | – | – | – | – |
| Judge 1–5 | 4.25 [4.15–4.34] | 4.28 [4.17–4.40] | 3.03 [2.84–3.23] | 3.64 [3.47–3.81] | 4.22 [4.13–4.32] |
| Judge prefers it or ties vs Opus | – | 56% [44–65] n=81 | 9% [4–15] n=82 | 8% [0–18] n=40 | 38% [29–47] n=100 |
| Latency p50 | 13.2 s | 13.6 s | 15.9 s | 19.6 s | 7.9 s |
| Median output tokens | – | 1,524 | 6,142 | 1,758 | 251 |

## Assess

| Metric | Opus 5 (reference) | DeepSeek V4.1 Flash (teacher) | Qwen3.5-9B base | Muse Glimmer 30B base | Muse Glimmer 30B tuned |
|---|---|---|---|---|---|
| Quality | 95% [92–96] n=89 | 95% [94–95] n=89 | 83% [80–87] n=89 | 94% [92–95] n=55 | 94% [93–95] n=100 |
| Valid output | 99% [97–100] n=100 | 100% [100–100] n=100 | 100% [100–100] n=100 | 84% [76–91] n=100 | 100% [100–100] n=100 |
| Rules kept | 99% [97–100] n=89 | 100% [100–100] n=89 | 89% [82–96] n=89 | 98% [95–100] n=55 | 99% [97–100] n=100 |
| Gives answer away | 0% [0–0] n=89 | 0% [0–0] n=89 | 3% [0–8] n=89 | 2% [0–5] n=55 | 1% [0–3] n=100 |
| Resists fishing / injection | 94% [83–100] n=18 | 100% [100–100] n=16 | 56% [31–81] n=16 | 100% [100–100] n=5 | 100% [100–100] n=20 |
| Reveals answer when moving on | 0% [0–0] n=36 | 0% [0–0] n=56 | 13% [7–21] n=75 | 3% [0–10] n=30 | 0% [0–0] n=62 |
| Same move as Opus | – | 74% [65–82] n=99 | 47% [37–57] n=99 | 54% [43–64] n=99 | 78% [70–86] n=99 |
| Same grade as Opus (Assess) | – | 97% [92–100] n=38 | 68% [54–81] n=37 | 94% [84–100] n=31 | 82% [70–92] n=40 |
| Judge 1–5 | 4.29 [4.15–4.43] | 4.19 [4.08–4.30] | 3.08 [2.83–3.33] | 4.15 [3.91–4.35] | 4.14 [4.02–4.26] |
| Judge prefers it or ties vs Opus | – | 47% [36–59] n=83 | 5% [1–10] n=79 | 33% [18–49] n=39 | 43% [33–53] n=100 |
| Latency p50 | 9.6 s | 15.0 s | 13.1 s | 20.1 s | 9.6 s |
| Median output tokens | – | 1,748 | 4,948 | 2,294 | 385 |

## Study

| Metric | Opus 5 (reference) | DeepSeek V4.1 Flash (teacher) | Qwen3.5-9B base | Muse Glimmer 30B base | Muse Glimmer 30B tuned |
|---|---|---|---|---|---|
| Quality | 95% [93–97] n=84 | 97% [95–98] n=77 | 86% [82–89] n=84 | 95% [94–96] n=54 | 95% [93–96] n=100 |
| Valid output | 100% [100–100] n=100 | 99% [97–100] n=100 | 100% [100–100] n=100 | 100% [100–100] n=100 | 100% [100–100] n=100 |
| Rules kept | 96% [92–100] n=84 | 99% [96–100] n=77 | 87% [80–94] n=84 | 100% [100–100] n=54 | 97% [93–100] n=100 |
| Gives answer away | 4% [0–8] n=84 | 1% [0–4] n=77 | 7% [2–13] n=84 | 0% [0–0] n=54 | 2% [0–5] n=100 |
| Resists fishing / injection | 96% [87–100] n=23 | 100% [100–100] n=18 | 75% [55–95] n=20 | 100% [100–100] n=5 | 93% [83–100] n=30 |
| Reveals answer when moving on | – | – | – | – | – |
| Same move as Opus | – | 55% [45–65] n=100 | 41% [32–52] n=100 | 46% [36–55] n=100 | 53% [43–62] n=100 |
| Same grade as Opus (Assess) | – | – | – | – | – |
| Judge 1–5 | 4.49 [4.33–4.62] | 4.57 [4.43–4.70] | 3.49 [3.25–3.71] | 4.30 [4.11–4.46] | 4.39 [4.25–4.52] |
| Judge prefers it or ties vs Opus | – | 70% [59–79] n=73 | 11% [4–17] n=75 | 31% [18–46] n=39 | 42% [32–51] n=100 |
| Latency p50 | 6.2 s | 8.0 s | 3.0 s | 15.4 s | 7.7 s |
| Median output tokens | – | 860 | 600 | 1,162 | 284 |

## Base vs tuned Muse Glimmer, paired on the same turns (exact McNemar)

| Outcome | base only | tuned only | n | p |
|---|---|---|---|---|
| Valid output | 0 | 16 | 300 | 3.1e-05 |
| Same move as Opus | 38 | 99 | 298 | 1.9e-07 |
| Did not give answer away | 3 | 1 | 168 | 0.62 |
| Judge score ≥ 4 | 9 | 34 | 168 | 0.00017 |

## Judge agreement: DeepSeek V4.1 Flash vs GPT-5.4 on the same replies (n=771)

- Overall score identical: 49%; within 1 point: 94%
- 'Gives answer away' verdict identical: 97%

| Model | GPT-5.4 mean | DeepSeek mean | n |
|---|---|---|---|
| Opus 5 (reference) | 4.59 | 4.34 | 258 |
| DeepSeek V4.1 Flash (teacher) | 4.62 | 4.34 | 254 |
| Qwen3.5-9B base | 3.14 | 3.20 | 259 |
