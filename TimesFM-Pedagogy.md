# TimesFM Pedagogy: Forecasting Fibonacci

Give TimesFM the first 32 Fibonacci numbers, F(0) … F(31) = 0, 1, 1, 2, …, 1,346,269, and ask it for the next one:

**F(32) = 2,178,309**

TimesFM matches patterns; it doesn't calculate. It never learns the rule F(n) = F(n−1) + F(n−2). The question is which *view* of the data gives it a shape it recognizes.

## How to use TimesFM (it is not an LLM)

### What it is

TimesFM is a *time series foundation model*. It shares the transformer architecture with LLMs, but that's where the resemblance ends:

| | An LLM | TimesFM |
|---|---|---|
| Input | text tokens | numbers, and nothing else: no labels, units, dates or instructions |
| Output | a sampled next token | a probability distribution over future *values*: 9 quantiles (10% … 90%) per future step |
| Trained on | text | a large corpus of real and synthetic time series (web traffic, search trends, sales, energy, weather, …) |
| Can you explain the task? | yes, with a prompt | no; there is no prompt, and every input is just "continue this" |
| Can it apply a rule? | often (it can do arithmetic, follow "each term is the sum of the previous two") | never; it only recognizes *shapes* it has seen before |
| Knows the scale or units? | yes ("$2 million", "degrees") | no; it normalizes everything away first and only sees shape |
| Same input, same output? | only with temperature 0 | yes, always; nothing is sampled |

So "why did TimesFM get this wrong?" is almost never answered by "it misunderstood". The usual answer is "this shape looks like something that, in its training data, usually did something else next".

### The core recipe: transform → forecast → transform back

When your data's shape is unusual (exponential growth, bounded percentages, cumulative totals, spiky counts), don't feed it in raw. Feed in an *equivalent* series with a familiar shape, then convert the forecast back:

```
your data ──transform T──▶ familiar shape ──TimesFM──▶ forecast (in T-space)
                                                               │
answer   ◀──────────────── inverse of T ◀──────────────────────┘
```

Shapes TimesFM handles well, roughly from easiest to hardest:
**constant level < straight-line trend < repeating cycle < exponential growth.**
Real data rarely grows exponentially for long, so the model has learned *not* to expect it to continue.

**Rules for choosing T:**

1. **T must be invertible.** You need to get back to real units: log ↔ exp, ratio ↔ multiply by the last value, difference ↔ cumulative sum.
2. **Make T produce one of the easy shapes above.** log turns exponential growth into a line; ratios turn it into a flat level.
3. **The inverse may need the original data.** The ratio inverse needs F(31) to turn "1.62× bigger" into a number. Differencing needs the last value. Keep the raw series around.
4. **Prefer an increasing T** (bigger in → bigger out), like log or multiplying by a positive number. Then the 10% / 50% / 90% quantiles in T-space map straight to the 10% / 50% / 90% quantiles in real space, so you get uncertainty for free.
5. **Read the median, not the mean.** Medians survive an increasing inverse transform; means don't (exp of the mean of log x ≠ the mean of x). In MLX-Swift-TS, `prediction.mean` is already the median.
6. **Errors change shape on the way back.** An error ε in log-space becomes a *relative* error of about ε in real space. An error ε in a ratio becomes a relative error of ε / ratio. A transform only helps if the error is still small after inverting.

### What the model already does for you

- **Normalization (both versions):** each 32-step patch is rescaled by the running mean and standard deviation, and the forecast is scaled back. So units and scale never matter: 1.6 and 1,600,000 look the same if they rise and fall the same way. Normalization can shift and stretch, but it can't bend a curve into a line. That's why you sometimes need your own T on top.
- **Linear detrending (3.0 only):** if a straight line explains most of the history (removing it leaves less than half the original spread), 3.0 subtracts the line, forecasts the remainder, and adds the line back. That's a built-in "transform → forecast → transform back" for linear trends. It's why the log transform is so much more accurate on 3.0 below.
- **Horizon:** 2.5 predicts 128 steps per pass and feeds its own median back in for longer horizons. 3.0 predicts any horizon in one pass.
- **Several related series (3.0 only):** series passed together as `[B, V, T]` share information through *variate attention*.

### Checklist before trusting a forecast

1. Is the history's shape one the model has likely seen (level, trend, cycle)? If not, transform it.
2. Is the context long enough to show the pattern? A cycle needs at least one or two full periods in the history.
3. Does the 80% interval make sense? Too narrow and wrong is overconfidence. Absurdly wide, or covering impossible values (such as negative counts), means the model is guessing.
4. Did you invert correctly, including the quantiles?

## Fibonacci: three views of the same data

We try three:

| Case | What the model is given | Shape it sees | How to get F(32) back |
|---|---|---|---|
| **raw** | F(0) … F(31) | flat for ~25 steps, then a sudden spike | nothing to undo |
| **log** | log F(1) … log F(31) | an almost straight line, slope ≈ 0.481 (= log φ) | F(32) = exp(forecast) |
| **ratio** | F(n) / F(n−1) for n = 2 … 31 | wobbles, then flat at φ ≈ 1.618 | F(32) = forecast × F(31) |

All numbers below come from `TimesFMFibonacciTests` and `TimesFM3FibonacciTests`. Each value is the median (50% quantile) forecast; the interval is the 10%–90% quantile range, i.e. the model's 80% prediction interval.

## Models and precision

| Model | Weights file | Stored as | Computed in |
|---|---|---|---|
| TimesFM 2.5 fp16 | `kunal732/timesfm-2.5-200m-transformers-mlx` | fp16 (472 MB) | fp32 |
| TimesFM 2.5 fp32 | `converted/timesfm25-fp32` | fp32 (925 MB), Google's exact weights | fp32 |
| TimesFM 3.0 fp32 | `converted/timesfm3-fp32` | fp32 (1.32 GB), bit-identical to Google's release | fp32 |
| TimesFM 3.0 fp16 | `converted/timesfm3-fp16` | fp16 (661 MB) | fp32 |

All four compute in fp32. "fp16" means only that each weight was rounded once, when the file was written. Each fp16 file is exactly its fp32 sibling rounded to float16 (checked tensor by tensor). So within a version, fp16 vs fp32 isolates precision, and 2.5 vs 3.0 isolates the architecture.

## 1. Forecast error by case

Signed error of the median forecast against the true F(32) = 2,178,309.

| Case | 2.5 fp16 | 2.5 fp32 | 3.0 fp32 | 3.0 fp16 |
|---|---|---|---|---|
| **raw** | 1,389,242 (**−36.2%**) | 1,389,695 (**−36.2%**) | 3,433,327 (**+57.6%**) | 3,243,824 (**+48.9%**) |
| **log** | 2,299,482 (+5.56%) | 2,298,598 (+5.52%) | 2,182,360 (+0.186%) | 2,182,363 (+0.186%) |
| **ratio** | 2,181,586 (+0.150%) | 2,181,586 (+0.150%) | 2,183,296 (+0.229%) | 2,183,357 (+0.232%) |

- **raw:** both versions fail, in opposite directions. 2.5 expects the spike to fade; 3.0 expects it to keep going.
- **log:** 3.0 is 30× more accurate than 2.5. TimesFM 3.0's new *linear detrending* step spots that log F(n) is almost a straight line. It subtracts the line, forecasts only the small leftover wiggle, then adds the line back. Our transform and the model's own preprocessing work together.
- **ratio:** both versions are good. It's the best case for 2.5, and for 3.0 it's slightly behind log. Which transform works best depends on the model.

## 2. Uncertainty (80% prediction interval)

| Case | Model | 10% – 90% interval | Width | Contains 2,178,309? |
|---|---|---|---|---|
| **raw** | 2.5 fp16 | 734,049 – 2,165,218 | 1,431,169 | ❌ just above the 90% bound |
| **raw** | 2.5 fp32 | 734,223 – 2,166,033 | 1,431,810 | ❌ just above the 90% bound |
| **raw** | 3.0 fp32 | −4,980,279 – 9,373,399 | 14,353,678 | ✅ (the interval covers almost everything) |
| **raw** | 3.0 fp16 | −5,591,254 – 9,659,729 | 15,250,984 | ✅ (the interval covers almost everything) |
| **log** | 2.5 fp16 | 1,163,808 – 5,493,706 | 4,329,898 | ✅ |
| **log** | 2.5 fp32 | 1,162,336 – 5,512,778 | 4,350,442 | ✅ |
| **log** | 3.0 fp32 | 2,177,035 – 2,189,046 | 12,010 | ✅ |
| **log** | 3.0 fp16 | 2,177,027 – 2,188,918 | 11,891 | ✅ |
| **ratio** | 2.5 fp16 | 2,177,862 – 2,184,726 | 6,864 | ✅ |
| **ratio** | 2.5 fp32 | 2,177,860 – 2,184,735 | 6,874 | ✅ |
| **ratio** | 3.0 fp32 | 2,172,394 – 2,189,959 | 17,565 | ✅ |
| **ratio** | 3.0 fp16 | 2,172,588 – 2,189,903 | 17,315 | ✅ |

- **raw, 2.5:** the model is *confidently wrong*. Its interval is narrow and misses the truth.
- **raw, 3.0:** the model admits it's guessing. The interval is 14–15 million wide and goes negative, for a quantity that's never negative. It "contains" the truth only because it contains nearly everything.
- **log and ratio:** the intervals are tight and correct. The one exception is 2.5 on log, where exp turns a modest log-space spread into a ±2× range.

## 3. Drift: fp16 vs fp32

How far the median moves when the same model runs with fp16-rounded weights instead of Google's exact fp32 weights.

| Case | Version | fp32 median | fp16 median | Gap in F(32) | Gap vs fp32 median | Model's confidence |
|---|---|---|---|---|---|---|
| **raw** | 2.5 | 1,389,695 | 1,389,242 | 453 | 0.033% | high (and wrong) |
| **raw** | 3.0 | 3,433,327 | 3,243,824 | 189,503 | **5.52%** | none (14M-wide interval) |
| **log** | 2.5 | 2,298,598 | 2,299,482 | 884 | 0.038% | moderate |
| **log** | 3.0 | 2,182,360 | 2,182,363 | 2.3 | 0.0001% | high |
| **ratio** | 2.5 | 2,181,586 | 2,181,586 | 0.7 | 0.00003% | high |
| **ratio** | 3.0 | 2,183,296 | 2,183,357 | 61.7 | 0.0028% | high |

Rounding the weights to fp16 changes each one by only about 0.02%. When the model is confident, that barely moves the answer, even if the answer is wrong (2.5 raw). When it's unsure (3.0 raw), its median sits on a flat, uncertain part of its output, and the same tiny change slides it about 190,000. So fp16's cost isn't a fixed percentage: it's largest where the forecast is least certain. On smooth everyday series (see the golden-value tests), fp16 moves TimesFM 3.0 forecasts by only about 0.2–0.4% of the series' spread.

## Takeaways

1. **Transform the data into a shape the model knows.** Raw exponential growth fails on both versions. log and ratio turn it into a line and a flat level, and errors fall from 36–58% to under 0.25%.
2. **Check the model's own preprocessing.** TimesFM 3.0's linear detrending gives the log transform its 30× improvement. The same transform does very differently on two model versions.
3. **Read the interval, not just the median.** A narrow interval that misses (2.5 raw) and an enormous one that "hits" (3.0 raw) are both warnings.
4. **fp16 is fine where the model is confident.** Use fp32 when you need Google's exact numbers, or when the forecast is very uncertain.

## Intuition experiments (measured)

Eleven short experiments, each isolating one idea about how a *pattern model* behaves, especially where it differs from an LLM. All run on the four checkpoints above; see `TimesFMIntuitionTests.swift` for the code and detailed comments. fp16 and fp32 agree closely on all of them (usually within 0.1%; at most ~1.6%, on the least certain forecast, #8 with a short history), so one number per version is shown.

| # | Experiment | What we fed it | Result: TimesFM 2.5 | Result: TimesFM 3.0 | Lesson |
|---|---|---|---|---|---|
| 1 | **Scale invariance** | a two-cycle wave; the same ×1000; the same +10,000 | ×1000 differs by ~1e-6 of the spread; +10,000 by ~1e-4 | same | It sees shape, never units. (The +10,000 gap is float32 rounding of the input itself.) |
| 2 | **Primes** | the first 50 primes (2 … 229) | 242, 242, 247, 254, 259 (truth 233, 239, 241, 251, 257) | 232.7, 236.6, 240.6, 244.6, 248.7: a steady +4 per step | A rule with no shape can't be learned; it extends the trend. Prime *gaps* are forecast as ~4 (the average), with an interval of ~1.5–8.5. |
| 3 | **Digits of π** | 256 digits | medians 3.9–4.6; interval ~0.7–8.3; error 2.04 | medians 4.4–4.6; interval ~0.6–8.5; error 2.10 | Pure noise gives "the average, uncertain". Always guessing 4.53 scores 2.10, so neither version beats it. |
| 4 | **Squares n²** | 0, 1, 4, …, 3969 | raw: worst error 2.9%; differenced: 0.02% | raw: 0.29%; differenced: **exactly 0** | Differencing turns a curve into a line (inverse: running sum from the last value). 3.0's detrending then fits 2n − 1 perfectly. |
| 5 | **Multiplicative seasonality** | 10 years of monthly sales: +20%/yr growth, December +40% | raw 4.0% error; log 0.99% | raw 2.1%; log 0.24% | log turns swings that grow with the level into fixed-size ones. Raw forecasts under-shoot the growing December spike. |
| 6 | **Percentages** | a noisy rate cycling between ~0.5% and 3.5% | raw 10% quantile dips to −0.06%; logit stays ≥ 0.11%. Logit median error 6% *lower* | raw dips to −0.08%; logit ≥ 0.10%. Logit median error 63% *higher* | logit guarantees valid rates, but can cost accuracy. Measure both. |
| 7 | **Counts with zeros** | 256 days, 70% zeros, average demand 0.85/day | median ≈ 0.04/day; 90% quantile ≈ 3.5; 10% quantile down to −0.09 | median ≈ 0.05/day; 90% ≈ 3.5; 10% down to −0.04 | The median of "mostly zero" *is* zero; use the mean or an upper quantile for stocking. log(1 + x) still returns slightly negative values (down to −0.04); clip at 0. |
| 8 | **Context vs period** | a sine with period 200: last 512 vs last 64 steps | error 0.014 vs 0.10 (7× worse) | error 0.004 vs 0.14 (36× worse) | It can only continue a cycle it can see. Give it one or two full periods of history. |
| 9 | **Random walk** | 512 coin-flip steps | median drifts to +2.1 by step 128; 80% width 2.7 / 10.7 / 18.1 at h = 1 / 32 / 128 | median within 0.25; width 2.8 / 14.6 / 29.9 | Ideal widths are 2.6 / 14.5 / 29.0 (they grow like √h). 3.0 matches almost exactly; 2.5 is overconfident far ahead. |
| 10 | **Leading indicator** | B = A shifted 5 steps later; B alone vs A and B together | identical: 2.5 treats series separately | forecast changes (by up to 0.06) but gets slightly *worse*: error 0.183 vs 0.166 | Variate attention mixes series, but hasn't learned "copy the leader". A capability, not a guarantee. |
| 11 | **Regime change** | level 10, then a jump to 20 three, ten or forty steps before the end | next step → 20 steps: 18.3 → 10.7 · 19.8 → 15.6 · 20.0 → 19.9 | 19.2 → 15.2 · 20.0 → 19.6 · 19.8 → 19.6 | The fresher the jump, the more it hedges back to the old level. 3.0 accepts the new level faster. |

## Multivariate and "Sequences and Series" experiments (measured)

Eighteen more experiments, in three files: `TimesFMMultivariateTests.swift` (1–6), `TimesFMSeriesTests.swift` (7–14, ideas from second-semester calculus) and `TimesFMSeriesMultivariateTests.swift` (15–18). Numbers are fp32; fp16 is within a few percent. Variates are forecast together from their pasts only; they are not covariates. TimesFM 2.5 treats variates independently (a control in every multivariate test); 3.0 has variate attention.

| # | Experiment | Result | Lesson |
|---|---|---|---|
| 1 | Lagged copy (B = A delayed 8) | Neither model copies; error is 1–4% better than guessing 0. 3.0 shifts B by up to 0.04 | Build the lagged column yourself |
| 2 | Unrelated companion | 2.5 unchanged; 3.0 moves a random walk by 23% of its std and the seasonal series gets ~16% worse | Don't bundle unrelated series into one 3.0 call untested |
| 3 | Different scales (20 vs 5000) | Rescaling changes forecasts by ≤ 2.5e-6 of std | Per-variate normalisation works |
| 4 | Conserved total (A + B = 100) | Gap 0.335 (2.5), 0.084 (3.0); deriving B = 100 − A gives 0 | Derive for consistency, not accuracy |
| 5 | Order swap | Differences ≤ 3e-6 of std | Variate order doesn't matter |
| 6 | Noisy twin | 3.0 error 0.568 → 0.382 with a clean twin; 2.5 unchanged (0.931) | Variate attention helps most here |
| 7 | Geometric a·rⁿ | Log route: 3.0 exact to ~1e-6; 2.5 7–19% off. Raw is orders of magnitude worse | Take logs (|a| and restore the sign for r < 0) |
| 8 | Σ1/n² vs Σ1/n | Harmonic climb forecast within 6%; forecasting terms then summing cuts worst error 2.2–2.7× | The model doesn't flatten the slow divergence over 64 steps |
| 9 | Leibniz series → π | Right side of π at 40/40 steps; MAE ~7e-4 | Zig-zag reproduced |
| 10 | Taylor T(x) vs sin x | T's climb underestimated (ends 2.9 / 5.6 vs 7.04) | It continues the pattern shown, not the function |
| 11 | Fixed-point iteration | Within 1.5% (cos) and 0.1% (√2) of the limit; 3.0 hedges ~3× more on cos | A settled sequence isn't automatically easy |
| 12 | Logistic map | r = 2.8, 3.3: 40–250× better than the mean. Chaos (3.9): no better than the mean; 80% band covers 75–78% | Deterministic ≠ forecastable; bands stay honest |
| 13 | Growth ln n, n², 2ⁿ, n! | 2ⁿ log route: 3.0 exact, 2.5 42–98% off. n! raw is NaN (2.5) or clipped (3.0) | Log helps exponentials, not gentle growth; n! breaks everything |
| 14 | Ratio test | Ratios forecast to < 0.008 (2.5), < 0.0007 (3.0); 1/n and 1/n² both look "just under 1" | A transformed view answers only what that view can |
| 15 | Terms + partial sums | Forecast terms then add up: 59× (2.5) and 19× (3.0) better than direct | Forecast increments, let arithmetic accumulate |
| 16 | Taylor + sin together | 3.0 halves T's error (0.27 → 0.13), sin barely changes | Mixing helped this pair by accident, not understanding |
| 17 | Convergent + divergent together | 3.0 mixing changes forecasts by ≤ 4e-3 | No leakage between two smooth curves |
| 18 | Series and remainder | Deriving A = L − B̂ is 2.7× better on 2.5; 3.0 joint keeps A + B = L 17× tighter | Forecast the cleaner variate, derive the other |

## Reproduce

```bash
xcodebuild test -scheme mlxtoto-Package -destination 'platform=macOS' \
  -only-testing:MLXTimeSeriesTests/TimesFMFibonacciTests \
  -only-testing:MLXTimeSeriesTests/TimesFM3FibonacciTests \
  -only-testing:MLXTimeSeriesTests/TimesFMIntuitionTests \
  -only-testing:MLXTimeSeriesTests/TimesFMMultivariateTests \
  -only-testing:MLXTimeSeriesTests/TimesFMSeriesTests \
  -only-testing:MLXTimeSeriesTests/TimesFMSeriesMultivariateTests
```

TimesFM 2.5 fp16 downloads from the Hub (`kunal732/timesfm-2.5-200m-transformers-mlx`). The other three are converted locally:

```bash
python Scripts/convert_ts_model.py --hf-path google/timesfm-2.5-200m-pytorch --model-type timesfm \
  --mlx-path converted/timesfm25-fp32 --dtype float32
python Scripts/convert_ts_model.py --hf-path google/timesfm-3.0-pytorch --mlx-path converted/timesfm3-fp32 --dtype float32
python Scripts/convert_ts_model.py --hf-path google/timesfm-3.0-pytorch --mlx-path converted/timesfm3-fp16
```

Any checkpoint that's missing is skipped.
