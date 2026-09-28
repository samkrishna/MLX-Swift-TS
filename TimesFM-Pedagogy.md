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
| TimesFM 2.5 | `kunal732/timesfm-2.5-200m-transformers-mlx` | fp16 (472 MB), rounded from Google's fp32 | fp32 |
| TimesFM 3.0 fp32 | `converted/timesfm3-fp32` | fp32 (1.32 GB), bit-identical to Google's release | fp32 |
| TimesFM 3.0 fp16 | `converted/timesfm3-fp16` | fp16 (661 MB), rounded from Google's fp32 | fp32 |

All three compute in fp32. "fp16" means only that each weight was rounded once, when the file was written.

## 1. Forecast error by case

Signed error of the median forecast against the true F(32) = 2,178,309.

| Case | TimesFM 2.5 (fp16) | TimesFM 3.0 (fp32) | TimesFM 3.0 (fp16) |
|---|---|---|---|
| **raw** | 1,389,242 (**−36.2%**) | 3,433,327 (**+57.6%**) | 3,243,824 (**+48.9%**) |
| **log** | 2,299,482 (+5.56%) | 2,182,360 (+0.186%) | 2,182,363 (+0.186%) |
| **ratio** | 2,181,586 (+0.150%) | 2,183,296 (+0.229%) | 2,183,357 (+0.232%) |

- **raw:** both versions fail, in opposite directions. 2.5 expects the spike to fade; 3.0 expects it to keep going.
- **log:** 3.0 is 30× more accurate than 2.5. TimesFM 3.0's new *linear detrending* step spots that log F(n) is almost a straight line. It subtracts the line, forecasts only the small leftover wiggle, then adds the line back. Our transform and the model's own preprocessing work together.
- **ratio:** both versions are good. It's the best case for 2.5, and for 3.0 it's slightly behind log. Which transform works best depends on the model.

## 2. Uncertainty (80% prediction interval)

| Case | Model | 10% – 90% interval | Width | Contains 2,178,309? |
|---|---|---|---|---|
| **raw** | 2.5 (fp16) | 734,049 – 2,165,218 | 1,431,169 | ❌ just above the 90% bound |
| **raw** | 3.0 (fp32) | −4,980,279 – 9,373,399 | 14,353,678 | ✅ (the interval covers almost everything) |
| **raw** | 3.0 (fp16) | −5,591,254 – 9,659,729 | 15,250,984 | ✅ (the interval covers almost everything) |
| **log** | 2.5 (fp16) | 1,163,808 – 5,493,706 | 4,329,898 | ✅ |
| **log** | 3.0 (fp32) | 2,177,035 – 2,189,046 | 12,010 | ✅ |
| **log** | 3.0 (fp16) | 2,177,027 – 2,188,918 | 11,891 | ✅ |
| **ratio** | 2.5 (fp16) | 2,177,862 – 2,184,726 | 6,864 | ✅ |
| **ratio** | 3.0 (fp32) | 2,172,394 – 2,189,959 | 17,565 | ✅ |
| **ratio** | 3.0 (fp16) | 2,172,588 – 2,189,903 | 17,315 | ✅ |

- **raw, 2.5:** the model is *confidently wrong*. Its interval is narrow and misses the truth.
- **raw, 3.0:** the model admits it's guessing. The interval is 14–15 million wide and goes negative, for a quantity that's never negative. It "contains" the truth only because it contains nearly everything.
- **log and ratio:** the intervals are tight and correct. The one exception is 2.5 on log, where exp turns a modest log-space spread into a ±2× range.

## 3. Drift: fp16 vs fp32 (TimesFM 3.0)

How far the median moves when the same model runs with fp16-rounded weights instead of Google's exact fp32 weights.

| Case | fp32 median | fp16 median | Gap in F(32) | Gap vs fp32 median | Model's confidence |
|---|---|---|---|---|---|
| **raw** | 3,433,327 | 3,243,824 | 189,503 | **5.52%** | none (14M-wide interval) |
| **log** | 2,182,360 | 2,182,363 | 2.3 | 0.0001% | high |
| **ratio** | 2,183,296 | 2,183,357 | 61.7 | 0.0028% | high |

In the model's own (transformed) output space, the gaps are:

| Case | fp32 | fp16 | Relative gap |
|---|---|---|---|
| raw | 3,433,327.2 | 3,243,823.8 | 5.5 × 10⁻² |
| log | 14.595918 | 14.595919 | 6.5 × 10⁻⁸ |
| ratio | 1.6217378 | 1.6217839 | 2.8 × 10⁻⁵ |

Rounding the weights to fp16 changes each one by only about 0.02%. When the model is confident (log, ratio), that barely moves the answer. When it's unsure (raw), its median sits on a flat, uncertain part of its output, and the same tiny change slides it about 190,000. So fp16's cost isn't a fixed percentage: it's largest exactly where the forecast is least trustworthy anyway. On smooth everyday series (see the golden-value tests), fp16 moves TimesFM 3.0 forecasts by only about 0.2–0.4% of the series' spread.

## Takeaways

1. **Transform the data into a shape the model knows.** Raw exponential growth fails on both versions. log and ratio turn it into a line and a flat level, and errors fall from 36–58% to under 0.25%.
2. **Check the model's own preprocessing.** TimesFM 3.0's linear detrending gives the log transform its 30× improvement. The same transform does very differently on two model versions.
3. **Read the interval, not just the median.** A narrow interval that misses (2.5 raw) and an enormous one that "hits" (3.0 raw) are both warnings.
4. **fp16 is fine where the model is confident.** Use fp32 when you need Google's exact numbers, or when the forecast is very uncertain.

## More examples to build intuition (suggested, not yet run)

Each of these isolates one idea about how a *pattern model* behaves, especially where it differs from an LLM. None has been run yet. Any of them could become a test in the same style as the Fibonacci walkthroughs.

| # | Example | What to feed it | Transform (and inverse) | What it teaches |
|---|---|---|---|---|
| 1 | **Scale invariance** | the same sine wave three times: as-is, ×1000, and +1,000,000 | none | Forecasts should match exactly after rescaling. The model sees shape, never units. |
| 2 | **A rule with no shape: primes** | the first 50 primes: 2, 3, 5, 7, 11, … | none; then try gaps between primes | It will extend the rough upward trend (~n log n) but can't produce the next *prime*. An LLM might recall it. A pattern model can't know about divisibility. |
| 3 | **Pure noise: digits of π** | 3, 1, 4, 1, 5, 9, 2, 6, … | none | The median should sit near the average digit (~4.5) with a wide interval. With no pattern, the best forecast is "the average, uncertain". |
| 4 | **Polynomial growth: squares n²** | 0, 1, 4, 9, 16, … | first differences (2n+1, a straight line), then a cumulative sum back from the last value | Differencing turns curves into lines. Like the ratio transform, the inverse needs the last real value. |
| 5 | **Multiplicative seasonality** | monthly sales that grow 5%/year with ±20% December peaks | log (makes the seasonality additive), then exp | Why log is the standard first move for business data: it turns "percent swings that grow with the level" into fixed-size swings. |
| 6 | **Bounded values: percentages** | a conversion rate drifting between 2% and 8% | logit, log(p / (1−p)), then the logistic function | Raw forecasts can leave 0–100%. Mapping to an unbounded scale and back keeps the quantiles in range. |
| 7 | **Counts with zeros** | daily sales of a slow-moving product: 0, 0, 3, 0, 1, 0, 0, 5, … | log1p, i.e. log(1 + x), then expm1; or clip at 0 | Raw quantiles can go negative. log1p handles zeros where log can't. |
| 8 | **Context length vs period** | a sine with period 200, given 64 vs 512 steps of history | none | With less than one full period the model can't see the cycle and extrapolates the local slope. It can only continue what it can see. |
| 9 | **Random walk** | a cumulative sum of coin flips | none | The median should stay near the last value, with the interval widening over the horizon (roughly like √h). "I don't know which way" is expressed as width, not as a guess. |
| 10 | **Leading indicator (3.0 only)** | two series as variates: B is A shifted 5 steps later | none; pass as `[1, 2, T]` | Variate attention lets B's forecast use A's recent moves. Compare against forecasting B alone. |
| 11 | **Regime change** | a flat series that steps up once, 10 steps before the end | none | The model weighs recent data but has no idea *why* the level moved. Useful for seeing how fast it adapts. |

Good first picks: **1** (scale invariance) and **3** (π digits), because they show most directly what "not an LLM" means. **4** (differencing) extends the Fibonacci transforms with the other most common trick.

## Reproduce

```bash
xcodebuild test -scheme mlxtoto-Package -destination 'platform=macOS' \
  -only-testing:MLXTimeSeriesTests/TimesFMFibonacciTests \
  -only-testing:MLXTimeSeriesTests/TimesFM3FibonacciTests
```

The TimesFM 3.0 tests need `converted/timesfm3-fp32` and/or `converted/timesfm3-fp16`:

```bash
python Scripts/convert_ts_model.py --hf-path google/timesfm-3.0-pytorch --mlx-path converted/timesfm3-fp16
python Scripts/convert_ts_model.py --hf-path google/timesfm-3.0-pytorch --mlx-path converted/timesfm3-fp32 --dtype float32
```
