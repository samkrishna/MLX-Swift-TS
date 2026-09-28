import Foundation
import MLX
import Testing

@testable import MLXTimeSeries

// MARK: - Building intuition: eleven small experiments with TimesFM
//
// The Fibonacci walkthroughs (TimesFMFibonacciTests, TimesFM3FibonacciTests) introduce the core
// idea: TimesFM is a *pattern matcher*, not a calculator, so you often transform your data into a
// familiar shape, forecast, and transform back. Read those first.
//
// This file is a set of short experiments, each isolating ONE idea about how a time series
// foundation model behaves, especially where it differs from an LLM. Every experiment runs on
// all four checkpoints (see TimesFMVariants.swift):
//
//     TimesFM 2.5 fp16   TimesFM 2.5 fp32   TimesFM 3.0 fp32   TimesFM 3.0 fp16
//
// All four compute in float32; fp16 vs fp32 is only how the weights are *stored*. Within a
// version, the fp16 file is exactly the fp32 one rounded, so comparing them isolates precision.
// Comparing 2.5 with 3.0 shows what the architecture changed. (Spoiler for precision: fp16 and
// fp32 usually agree to about 0.1% here. The biggest gap is ~1.6%, on experiment 8's
// short-history forecast, the least certain one; uncertain forecasts are where rounding shows,
// as with raw Fibonacci. So the "Observed" comments quote one number per version.)
//
// ─────────────────────────────────────────────────────────────────────────────────────────
// The experiments
// ─────────────────────────────────────────────────────────────────────────────────────────
//
//   #   Experiment                 Idea
//   ──  ─────────────────────────  ─────────────────────────────────────────────────────────
//   1   Scale invariance           It sees shape, never units: ×1000 or +10,000 changes nothing.
//   2   Primes                     A rule with no shape can't be learned: it forecasts ~n log n.
//   3   Digits of π                Pure noise: best it can do is "the average, uncertain".
//   4   Squares n²                 Differencing turns a curve into a line (and 3.0 nails it).
//   5   Multiplicative seasonality log turns growing seasonal swings into fixed ones.
//   6   Percentages                logit keeps forecasts inside 0–100%, at some cost.
//   7   Counts with zeros          The median of "mostly zero" is zero; negatives need clipping.
//   8   Context length vs period   It can only continue a cycle it can see.
//   9   Random walk                "Don't know" is expressed as a widening interval.
//   10  Leading indicator          3.0 mixes related series, but mixing isn't automatically better.
//   11  Regime change              The fresher a change, the more it hedges back to the old level.
//
// Conventions used below:
//   - `forecast(...)` (TimesFMWalkthroughSupport.swift) returns the median (50% quantile) and the
//     10% / 90% quantiles for every future step. `low...high` is the 80% prediction interval.
//   - Errors are measured against the *known* continuation of a synthetic series, so we can say
//     exactly how right or wrong the model is.
//   - "Random" data comes from a seeded generator, so every run and every model sees identical
//     input.

/// The first 300 decimal digits of π, starting "3141592653...".
let piDigits = Array(
    "314159265358979323846264338327950288419716939937510582097494459230781640628620899862803482534211706798214808651328230664709384460955058223172535940812848111745028410270193852110555964462294895493038196442881097566593344612847564823378678316527120190914564856692346034861045432664821339360726024914127"
).map { Float(String($0))! }

@Suite(
    "TimesFM intuition experiments",
    // Needs at least one real checkpoint; random weights would forecast noise.
    .enabled(if: !TimesFMVariant.available.isEmpty),
    // One model in memory at a time (up to 1.3 GB each in float32).
    .serialized)
struct TimesFMIntuitionTests {

    // MARK: 1. Scale invariance

    /// A smooth two-cycle wave: period 24 (amplitude 10) plus period 7 (amplitude 0.5).
    static func wave(_ t: Int) -> Float {
        let x = Double(t)
        let slow: Double = 10 * sin(2 * Double.pi * x / 24)
        let fast: Double = 0.5 * sin(2 * Double.pi * x / 7)
        return Float(slow + fast)
    }

    @Test("1. Scale invariance: ×1000 or +10,000 gives the same forecast, rescaled", arguments: TimesFMVariant.available)
    func testScaleInvariance(model: TimesFMVariant) throws {
        let forecaster = try model.load()

        // ── The idea. An LLM reads "$2 million" and "2 degrees" very differently; the units are
        // part of the meaning. TimesFM never sees units. Before anything else, it rescales each
        // 32-step patch by its running mean and standard deviation, forecasts in that unitless
        // space, and scales the answer back. So if we stretch or shift the input, the forecast
        // should stretch or shift by exactly the same amount.
        let x = (0 ..< 256).map(Self.wave)

        // Three versions of the same series: as-is, 1000 times bigger, and shifted up by 10,000.
        let original = forecast(forecaster, x, horizon: 48)
        let scaled = forecast(forecaster, x.map { $0 * 1000 }, horizon: 48)
        let shifted = forecast(forecaster, x.map { $0 + 10_000 }, horizon: 48)

        // ── Undo the stretch/shift on the forecasts and compare every quantile of every step.
        var scaleGap: Float = 0
        var shiftGap: Float = 0
        for h in 0 ..< 48 {
            for q in 0 ..< 9 {
                let base = original.quantiles[0][h][q]
                scaleGap = max(scaleGap, abs(scaled.quantiles[0][h][q] / 1000 - base))
                shiftGap = max(shiftGap, abs(shifted.quantiles[0][h][q] - 10_000 - base))
            }
        }
        // Measure the gaps relative to the series' spread (its std is about 7.04).
        let spread = standardDeviation(x)
        let truth = (256 ..< 304).map(Self.wave)
        print("[Intuition 1 scale \(model.name)] ×1000 gap/std \(scaleGap / spread), +10,000 gap/std \(shiftGap / spread), MAE vs truth \(meanAbsError(original.median(), truth))")

        // Observed: ×1000 differs by about 1e-6 of the spread, which is float32 rounding.
        // +10,000 differs by about 1e-4: near 10,000 a float32 can only represent steps of about
        // 0.001, so the shifted *input* itself already lost that much detail.
        #expect(scaleGap / spread < 1e-4)
        #expect(shiftGap / spread < 1e-3)

        // Bonus: both versions forecast this wave well, 3.0 more so (MAE ≈ 0.07 vs ≈ 0.17
        // for 2.5, on a wave that swings ±10).
        #expect(meanAbsError(original.median(), truth) < 0.3)
    }

    // MARK: 2. Primes

    /// The first `count` primes, by trial division (fine for a few dozen).
    static func primes(_ count: Int) -> [Int] {
        var found: [Int] = []
        var n = 2
        while found.count < count {
            if found.allSatisfy({ n % $0 != 0 }) { found.append(n) }
            n += 1
        }
        return found
    }

    @Test("2. Primes: a rule with no shape; it forecasts the trend, not the next prime", arguments: TimesFMVariant.available)
    func testPrimes(model: TimesFMVariant) throws {
        let forecaster = try model.load()

        // ── The idea. "What comes after 229 in the primes?" An LLM has probably memorized 233.
        // TimesFM can't: it has never been told what a prime is, and primality has no *shape*.
        // What it can see is that the sequence climbs a little faster than a straight line
        // (the n-th prime is roughly n·log n).
        let all = Self.primes(56)
        let history = all.prefix(50).map { Float($0) }  // 2, 3, 5, ..., 229
        let truth = all[50 ..< 55].map { Float($0) }  // 233, 239, 241, 251, 257

        // ── Forecast the next five primes directly.
        let next = forecast(forecaster, history, horizon: 5)
        print("[Intuition 2 primes \(model.name)] median \(next.median()), truth \(truth)")

        // Observed:
        //   3.0: 232.7, 236.6, 240.6, 244.6, 248.7. A steady +4 per step: TimesFM 3.0's linear
        //        detrending sees an almost straight climb and extends it. The true steps are
        //        +4, +6, +2, +10, +6. The first value rounds to 233 by luck, not knowledge.
        //   2.5: 242.3, 242.1, 246.9, 253.6, 259.2. Jumps 13 above the last prime and wobbles.
        // Either way, the forecast is off by several units somewhere. It's a trend, not a rule.
        let errors = zip(next.median(), truth).map { abs($0 - $1) }
        #expect(errors.max()! > 5)

        // ── Transform: gaps between consecutive primes. This is how you'd make "the next prime"
        // a forecasting problem: forecast the next gap, then add it to the last prime.
        // The 49 gaps between the 50 primes we gave it, ending with 229 − 227 = 2.
        let gaps = (1 ..< 50).map { Float(all[$0] - all[$0 - 1]) }  // 1, 2, 2, 4, 2, 4, ...
        let trueGaps = (50 ..< 55).map { Float(all[$0] - all[$0 - 1]) }  // 4, 6, 2, 10, 6
        let averageGap = gaps.reduce(0, +) / Float(gaps.count)
        let gapForecast = forecast(forecaster, gaps, horizon: 5)
        print("[Intuition 2 prime gaps \(model.name)] median \(gapForecast.median()), 10–90% \(gapForecast.low()[0])…\(gapForecast.high()[0]), truth \(trueGaps), average gap \(averageGap)")

        // Observed: every model forecasts a gap of about 3.7–4.1 at every step (the average gap
        // so far is 4.63), with an 80% interval of roughly 1.5 to 8.5. The true gaps are 4, 6, 2,
        // 10, 6. Prime gaps look like noise around an average, so "about the average, give or
        // take" is all it can say. Compare experiment 3 (π digits).
        for gap in gapForecast.median() {
            #expect(gap > 3 && gap < 5.5)
        }
    }

    // MARK: 3. Digits of π

    @Test("3. Digits of π: pure noise, so the forecast is 'the average, uncertain'", arguments: TimesFMVariant.available)
    func testPiDigits(model: TimesFMVariant) throws {
        let forecaster = try model.load()

        // ── The idea. The digits of π are completely determined by a formula, yet as a sequence
        // they behave like random noise: no trend, no cycle, no memory. There's no shape to
        // continue, so the best a pattern model can do is forecast the average and admit a
        // wide range. (An LLM might simply *recite* the next digits from memory.)
        let history = Array(piDigits[0 ..< 256])  // 3, 1, 4, 1, 5, 9, ... (256 digits)
        let truth = Array(piDigits[256 ..< 288])  // the next 32 digits
        let average = history.reduce(0, +) / Float(history.count)  // 4.53

        let next = forecast(forecaster, history, horizon: 32)

        // A "forecast" that knows nothing: always predict the historical average.
        let naiveError = meanAbsError([Float](repeating: average, count: 32), truth)
        let modelError = meanAbsError(next.median(), truth)
        let meanLow = next.low().reduce(0, +) / 32
        let meanHigh = next.high().reduce(0, +) / 32
        print("[Intuition 3 π \(model.name)] medians \(next.median().min()!)…\(next.median().max()!), mean 80% interval \(meanLow)…\(meanHigh), MAE \(modelError) vs always-average \(naiveError)")

        // Observed:
        //   3.0: medians 4.42–4.56 (hugging the average, 4.53); 80% interval ≈ 0.6 to 8.5.
        //   2.5: medians 3.95–4.61; 80% interval ≈ 0.7 to 8.3.
        //   Mean absolute error ≈ 2.04 (2.5) and 2.10 (3.0), vs 2.10 for "always 4.53".
        // So the model is no better than guessing the average, which is the right answer for
        // noise. The honest part is the interval: it covers nearly every digit, 0 to 9.
        for m in next.median() {
            #expect(m > 3.5 && m < 5.5)
        }
        #expect(meanLow < 1.5 && meanHigh > 7.5)
        #expect(modelError > 0.9 * naiveError)
    }

    // MARK: 4. Squares and differencing

    @Test("4. Squares n²: differencing turns a curve into a line", arguments: TimesFMVariant.available)
    func testSquaresDifferencing(model: TimesFMVariant) throws {
        let forecaster = try model.load()

        // ── The idea. n² curves upward. TimesFM handles it tolerably, but a *straight line* is
        // much easier. Differencing is the other classic transform (besides log and ratios):
        //     d(n) = n² − (n−1)² = 2n − 1,
        // so the differences 1, 3, 5, 7, ... are a perfect straight line.
        // Its inverse is a running sum starting from the last real value, just as the Fibonacci
        // ratio transform needed F(31):
        //     next square = last square + next difference.
        let squares = (0 ..< 64).map { Float($0 * $0) }  // 0, 1, 4, ..., 3969
        let truth = (64 ..< 72).map { Float($0 * $0) }  // 4096, 4225, ..., 5041

        // Worst error relative to the true value, across the 8 forecast steps.
        func worstRelativeError(_ p: [Float]) -> Float {
            zip(p, truth).map { abs($0 - $1) / $1 }.max()!
        }

        // ── 1) Raw: forecast the squares directly.
        let raw = forecast(forecaster, squares, horizon: 8).median()

        // ── 2) Differenced: forecast 2n − 1, then add back up from 63² = 3969.
        let differences = (1 ..< 64).map { (n: Int) -> Float in Float(2 * n - 1) }  // 1, 3, ..., 125
        let nextDifferences = forecast(forecaster, differences, horizon: 8).median()
        var running = squares.last!
        let rebuilt = nextDifferences.map { d -> Float in
            running += d
            return running
        }
        print("[Intuition 4 squares \(model.name)] raw \(raw) (worst \(worstRelativeError(raw))), next differences \(nextDifferences), rebuilt \(rebuilt) (worst \(worstRelativeError(rebuilt)))")

        // Observed:
        //   raw:          worst error 2.9% (2.5), 0.29% (3.0)
        //   differenced:  worst error 0.02% (2.5), exactly 0 (3.0)
        // 3.0 returns the differences 127, 129, 131, ..., 141 *exactly*: its linear-detrending
        // step fits the perfect line 2n − 1, removes it, is left with nothing to forecast, and
        // adds the line back. Here our transform and the model's preprocessing stack perfectly.
        #expect(worstRelativeError(rebuilt) < 0.001)
        #expect(worstRelativeError(rebuilt) < worstRelativeError(raw))
    }

    // MARK: 5. Multiplicative seasonality

    /// Monthly sales: 20% yearly growth times a fixed seasonal pattern that peaks in December.
    /// Because the pattern *multiplies* the level, the December spike grows every year.
    static func sales(_ month: Int) -> Float {
        let seasonal: [Double] = [0.9, 0.85, 0.95, 1.0, 1.0, 1.05, 1.05, 1.0, 0.95, 1.0, 1.1, 1.4]
        let level: Double = 100 * pow(1.20, Double(month) / 12)
        return Float(level * seasonal[month % 12])
    }

    @Test("5. Multiplicative seasonality: log turns growing swings into fixed ones", arguments: TimesFMVariant.available)
    func testMultiplicativeSeasonality(model: TimesFMVariant) throws {
        let forecaster = try model.load()

        // ── The idea. In real business data, seasonal swings are usually a *percentage* of the
        // level: December is "+40%", not "+40 units". As the business grows, the swings grow
        // too. Taking log turns multiplication into addition,
        //     log(level × season) = log(level) + log(season),
        // so the growth becomes a straight line and the seasonal swing becomes a fixed-size
        // wiggle on top. Both are shapes TimesFM handles well. That's why log is the standard
        // first move for sales-like data.
        let history = (0 ..< 120).map(Self.sales)  // 10 years of months
        let truth = (120 ..< 144).map(Self.sales)  // the next 2 years

        // Mean absolute percentage error over the 24 forecast months.
        func mape(_ p: [Float]) -> Float {
            zip(p, truth).map { abs($0 - $1) / $1 }.reduce(0, +) / 24 * 100
        }

        // ── Raw vs log (then exp back; exp is increasing, so the median stays the median).
        let raw = forecast(forecaster, history, horizon: 24).median()
        let viaLog = forecast(forecaster, history.map { log($0) }, horizon: 24).median().map { exp($0) }
        print("[Intuition 5 seasonality \(model.name)] raw MAPE \(mape(raw))%, log MAPE \(mape(viaLog))%; December truth \(truth[11]), raw \(raw[11]), log \(viaLog[11])")

        // Observed:
        //   2.5: raw 4.0% average error, log 0.99%
        //   3.0: raw 2.1%,                log 0.24%
        // Raw forecasts under-shoot the growing December spike (truth ≈ 1,025; raw ≈ 991 for
        // 2.5 and ≈ 981 for 3.0), because the model continues the spike at its *past* size.
        // In log space the spike is the same size every year, so it's easy to continue.
        #expect(mape(viaLog) < mape(raw) / 2)
    }

    // MARK: 6. Percentages

    @Test("6. Percentages: logit keeps forecasts inside 0–100%, at some cost", arguments: TimesFMVariant.available)
    func testPercentagesLogit(model: TimesFMVariant) throws {
        let forecaster = try model.load()

        // ── The idea. A conversion rate lives between 0 and 1. TimesFM doesn't know that; it only
        // knows numbers. When the rate hovers near 0, its lower quantiles can dip below 0, an
        // impossible value. The logit transform,
        //     z = log(p / (1 − p))        (inverse: p = 1 / (1 + e^(−z))),
        // maps (0, 1) onto the whole number line, so any forecast in z-space maps back to a
        // valid rate. The logistic inverse is increasing, so the quantiles stay quantiles.
        var random = SeededRandom(seed: 6)
        var rates = [Float]()
        for t in 0 ..< 230 {
            // A rate cycling between about 0.5% and 3.5% every 30 days, plus noise, floored at 0.1%.
            let signal: Double = 0.02 + 0.015 * sin(2 * Double.pi * Double(t) / 30)
            rates.append(Float(max(0.001, signal + 0.006 * random.normal())))
        }
        let history = Array(rates[0 ..< 200])
        // The noise-free signal for the next 30 days: what a perfect forecast would track.
        let signal = (200 ..< 230).map { (t: Int) -> Float in
            Float(0.02 + 0.015 * sin(2 * Double.pi * Double(t) / 30))
        }
        func logistic(_ z: Float) -> Float { 1 / (1 + exp(-z)) }

        // ── Raw vs logit.
        let raw = forecast(forecaster, history, horizon: 30)
        let viaLogit = forecast(forecaster, history.map { log($0 / (1 - $0)) }, horizon: 30)
        let logitLow = viaLogit.low().map(logistic)
        let rawError = meanAbsError(raw.median(), signal)
        let logitError = meanAbsError(viaLogit.median().map(logistic), signal)
        print("[Intuition 6 percentages \(model.name)] raw lowest 10% quantile \(raw.low().min()!), logit lowest \(logitLow.min()!); error vs signal raw \(rawError), logit \(logitError)")

        // Observed: raw 10% quantiles go negative on some days (lowest ≈ −0.06% for 2.5 and
        // −0.08% for 3.0). Through logit, every quantile stays above 0 (lowest ≈ 0.10–0.11%).
        #expect(raw.low().min()! < 0)
        #expect(logitLow.allSatisfy { $0 > 0 && $0 < 1 })

        // The catch: the median doesn't necessarily get better. 2.5 is slightly more accurate
        // through logit (error 0.077 vs 0.082 percentage points), but 3.0 is *less* accurate
        // (0.142 vs 0.087). The transform stretches the region near 0, which changes what the
        // model sees. Transforms fix one thing and can cost another, so measure both.
        print("[Intuition 6 percentages \(model.name)] logit/raw error ratio \(logitError / rawError)")
    }

    // MARK: 7. Counts with zeros

    @Test("7. Counts with zeros: the median is 0, and negatives need clipping", arguments: TimesFMVariant.available)
    func testIntermittentCounts(model: TimesFMVariant) throws {
        let forecaster = try model.load()

        // ── The idea. A slow-moving product: most days nobody buys it (0), some days 1–5 units.
        // Two surprises for newcomers:
        //   (a) The *median* forecast is ~0, even though average demand is ~0.85/day. On a
        //       typical day nothing sells, so "0" really is the middle outcome. For stocking
        //       shelves you want the mean or an upper quantile, not the median.
        //   (b) The lower quantiles go below 0, which is impossible for a count.
        var random = SeededRandom(seed: 7)
        let sales = (0 ..< 256).map { _ -> Float in
            // 30% chance of a sale on any day; if so, 1 to 5 units.
            random.uniform() < 0.3 ? Float(1 + Int(random.uniform() * 5)) : 0
        }
        let averageDemand = sales.reduce(0, +) / Float(sales.count)  // 0.85

        let raw = forecast(forecaster, sales, horizon: 28)
        let medianAverage = raw.median().reduce(0, +) / 28
        let highAverage = raw.high().reduce(0, +) / 28
        print("[Intuition 7 counts \(model.name)] average demand \(averageDemand), average median forecast \(medianAverage), average 90% quantile \(highAverage), lowest 10% quantile \(raw.low().min()!)")

        // Observed: the median averages about 0.04–0.05 units/day, against real average demand
        // of 0.85. The 90% quantile averages about 3.5: "on a busy day, expect up to ~3–4".
        #expect(medianAverage < 0.2)
        #expect(highAverage > 2)
        // And the 10% quantile goes negative (lowest ≈ −0.09 for 2.5, −0.04 for 3.0).
        #expect(raw.low().min()! < 0)

        // ── A common suggestion is log(1 + x), which handles 0 (log 0 doesn't exist). Its inverse
        // e^z − 1 can still go slightly below 0, down to −1. It shrinks the problem but doesn't
        // remove it. Here the lowest value comes back at about −0.04 (2.5) and −0.02 (3.0).
        let viaLog1p = forecast(forecaster, sales.map { log1p($0) }, horizon: 28)
        let log1pLow = viaLog1p.low().map { expm1($0) }
        print("[Intuition 7 counts \(model.name)] log1p lowest 10% quantile \(log1pLow.min()!)")
        #expect(log1pLow.min()! > -1)

        // The reliable fix for counts is simply to clip at 0 after forecasting.
        let clipped = raw.low().map { max($0, 0) }
        #expect(clipped.allSatisfy { $0 >= 0 })
    }

    // MARK: 8. Context length vs period

    @Test("8. Context length vs period: it can only continue a cycle it can see", arguments: TimesFMVariant.available)
    func testContextVersusPeriod(model: TimesFMVariant) throws {
        let forecaster = try model.load()

        // ── The idea. A slow cycle with a period of 200 steps. Give the model the last 512 steps
        // (2.5 full cycles) or only the last 64 (a third of one cycle). An LLM can be *told*
        // "this repeats every 200 steps". TimesFM has to *see* it. With 64 steps it sees a
        // curved segment, not a cycle, so it can only extend the local curve.
        func cycle(_ t: Int) -> Float { Float(sin(2 * Double.pi * Double(t) / 200)) }
        let full = (0 ..< 512).map(cycle)
        let truth = (512 ..< 612).map(cycle)  // the next 100 steps

        let longView = forecast(forecaster, full, horizon: 100).median()
        let shortView = forecast(forecaster, Array(full.suffix(64)), horizon: 100).median()
        let longError = meanAbsError(longView, truth)
        let shortError = meanAbsError(shortView, truth)
        print("[Intuition 8 context \(model.name)] 512-step history MAE \(longError), 64-step MAE \(shortError); step 100: truth \(truth[99]), long \(longView[99]), short \(shortView[99])")

        // Observed:
        //   512 steps: MAE 0.014 (2.5), 0.004 (3.0). Essentially perfect.
        //   64 steps:  MAE 0.10  (2.5), 0.14  (3.0). 7 to 36 times worse.
        // At step 100 the truth is +0.34 and on its way up. From 64 steps, 2.5 says −0.03
        // (it hasn't turned yet) and 3.0 says +0.64 (it overshoots). Both are guessing where
        // the curve goes next. Rule of thumb: give at least one or two full periods of history.
        #expect(shortError > 5 * longError)
    }

    // MARK: 9. Random walk

    @Test("9. Random walk: uncertainty shows up as a widening interval", arguments: TimesFMVariant.available)
    func testRandomWalk(model: TimesFMVariant) throws {
        let forecaster = try model.load()

        // ── The idea. A random walk moves up or down by 1 each step, at random. Its future is
        // unpredictable, but *how* unpredictable is known exactly: after h steps the position
        // is spread out like a bell curve with standard deviation √h. So the ideal forecast is:
        //   - median: stay at the last value (up and down are equally likely);
        //   - 80% interval: last value ± 1.28·√h, a width of 2.56·√h. That's about 2.6 at
        //     h = 1, 14.5 at h = 32, and 29.0 at h = 128.
        var random = SeededRandom(seed: 9)
        var position: Float = 0
        let walk = (0 ..< 512).map { _ -> Float in
            position += random.uniform() < 0.5 ? -1 : 1
            return position
        }
        // (With this seed the walk happens to end back at exactly 0 after 512 steps.)

        let next = forecast(forecaster, walk, horizon: 128)
        let width = zip(next.high(), next.low()).map { $0 - $1 }
        func ideal(_ h: Int) -> Float { 2.563 * Float(h).squareRoot() }
        print("[Intuition 9 random walk \(model.name)] last \(walk.last!); median at h=1,32,128: \(next.median()[0]), \(next.median()[31]), \(next.median()[127]); 80% width \(width[0]), \(width[31]), \(width[127]) vs ideal \(ideal(1)), \(ideal(32)), \(ideal(128))")

        // Observed medians: 3.0 stays within 0.25 of the last value throughout; 2.5 drifts up to
        // +2.1 by step 128. Both are small next to the interval width, so both effectively say
        // "no idea which way".
        for h in [0, 31, 127] {
            #expect(abs(next.median()[h] - walk.last!) < 0.25 * width[h])
        }

        // Observed widths (h = 1, 32, 128):
        //   ideal:  2.6, 14.5, 29.0
        //   3.0:    2.8, 14.6, 29.9   ← almost exactly the √h law
        //   2.5:    2.7, 10.7, 18.1   ← too narrow for long horizons
        // 128 steps fit in a single forward pass for both versions (2.5 outputs 128 steps per
        // pass), so this isn't 2.5's feed-back loop at work. 2.5's quantiles simply spread
        // too little at long range on this input: it is overconfident about how far a random
        // walk can wander. 3.0 matches the √h law almost exactly. If you plan with 2.5's
        // intervals far ahead, expect reality to leave them more often than 20% of the time.
        #expect(width[127] > 3 * width[0])
        switch model.version {
        case .v3:
            #expect(abs(width[127] / ideal(128) - 1) < 0.2)
        case .v25:
            #expect(width[127] < 0.75 * ideal(128))
        }
    }

    // MARK: 10. Leading indicator

    /// A smooth, irregular signal: three sines with unrelated periods, plus optional noise.
    static func leader(_ t: Int, noise: Double) -> Float {
        let x = Double(t)
        let a: Double = sin(2 * Double.pi * x / 17.3)
        let b: Double = 0.7 * sin(2 * Double.pi * x / 29.1)
        let c: Double = 0.5 * sin(2 * Double.pi * x / 43.7)
        return Float(a + b + c + noise)
    }

    @Test("10. Leading indicator: 3.0 mixes related series, but not automatically for the better", arguments: TimesFMVariant.available)
    func testLeadingIndicator(model: TimesFMVariant) throws {
        let forecaster = try model.load()

        // ── The idea. Series B copies series A five steps later: B(t) = A(t − 5). So B's next
        // five values are already sitting in A's last five. A person would spot that at once.
        // Can TimesFM use it if we hand it both series together?
        //   - TimesFM 2.5 treats every series separately. Passing A alongside B changes
        //     nothing, so it's a control.
        //   - TimesFM 3.0 has *variate attention*: series passed together as [1, 2, T] can
        //     exchange information at every layer.
        var random = SeededRandom(seed: 10)
        var a = [Float]()
        for t in 0 ..< 266 { a.append(Self.leader(t, noise: 0.2 * random.normal())) }
        let b = (0 ..< 266).map { $0 >= 5 ? a[$0 - 5] : 0 }
        let T = 256
        let truth = Array(b[T ..< T + 5])  // = a[251..<256], already in A's history

        // ── B alone vs A and B together (A as variate 0, B as variate 1).
        let alone = forecast(forecaster, Array(b[0 ..< T]), horizon: 5).median()
        let together = forecast(forecaster, [Array(a[0 ..< T]), Array(b[0 ..< T])], horizon: 5).median(1)
        let aloneError = meanAbsError(alone, truth)
        let togetherError = meanAbsError(together, truth)
        let change = zip(alone, together).map { abs($0 - $1) }.max()!
        print("[Intuition 10 leading \(model.name)] B alone MAE \(aloneError), with A MAE \(togetherError), largest change \(change)")

        switch model.version {
        case .v25:
            // Observed: identical to the last digit. 2.5 never looks across series.
            #expect(change == 0)
        case .v3:
            // Observed: the forecast *does* change (by up to ≈ 0.06), so 3.0 is mixing the two
            // series, but it gets slightly *worse* (MAE 0.183 with A vs 0.166 alone). The
            // pretrained model hasn't learned "copy the other series with a lag", and it can't
            // be told. Mixing related series is a capability, not a guarantee. Test it on your
            // data before relying on it.
            #expect(change > 1e-3)
            print("[Intuition 10 leading \(model.name)] joint better? \(togetherError < aloneError)")
        }
    }

    // MARK: 11. Regime change

    @Test("11. Regime change: the fresher the jump, the more it hedges back", arguments: TimesFMVariant.available)
    func testRegimeChange(model: TimesFMVariant) throws {
        let forecaster = try model.load()

        // ── The idea. A series sits at 10 for a long time, then jumps to 20. Is the jump
        // permanent or a blip? The model can't know *why* it happened (a price change? a
        // sensor glitch?), so it has to judge from the data alone: how long has the new level
        // lasted? We try the jump 3, 10 and 40 steps before the end of a 256-step history.
        var random = SeededRandom(seed: 11)
        var results = [(since: Int, next: Float, later: Float)]()
        for since in [3, 10, 40] {
            let series = (0 ..< 256).map { t -> Float in
                (t >= 256 - since ? 20 : 10) + Float(0.3 * random.normal())
            }
            let f = forecast(forecaster, series, horizon: 20)
            results.append((since, f.median()[0], f.median()[19]))
            print("[Intuition 11 regime \(model.name)] jump \(since) steps ago: median next \(f.median()[0]), in 20 steps \(f.median()[19]); 80% at 20 steps \(f.low()[19])…\(f.high()[19])")
        }

        // Observed median (next step → 20 steps ahead):
        //                     3 steps ago     10 steps ago    40 steps ago
        //   2.5              18.3 → 10.7     19.8 → 15.6     20.0 → 19.9
        //   3.0              19.2 → 15.2     20.0 → 19.6     19.8 → 19.6
        // A 3-step-old jump: both expect it to fade, 2.5 all the way back to 10. After 10 steps,
        // 3.0 already treats 20 as the new normal, while 2.5 still hedges halfway. After 40,
        // both believe it. The 80% intervals 20 steps out still reach down to ~10 in most cases:
        // "probably stays, but it could go back."
        let (fresh, _, old) = (results[0], results[1], results[2])
        #expect(fresh.later < fresh.next)  // a very fresh jump is expected to fade
        #expect(old.later > 19)  // an established level is expected to hold
        switch model.version {
        case .v3: #expect(results[1].later > 19)  // 3.0 adopts the new level within 10 steps
        case .v25: #expect(results[1].later < 17)  // 2.5 is still hedging after 10 steps
        }
    }
}
