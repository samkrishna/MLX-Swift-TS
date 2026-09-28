import Foundation
import MLX
import Testing

@testable import MLXTimeSeries

// MARK: - Forecasting the 33rd Fibonacci number with TimesFM 2.5
//
// Goal: give TimesFM the first 32 Fibonacci numbers,
//
//     F(0) ... F(31) = 0, 1, 1, 2, 3, 5, 8, ..., 832_040, 1_346_269
//
// and ask it for the 33rd, F(32) = 2_178_309.
//
// ─────────────────────────────────────────────────────────────────────────────────────────
// THE CORE IDEA: transform → forecast → transform back
// ─────────────────────────────────────────────────────────────────────────────────────────
//
// TimesFM is a *pattern matcher*, not a calculator. It was pretrained on huge numbers of
// real-world series (web traffic, sales, energy use, weather, ...). From those it learned what
// usually comes next after a given *shape*: trends continue, cycles repeat, spikes fade, flat
// lines stay flat. It never sees a formula. It can't work out that
// F(n) = F(n-1) + F(n-2), however many examples you show it.
//
// So the question to ask is: "does my data look like something TimesFM has seen a lot of?"
// If not, don't give it the raw data. Give it an equivalent series that does, and convert the
// answer back afterwards:
//
//        your data  ──transform──▶  familiar shape  ──TimesFM──▶  forecast (familiar shape)
//                                                                         │
//        answer    ◀─────────────────── inverse transform ─────────────────┘
//
// Requirements for a transform T to work here:
//
//   1. It has an inverse. You must be able to get back from T-space to real numbers.
//      log ↔ exp. Ratio r(n) = F(n)/F(n-1) ↔ F(n) = r(n) · F(n-1).
//
//   2. Its output is a shape TimesFM is good at. Roughly, from easiest to hardest:
//      a constant level  <  a straight-line trend  <  a repeating cycle  <  exponential growth.
//      Real data rarely grows exponentially for long, so the model hasn't learned to expect it.
//
//   3. The inverse may need information from the original data. The ratio inverse needs the
//      last real value, F(31), to turn "the next ratio" back into "the next number".
//
// Uncertainty comes along for free when T is *increasing* (bigger in → bigger out), like log,
// or multiplying by a positive number. Then the 10% quantile in T-space maps to the 10% quantile
// in real space, and the same holds for the median and the 90% quantile. That's one reason we
// read the *median* rather than the mean: medians survive the inverse transform, but means
// don't (exp(mean of log x) ≠ mean of x).
//
// Errors change shape too. An error of ε in log space becomes a *relative* error of about ε in
// real space, because exp(a + ε) = exp(a)·exp(ε) ≈ exp(a)·(1 + ε). An error ε in the ratio
// becomes a relative error of ε/ratio. A transform that makes the model more accurate is
// only worth it if the error stays small after you convert back.
//
// The three tests below apply this to Fibonacci:
//
//   Transform           What TimesFM sees                        Result for F(32)
//   ──────────────────  ───────────────────────────────────────  ────────────────────────
//   none (raw)          flat line, then a sudden spike at the end  1,389,242  (−36%)
//   log                 an almost straight line (slope ≈ 0.48)      2,299,482  (+5.6%)
//   ratio F(n)/F(n-1)   wobbles, then settles at 1.618…             2,181,586  (+0.15%)
//
// None of them gives exactly 2,178,309. TimesFM approximates.
//
// ─────────────────────────────────────────────────────────────────────────────────────────
// How TimesFM processes any input (the same for all three tests)
// ─────────────────────────────────────────────────────────────────────────────────────────
//
// - It splits the history into patches of 32 steps. If the length isn't a multiple of 32,
//   it pads at the front with "missing" markers that the model ignores.
// - Each patch is normalized: subtract the running mean, divide by the running std. So the
//   model sees only the *shape* of your data, never its units or scale. Values in the
//   millions and values around 1.6 look the same to it if they rise and fall the same way.
//   The forecast is then scaled back with the same mean and std. This built-in normalization is the model's own
//   transform → forecast → transform back. It can shift and stretch, but it can't bend a
//   curve into a line. That's why we sometimes need a transform of our own on top.
// - One forward pass predicts the next 128 steps. Asking for 1 step just keeps the first one.
// - For every step it returns a mean plus 9 quantiles (10%, 20%, ..., 90%). The 50% quantile,
//   the median, is the point forecast and is returned as `prediction.mean`.

@Suite(
    "TimesFM 2.5 Fibonacci walkthrough",
    // These tests need the real 200M-parameter checkpoint. Random weights would forecast
    // noise, so skip rather than fail when it isn't on this machine.
    .enabled(if: FileManager.default.fileExists(atPath: timesFM25Checkpoint.path)),
    // Run one at a time: each test loads its own copy of the model (~800 MB in float32).
    .serialized)
struct TimesFMFibonacciTests {

    /// F(32), the 33rd element when counting from F(0) = 0. This is the answer we're after.
    let expected33rd = 2_178_309

    /// The first `count` Fibonacci numbers, starting 0, 1, 1, 2, ...
    func fibonacci(_ count: Int) -> [Int] {
        // Seed with F(0) = 0 and F(1) = 1.
        var sequence = [0, 1]
        // Each new term is the sum of the two before it. This is exactly the rule TimesFM will
        // never be told and can't infer.
        while sequence.count < count {
            sequence.append(sequence[sequence.count - 1] + sequence[sequence.count - 2])
        }
        // prefix handles count < 2 too (not needed here, but harmless).
        return Array(sequence.prefix(count))
    }

    /// What TimesFM says about the single next step of a history.
    struct NextStep {
        /// Median (50% quantile): the point forecast. Half the model's probability is above
        /// this value and half below.
        let median: Float
        /// 10% quantile: the model thinks there's about a 10% chance the value is below this.
        let low: Float
        /// 90% quantile: about a 10% chance the value is above this. So `low...high` is the
        /// model's 80% prediction interval.
        let high: Float
    }

    /// Feeds `history` to TimesFM and returns its forecast for the next single step.
    /// Every test calls this. Only what goes in (and how the answer is converted back) differs.
    func forecastNext(_ forecaster: TimeSeriesForecaster, history: [Float]) -> NextStep {
        // Wrap the plain Swift array in an MLX tensor. MLX runs the model on the GPU.
        let series = MLXArray(history)

        // TimesFM expects [batch, variables, time]: several series at once, each possibly with
        // several variables. We have 1 batch × 1 variable × T steps. `.univariate` does that
        // reshape and marks every step as observed (no missing values).
        let input = TimeSeriesInput.univariate(series)

        // Run the model. predictionLength = how many future steps we want: 1, just the next
        // value. Internally this is one forward pass (it always computes 128 steps and trims).
        let prediction = forecaster.forecast(input: input, predictionLength: 1)

        // prediction.mean has shape [1, 1, 1] = [batch, variable, step]. That's our single
        // median value. Flatten it and read it out as a Swift Float.
        let median = prediction.mean.reshaped(1)[0].item(Float.self)

        // prediction.quantiles has shape [1, 1, 1, 9]: the 9 quantile levels
        // 10%, 20%, ..., 90% for that one step. Flatten to 9 values.
        let quantiles = prediction.quantiles!.reshaped(9)

        return NextStep(
            median: median,
            low: quantiles[0].item(Float.self),  // index 0 → 10% quantile
            high: quantiles[8].item(Float.self))  // index 8 → 90% quantile
    }

    // MARK: 1. Raw values (no transform)

    @Test("Raw values: TimesFM can't follow exponential growth")
    func testRawFibonacci() throws {
        // Load the pretrained model the same way ModelArena does (config.json + weights).
        let forecaster = try loadTimesFM25Forecaster()

        // ── Transform: none. Just convert Int → Float, since the model works in floats.
        // Float32 stores whole numbers exactly up to 2^24 = 16,777,216, so every value here
        // (and the answer, 2,178,309) converts without rounding.
        let history = fibonacci(32).map { Float($0) }

        // Sanity checks: 32 values ending at F(31). 32 values are exactly one patch, so there's
        // no padding. The model normalizes this one patch by its mean (~110,000) and std
        // (~282,000).
        #expect(history.count == 32)
        #expect(history.last == 1_346_269)  // F(31)

        // ── Forecast.
        // After normalization the model sees roughly: −0.39, −0.39, −0.39, ... (a long,
        // almost flat stretch), then 0.74, 1.43, 2.56, 4.38 in the last four steps. A flat
        // line with a sudden spike at the end. In real-world data, spikes like that usually level off or fall back.
        let next = forecastNext(forecaster, history: history)

        // ── Transform back: none needed. The model already undid its own normalization, so
        // `next.median` is in real units.
        print("[Fibonacci raw] F(32) median \(next.median), 10–90% [\(next.low), \(next.high)], truth \(expected33rd)")

        // Observed: median ≈ 1,389,000, only ~3% above F(31). The truth, 2,178,309, sits
        // just above even the 90% quantile (~2,165,000). The model is confident growth
        // slows down. Exponential growth is the one shape it has learned *not* to trust.
        #expect(next.median < 0.8 * Float(expected33rd))  // way short of the truth...
        #expect(next.median > Float(history.last!))  // ...though it does expect *some* rise
    }

    // MARK: 2. Log transform

    @Test("Log values: exponential growth becomes a straight line (~6% error)")
    func testLogFibonacci() throws {
        let forecaster = try loadTimesFM25Forecaster()

        // ── Transform: natural log.
        // Fibonacci grows like φ^n / √5, where φ = 1.618... Taking logs:
        //     log F(n) ≈ n · log φ − log √5 = 0.481·n − 0.805
        // an equation of a straight line with slope 0.481. Lines are among the easiest shapes
        // TimesFM knows.
        //
        // log(0) = −∞, which would break everything, so drop F(0) with `dropFirst()`. That
        // leaves 31 values, F(1)...F(31), whose logs run 0, 0, 0.69, 1.10, 1.61, ..., 14.11.
        // 31 isn't a multiple of 32, so the model pads one "missing" slot at the front.
        let history = fibonacci(32).dropFirst().map { Float(log(Double($0))) }

        // ── Forecast in log space. The model continues the line:
        // 14.11 plus about one more step of slope. It predicts about 14.65, while the true
        // next log value is log(2,178,309) = 14.59.
        let next = forecastNext(forecaster, history: history)

        // ── Transform back: exp undoes log. exp is increasing, so the 10%/50%/90% quantiles in
        // log space become the 10%/50%/90% quantiles in real space. We map all three.
        let median = exp(next.median)
        let low = exp(next.low)
        let high = exp(next.high)
        print("[Fibonacci log] F(32) median \(median), 10–90% [\(low), \(high)], truth \(expected33rd)")

        // Observed: ≈ 2,299,000, about 5.6% high. The model is off by only 0.054 in log space
        // (14.648 vs 14.594). exp turns that into a *multiplicative* error:
        // e^0.054 ≈ 1.056, i.e. +5.6%. Small log errors become percentage errors.
        let relativeError = abs(median - Float(expected33rd)) / Float(expected33rd)
        #expect(relativeError < 0.10)
    }

    // MARK: 3. Ratio transform

    @Test("Ratios F(n)/F(n-1): converge to φ, forecast within 1%")
    func testRatioFibonacci() throws {
        let forecaster = try loadTimesFM25Forecaster()

        // Keep the integers around: the inverse transform will need F(31).
        let fib = fibonacci(32)

        // ── Transform: ratio of each term to the one before, r(n) = F(n) / F(n-1).
        // F(0) = 0 can't be a denominator, so start at n = 2:
        //     r(2) = 1/1 = 1.0,  r(3) = 2/1 = 2.0,  r(4) = 3/2 = 1.5,  r(5) = 5/3 = 1.667,
        //     r(6) = 8/5 = 1.6,  ...  r(31) = 1,346,269 / 832,040 = 1.6180339...
        // This wobbles for a few steps and then settles at the golden ratio φ. A settled,
        // constant level is the easiest shape there is: the forecast is "more of the same".
        // n runs from 2 to 31, so that's 30 ratios. The model pads 2 missing slots at the front.
        let ratios = (2 ..< fib.count).map { Float(fib[$0]) / Float(fib[$0 - 1]) }

        // ── Forecast in ratio space: TimesFM predicts the next ratio, r(32) = F(32)/F(31).
        let next = forecastNext(forecaster, history: ratios)

        // The value the ratios converge to, for comparison: φ = (1 + √5) / 2.
        let golden = Float((1 + 5.0.squareRoot()) / 2)

        // ── Transform back: rearrange r(32) = F(32) / F(31) into F(32) = r(32) · F(31).
        // This inverse needs a real value from the original data, F(31) = 1,346,269. The ratio
        // alone only says "how much bigger"; F(31) says "bigger than what". (To forecast several
        // steps you'd chain it: F(33) = r(33) · F(32), and so on, so errors compound.)
        let estimate = next.median * Float(fib[31])
        // The 80% interval converts the same way. Multiplying by a positive number keeps order.
        let low = next.low * Float(fib[31])
        let high = next.high * Float(fib[31])
        print("[Fibonacci ratio] next ratio \(next.median) (φ = \(golden)), F(32) ≈ \(estimate), 10–90% [\(low), \(high)], truth \(expected33rd)")

        // Observed: ratio ≈ 1.6205 vs φ = 1.6180, an error of 0.0024. Relative to φ that's
        // 0.0024 / 1.618 ≈ 0.15%, which carries straight through the multiplication:
        // F(32) ≈ 2,181,600, about 0.15% high. Still not exactly 2,178,309. It's a
        // good approximation, not a calculation.
        #expect(abs(next.median - golden) < 0.01)
        let relativeError = abs(estimate - Float(expected33rd)) / Float(expected33rd)
        #expect(relativeError < 0.01)
    }
}
