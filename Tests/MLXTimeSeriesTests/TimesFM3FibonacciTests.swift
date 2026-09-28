import Foundation
import MLX
import Testing

@testable import MLXTimeSeries

// MARK: - Forecasting the 33rd Fibonacci number with TimesFM 3.0
//
// The same exercise as TimesFMFibonacciTests.swift, now with TimesFM 3.0, and run twice: once
// with Google's exact fp32 weights and once with the half-size fp16 copy. Read that file first
// for the core idea (transform → forecast → transform back). This file focuses on what changes
// with 3.0, and on what fp16 costs you.
//
// Goal: give TimesFM the first 32 Fibonacci numbers,
//
//     F(0) ... F(31) = 0, 1, 1, 2, 3, 5, 8, ..., 832_040, 1_346_269
//
// and ask it for the 33rd, F(32) = 2_178_309.
//
// ─────────────────────────────────────────────────────────────────────────────────────────
// Results side by side
// ─────────────────────────────────────────────────────────────────────────────────────────
//
//   Transform           TimesFM 2.5            TimesFM 3.0 fp32         TimesFM 3.0 fp16
//   ──────────────────  ─────────────────────  ───────────────────────  ───────────────────────
//   none (raw)          1,389,242  (−36%)      3,433,329  (+58%)        3,243,936  (+49%)
//   log                 2,299,482  (+5.6%)     2,182,360  (+0.19%)      2,182,363  (+0.19%)
//   ratio F(n)/F(n-1)   2,181,586  (+0.15%)    2,183,295  (+0.23%)      2,183,358  (+0.23%)
//
// Three things stand out:
//
//   1. Raw values are still hopeless, but in the other direction. TimesFM 2.5 confidently
//      predicted that growth would slow down. TimesFM 3.0 overshoots, and its 80% interval is
//      enormous (about −5 million to +9 million). It is effectively saying "I have no idea",
//      which is a more honest answer than 2.5's.
//
//   2. The log transform went from 5.6% error to 0.19%. That is TimesFM 3.0's new *linear
//      detrending* step at work (explained below). log F(n) is almost a straight line, so the
//      model removes that line exactly, forecasts only the small wiggle left over, then adds
//      the line back. The transform we chose and the model's own preprocessing now stack.
//
//   3. fp16 vs fp32 hardly matters for log and ratio (the answers agree to about 0.003%),
//      but moves the raw forecast by about 190,000. When the model is this unsure, tiny
//      changes in the weights move the median a lot. Rounding error is not a fixed percentage:
//      it gets amplified exactly where the model's answer is least trustworthy anyway.
//
// ─────────────────────────────────────────────────────────────────────────────────────────
// How TimesFM 3.0 processes any input (the same for all three tests)
// ─────────────────────────────────────────────────────────────────────────────────────────
//
// - It splits the history into patches of 32 steps, left-padding with "missing" markers when
//   the length isn't a multiple of 32. All three histories here fit in a single patch.
// - NEW IN 3.0, linear detrending: it fits a straight line through the history (least squares).
//   If removing that line shrinks the series' spread (standard deviation) to less than half of
//   what it was, the line is subtracted before the model sees the data and added back to the
//   forecast afterwards. Otherwise the data goes in unchanged. This is a built-in
//   "transform → forecast → transform back" for linear trends.
// - Each patch is normalized by the running mean and std of everything up to and including it,
//   so again the model sees only shape, never units or scale.
// - Each patch predicts the next 64 steps × 9 quantiles (10%, 20%, ..., 90%). There's no
//   separate mean channel any more; `prediction.mean` holds the 50% quantile, the median.
// - NEW IN 3.0, single pass: longer horizons aren't decoded step by step. Empty "future"
//   patches are appended and the whole forecast comes out of one forward pass, with the
//   overlapping 64-step predictions blended together. For a 1-step forecast like ours,
//   only the last real patch's prediction is used, so none of that machinery comes into play.
// - NEW IN 3.0, variate attention: several related series can be forecast together and share
//   information. We pass one series at a time, so it doesn't matter here.

@Suite(
    "TimesFM 3.0 Fibonacci walkthrough",
    // These tests need the real checkpoint(s) in converted/timesfm3-fp32 and/or
    // converted/timesfm3-fp16. Random weights would forecast noise, so skip rather than fail.
    .enabled(if: !TimesFM3Weights.available.isEmpty),
    // One at a time: each test case loads its own copy of the model (1.3 GB in float32).
    .serialized)
struct TimesFM3FibonacciTests {

    /// F(32), the 33rd element when counting from F(0) = 0. This is the answer we're after.
    let expected33rd = 2_178_309

    /// The first `count` Fibonacci numbers, starting 0, 1, 1, 2, ...
    func fibonacci(_ count: Int) -> [Int] {
        // Seed with F(0) = 0 and F(1) = 1, then each term is the sum of the two before it.
        // As with 2.5, this rule is never shown to the model. It only ever sees numbers.
        var sequence = [0, 1]
        while sequence.count < count {
            sequence.append(sequence[sequence.count - 1] + sequence[sequence.count - 2])
        }
        return Array(sequence.prefix(count))
    }

    /// What TimesFM says about the single next step of a history.
    struct NextStep {
        /// Median (50% quantile): the point forecast.
        let median: Float
        /// 10% quantile: about a 10% chance the true value is below this.
        let low: Float
        /// 90% quantile: about a 10% chance the true value is above this.
        /// `low...high` is the model's 80% prediction interval.
        let high: Float
    }

    /// Loads one of the converted checkpoints the way an app would: a directory holding
    /// `config.json` + `model.safetensors`. `config.json` says `"model_type": "timesfm3"`, so
    /// the registry builds a `TimesFM3Model`. The loader converts the weights to float32 either
    /// way (the model's `inferenceDtype`), so fp16 vs fp32 only changes the *stored* weights.
    func load(_ weights: TimesFM3Weights) throws -> TimeSeriesForecaster {
        try TimeSeriesForecaster.loadFromDirectory(weights.directory)
    }

    /// Feeds `history` to TimesFM and returns its forecast for the next single step.
    func forecastNext(_ forecaster: TimeSeriesForecaster, history: [Float]) -> NextStep {
        // [batch = 1, variables = 1, time = T], every step marked as observed.
        let input = TimeSeriesInput.univariate(MLXArray(history))

        // One forward pass, keeping just the first predicted step.
        let prediction = forecaster.forecast(input: input, predictionLength: 1)

        // prediction.quantiles is [1, 1, 1, 9]: batch, variable, step, then the 9 quantile
        // levels 10%...90%. TimesFM 3 sorts them, so index 0 is always the lowest.
        let quantiles = prediction.quantiles!.reshaped(9)

        return NextStep(
            median: prediction.mean.reshaped(1)[0].item(Float.self),  // = quantiles[4]
            low: quantiles[0].item(Float.self),  // 10%
            high: quantiles[8].item(Float.self))  // 90%
    }

    /// Asks the loaded model whether linear detrending kicks in for `history`, and the line it
    /// fits. This reaches into the model (`@testable`) purely to show what it does internally.
    func detrending(_ forecaster: TimeSeriesForecaster, history: [Float])
        -> (applied: Bool, lastValueOnLine: Float, slopePerStep: Float)
    {
        let model = forecaster.model as! TimesFM3Model
        // All three histories are ≤ 32 steps, so they are padded to exactly one 32-step patch,
        // with the padding marked as missing. Rebuild that padded patch the same way.
        let pad = (32 - history.count % 32) % 32
        let values = MLXArray([Float](repeating: 0, count: pad) + history).reshaped(1, 1, -1)
        let masked = MLXArray([Bool](repeating: true, count: pad)
            + [Bool](repeating: false, count: history.count)).reshaped(1, 1, -1)
        let trend = model.detrend(values, masked: masked, context: pad + history.count)
        // The model measures time in units of the context length, with t = 0 at the last step,
        // so the fitted line's value at the last step is the intercept, and its slope per step
        // is slope / context.
        return (
            trend.apply.item(Bool.self),
            trend.intercept.item(Float.self),
            trend.slope.item(Float.self) / Float(pad + history.count))
    }

    // MARK: 1. Raw values (no transform)

    @Test("Raw values: TimesFM 3 overshoots and knows it's guessing", arguments: TimesFM3Weights.available)
    func testRawFibonacci(weights: TimesFM3Weights) throws {
        let forecaster = try load(weights)

        // ── Transform: none. 32 values, exactly one patch, so no padding. Float32 holds every
        // one of these integers exactly (they're all below 2^24).
        let history = fibonacci(32).map { Float($0) }
        #expect(history.last == 1_346_269)  // F(31)

        // ── Detrending check: does a straight line explain this data? No. The best-fit line
        // is a poor match for a curve that sits near zero for 25 steps and then shoots up, so
        // removing it doesn't halve the spread, and the model gets the raw values.
        let trend = detrending(forecaster, history: history)
        #expect(!trend.applied)

        // ── Forecast. After normalization the model sees what 2.5 saw: a long, nearly flat
        // stretch and then a sudden spike. 2.5 bet on the spike fading. 3.0 extrapolates the
        // spike instead, and hedges enormously.
        let next = forecastNext(forecaster, history: history)
        print("[Fibonacci3 raw \(weights.name)] F(32) median \(next.median), 10–90% [\(next.low), \(next.high)], truth \(expected33rd)")

        // Observed:
        //   fp32: median ≈ 3,433,000 (+58%), 80% interval ≈ [−4,980,000, 9,374,000]
        //   fp16: median ≈ 3,244,000 (+49%), 80% interval ≈ [−5,591,000, 9,660,000]
        // The interval is about 14 million wide and dips below zero, for a quantity that has
        // never been negative. The truth is inside it, but only because the interval covers
        // almost everything. A forecast this vague is useless, and the model is telling us so.
        // The median is well off the truth, in the opposite direction from 2.5...
        #expect(next.median > 1.3 * Float(expected33rd))
        // ...and the 80% interval is wider than F(31) itself, many times over.
        #expect(next.high - next.low > 5 * Float(history.last!))
    }

    // MARK: 2. Log transform

    @Test("Log values: detrending turns this into a near-perfect line (~0.2% error)", arguments: TimesFM3Weights.available)
    func testLogFibonacci(weights: TimesFM3Weights) throws {
        let forecaster = try load(weights)

        // ── Transform: natural log. Fibonacci grows like φ^n / √5, so
        //     log F(n) ≈ 0.481·n − 0.805,
        // almost a straight line. Drop F(0) because log(0) = −∞. That leaves 31 values,
        // padded with one missing slot to make a 32-step patch.
        let history = fibonacci(32).dropFirst().map { Float(log(Double($0))) }

        // ── Detrending check: this time the line fits so well that removing it shrinks the
        // spread far below half, so detrending kicks in. The fitted line has slope ≈ 0.48 per
        // step (log φ = 0.481) and reaches ≈ 14.10 at the last step (log F(31) = 14.113).
        let trend = detrending(forecaster, history: history)
        #expect(trend.applied)
        #expect(abs(trend.slopePerStep - Float(log((1 + 5.0.squareRoot()) / 2))) < 0.01)

        // ── Forecast. What the network actually sees is log F(n) *minus* that line: a small
        // leftover wiggle, biggest at the very start where F(1) = F(2) = 1 breaks the pattern.
        // The network forecasts the next wiggle (tiny), and the model adds the line back:
        //     next ≈ 14.10 + 0.48 + (small correction) ≈ 14.596,
        // against the true log F(32) = 14.594. Compare 2.5, which had to learn the slope
        // itself and landed at 14.648.
        let next = forecastNext(forecaster, history: history)

        // ── Transform back: exp undoes log, and keeps the quantiles in order.
        let median = exp(next.median)
        let low = exp(next.low)
        let high = exp(next.high)
        print("[Fibonacci3 log \(weights.name)] F(32) median \(median), 10–90% [\(low), \(high)], truth \(expected33rd)")

        // Observed (fp32 and fp16 agree to within a few units):
        //   median ≈ 2,182,360 (+0.19%), 80% interval ≈ [2,177,000, 2,189,000]
        // A log-space error of 0.002 becomes a relative error of about 0.2% after exp.
        // Unlike 2.5, the truth sits inside a tight 80% interval.
        let relativeError = abs(median - Float(expected33rd)) / Float(expected33rd)
        #expect(relativeError < 0.01)
        #expect(low <= Float(expected33rd) && Float(expected33rd) <= high)
    }

    // MARK: 3. Ratio transform

    @Test("Ratios F(n)/F(n-1): converge to φ, forecast within 1%", arguments: TimesFM3Weights.available)
    func testRatioFibonacci(weights: TimesFM3Weights) throws {
        let forecaster = try load(weights)

        // Keep the integers: the inverse transform needs F(31).
        let fib = fibonacci(32)

        // ── Transform: r(n) = F(n) / F(n-1) for n = 2...31, i.e. 30 values:
        //     1.0, 2.0, 1.5, 1.667, 1.6, ..., 1.6180339
        // A few wobbles, then flat at the golden ratio φ.
        let ratios = (2 ..< fib.count).map { Float(fib[$0]) / Float(fib[$0 - 1]) }

        // ── Detrending check: no. There's no steady slope here; the series settles to a flat
        // level, so a line explains little of the (early) wobble and the data goes in as-is.
        #expect(!detrending(forecaster, history: ratios).applied)

        // ── Forecast in ratio space: the next ratio r(32) = F(32) / F(31).
        let next = forecastNext(forecaster, history: ratios)
        let golden = Float((1 + 5.0.squareRoot()) / 2)

        // ── Transform back: F(32) = r(32) · F(31). Multiplying by a positive number keeps the
        // quantiles in order, so the interval converts the same way.
        let estimate = next.median * Float(fib[31])
        let low = next.low * Float(fib[31])
        let high = next.high * Float(fib[31])
        print("[Fibonacci3 ratio \(weights.name)] next ratio \(next.median) (φ = \(golden)), F(32) ≈ \(estimate), 10–90% [\(low), \(high)], truth \(expected33rd)")

        // Observed:
        //   fp32: ratio ≈ 1.62174, F(32) ≈ 2,183,300 (+0.23%), 80% ≈ [2,172,400, 2,190,000]
        //   fp16: ratio ≈ 1.62178, F(32) ≈ 2,183,360 (+0.23%), 80% ≈ [2,172,600, 2,189,900]
        // Slightly worse than 2.5's 0.15% here, and slightly worse than 3.0's own log result.
        // The two good transforms have swapped places. Which transform works best depends on
        // the model, so it's worth trying more than one.
        #expect(abs(next.median - golden) < 0.01)
        let relativeError = abs(estimate - Float(expected33rd)) / Float(expected33rd)
        #expect(relativeError < 0.01)
        #expect(low <= Float(expected33rd) && Float(expected33rd) <= high)
    }

    // MARK: 4. fp16 vs fp32

    @Test(
        "fp16 rounding barely moves confident forecasts, but moves an unsure one a lot",
        .enabled(if: TimesFM3Weights.available.count == 2))
    func testPrecisionSensitivity() throws {
        // Load both copies: Google's exact fp32 weights and the rounded fp16 copy. Both run in
        // float32; the only difference is that every fp16 weight was rounded once, by about
        // 0.02% on average, when the file was written.
        let fp32 = try load(.fp32)
        let fp16 = try load(.fp16)

        let fib = fibonacci(32)
        let histories: [(name: String, history: [Float])] = [
            ("raw", fib.map { Float($0) }),
            ("log", fib.dropFirst().map { Float(log(Double($0))) }),
            ("ratio", (2 ..< fib.count).map { Float(fib[$0]) / Float(fib[$0 - 1]) }),
        ]

        // Compare the two medians, measured against the size of the thing being forecast.
        var relativeGap = [String: Float]()
        for (name, history) in histories {
            let a = forecastNext(fp32, history: history).median
            let b = forecastNext(fp16, history: history).median
            relativeGap[name] = abs(a - b) / abs(a)
            print("[Fibonacci3 precision] \(name): fp32 \(a), fp16 \(b), relative gap \(relativeGap[name]!)")
        }

        // Observed: log and ratio differ by roughly 1e-7 and 3e-5 (in the transformed space).
        // The model is confident there, so small weight changes barely move its answer.
        #expect(relativeGap["log"]! < 1e-4)
        #expect(relativeGap["ratio"]! < 1e-4)

        // Raw differs by about 5.5% (≈ 3,433,000 vs ≈ 3,244,000). With an 80% interval ~14
        // million wide, the median sits on a very flat, uncertain part of the model's output,
        // so a tiny nudge to the weights slides it a long way. This is also why the earlier
        // golden-value tests allow more slack for fp16: its cost depends on the input.
        #expect(relativeGap["raw"]! > 0.01)
    }
}
