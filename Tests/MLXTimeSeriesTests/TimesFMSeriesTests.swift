import Foundation
import MLX
import Testing

@testable import MLXTimeSeries

// MARK: - Building intuition: eight experiments from "Sequences and Series"
//
// Read TimesFMFibonacciTests and TimesFMIntuitionTests first. This file borrows its material from
// second-semester calculus (sequences, series, convergence) because the maths gives us something
// rare: sequences that are perfectly deterministic AND have a known limit or growth rate, so we
// can score a forecast against the exact truth and ask a sharper question than "is it close?":
//
//     Does the model reproduce the mathematical BEHAVIOR (converging, diverging, alternating,
//     going chaotic), or just continue the last few values?
//
// The model has never seen a formula. It only sees numbers. Where the behavior is visible in a
// few dozen numbers it can continue it; where it isn't (a series that diverges too slowly to
// notice, a function that has left the range where a polynomial fits it) it can't, and neither
// could you from the numbers alone.
//
// All experiments run on the four checkpoints (see TimesFMVariants.swift):
//
//     TimesFM 2.5 fp16   TimesFM 2.5 fp32   TimesFM 3.0 fp32   TimesFM 3.0 fp16
//
// All four compute in float32; fp16 vs fp32 is only how the weights are *stored*.
//
// ─────────────────────────────────────────────────────────────────────────────────────────
// The experiments
// ─────────────────────────────────────────────────────────────────────────────────────────
//
//   #   Experiment                  Idea
//   ──  ──────────────────────────  ───────────────────────────────────────────────────────
//   7   Geometric sequences a·rⁿ    log turns a·rⁿ into a straight line (even when r < 0).
//   8   Convergent vs divergent     Σ1/n² converges, Σ1/n diverges; can the model tell?
//   9   Alternating series          Leibniz partial sums oscillate down onto π.
//   10  Taylor polynomial vs sin x  Forecasting the pattern you are shown, not the function.
//   11  Fixed-point iteration       xₙ₊₁ = cos xₙ converges; Newton's method converges fast.
//   12  Logistic map                Where a deterministic rule stops being forecastable.
//   13  Growth-rate hierarchy       ln n, n², 2ⁿ, n!: what log does to each.
//   14  Ratio test                  The ratio aₙ₊₁/aₙ, and where it stops being decisive.
//
// (Numbering continues from TimesFMMultivariateTests, 1–6; the multivariate series experiments
// are 15–18 in TimesFMSeriesMultivariateTests.)
//
// Conventions: `forecast(...)` returns median, 10% and 90% quantiles per step. Errors are measured
// against the exact mathematical continuation.

@Suite(
    "TimesFM sequences and series experiments",
    .enabled(if: !TimesFMVariant.available.isEmpty),
    .serialized)
struct TimesFMSeriesTests {

    // MARK: 7. Geometric sequences

    @Test("7. Geometric sequences a·rⁿ: log makes them straight lines", arguments: TimesFMVariant.available)
    func testGeometricSequences(model: TimesFMVariant) throws {
        let forecaster = try model.load()

        // ── The idea. aₙ = 100·rⁿ. If |r| < 1 the terms shrink toward 0; if r > 1 they explode.
        // Either way the curve bends, but log(aₙ) = log 100 + n·log r is a STRAIGHT LINE, which is
        // the easiest thing there is to forecast. This is the Fibonacci lesson again (Fibonacci
        // is asymptotically geometric with r = 1.618).
        //   r = 0.8   decays smoothly
        //   r = 1.1   grows smoothly
        //   r = −0.7  decays while flipping sign every step: log can't take a negative, so we
        //             log the *size* |aₙ| and put the sign back by hand ((−1)ⁿ is known).
        let history = 64
        let horizon = 16
        func term(_ r: Double, _ n: Int) -> Float { Float(100 * pow(r, Double(n))) }

        var rawErrors: [Double: Float] = [:]
        var logErrors: [Double: Float] = [:]
        for r in [0.8, 1.1] {
            let series = (0 ..< history).map { term(r, $0) }
            let truth = (history ..< history + horizon).map { term(r, $0) }
            let raw = forecast(forecaster, series, horizon: horizon).median()
            // Forecast log(a), then exp() it back. exp is increasing, so a median maps to a median.
            let logged = forecast(forecaster, series.map { Float(log(Double($0))) }, horizon: horizon)
                .median().map { Float(exp(Double($0))) }
            rawErrors[r] = worstRelativeError(raw, truth)
            logErrors[r] = worstRelativeError(logged, truth)
            print("[Series 7 geometric r=\(r) \(model.name)] worst relative error: raw \(rawErrors[r]!), log \(logErrors[r]!)")
        }

        // r = −0.7: forecast log|a|, restore the sign (−1)^n by hand.
        let alternating = (0 ..< history).map { term(-0.7, $0) }
        let alternatingTruth = (history ..< history + horizon).map { term(-0.7, $0) }
        let rawAlt = forecast(forecaster, alternating, horizon: horizon).median()
        let sizes = forecast(forecaster, alternating.map { Float(log(Double(abs($0)))) }, horizon: horizon).median()
        let signedBack = sizes.enumerated().map { (i: Int, s: Float) -> Float in
            let sign: Float = (history + i) % 2 == 0 ? 1 : -1
            return sign * Float(exp(Double(s)))
        }
        let rawAltError = worstRelativeError(rawAlt, alternatingTruth)
        let signedError = worstRelativeError(signedBack, alternatingTruth)
        print("[Series 7 geometric r=-0.7 \(model.name)] worst relative error: raw \(rawAltError), log|a| with sign restored \(signedError); raw first values \(Array(rawAlt.prefix(4))) truth \(Array(alternatingTruth.prefix(4)))")

        // Observed (worst relative error over 16 steps, |forecast − truth| / |truth|; fp32 rows,
        // fp16 within a few percent):
        //
        //                  r = 0.8            r = 1.1           r = −0.7
        //                  raw     → log      raw    → log      raw      → log|a| + sign
        //     TimesFM 2.5  1.7e5   → 0.122    1.60   → 0.075    1.9e9    → 0.187
        //     TimesFM 3.0  4.5e4   → 1.1e-6   0.112  → 1.4e-6   6.3e8    → 1.3e-6
        //
        //   * The transform is everything. Raw, the r = 0.8 sequence is hopeless in relative terms:
        //     the true terms have shrunk to 6e-5 … 2e-6 by the forecast window (from a start of
        //     100), and the raw forecast misses them by a factor of 1e5. The absolute error is
        //     tiny; relative error just makes the mismatch impossible to miss. For r = −0.7 the
        //     truth is about 1e-8 and the raw forecasts are 0.007…0.06 on 2.5, and about −0.03…
        //     −0.08 on 3.0.
        //   * In log space the line is straight, and 3.0 nails it to about 1e-6 relative error,
        //     which is float32 rounding. That is consistent with 3.0's linear detrending: an
        //     exactly linear log-series is removed and re-added, leaving nothing to forecast.
        //   * 2.5 has no such step and is off by 7–19% even in log space. Still, going from raw
        //     to log improves it by 1.3 orders of magnitude (r = 1.1) to 6 (r = 0.8) and 10
        //     (r = −0.7), just far less exactly than 3.0.
        //   * r = −0.7 alternates sign, so log needs the absolute value and the sign is put back
        //     by hand ((−1)ⁿ is known). The trick works because we *know* the sign rule; the
        //     model was never asked to learn it. Feeding the raw sign-flipping series is the
        //     worst case of all (relative error 1e9).
        switch model.version {
        case .v25:
            for r in [0.8, 1.1] { #expect(logErrors[r]! < 0.3, "2.5 log route should be within 30%") }
            #expect(signedError < 0.5)
        case .v3:
            for r in [0.8, 1.1] { #expect(logErrors[r]! < 1e-4, "3.0 detrends an exactly linear log-series") }
            #expect(signedError < 1e-4)
        }
        for r in [0.8, 1.1] { #expect(logErrors[r]! < rawErrors[r]! / 5, "The log route should beat the raw route") }
        #expect(signedError < rawAltError / 1e3, "Restoring the sign by hand should beat raw by orders of magnitude")
    }

    // MARK: 8. Convergent vs divergent series

    @Test("8. Convergent vs divergent: Σ1/n² settles, Σ1/n creeps up forever", arguments: TimesFMVariant.available)
    func testConvergentVsDivergent(model: TimesFMVariant) throws {
        let forecaster = try model.load()

        // ── The idea. Partial sums Sₙ = a₁ + … + aₙ. For aₙ = 1/n² the sums converge (to π²/6 =
        // 1.6449). For aₙ = 1/n (the harmonic series) they diverge to infinity, but incredibly
        // slowly: Hₙ ≈ ln n + 0.577, so from n = 128 to n = 192 it only rises by ln(1.5) = 0.405.
        // Inside a 128-number window the two curves look alike: both rise, both bend flat. So:
        //   - Will the model flatten the divergent one, missing its slow climb?
        //   - Can the transform "forecast the TERMS, then add them up" rescue it? (This is the
        //     mirror of the squares experiment: there we differenced and rebuilt with a running
        //     sum; here the running sum IS the data, and the terms are its differences.)
        let history = 128
        let horizon = 64
        func partialSums(_ term: (Int) -> Double, upTo n: Int) -> [Float] {
            var total = 0.0
            return (1 ... n).map { k -> Float in
                total += term(k)
                return Float(total)
            }
        }
        let basel = partialSums({ 1 / Double($0 * $0) }, upTo: history + horizon)
        let harmonic = partialSums({ 1 / Double($0) }, upTo: history + horizon)
        let limit = Float(Double.pi * Double.pi / 6)

        // Convergent: sums forecast directly.
        let baselNext = forecast(forecaster, Array(basel[0 ..< history]), horizon: horizon)
        let baselTruth = Array(basel[history...])
        // Divergent: (a) direct, (b) forecast terms 1/n in log space, then cumulative-sum.
        let harmonicNext = forecast(forecaster, Array(harmonic[0 ..< history]), horizon: horizon).median()
        let harmonicTruth = Array(harmonic[history...])
        let terms = (1 ... history).map { Float(log(1 / Double($0))) }
        let nextTerms = forecast(forecaster, terms, horizon: horizon).median().map { Float(exp(Double($0))) }
        var running = harmonic[history - 1]
        let rebuilt = nextTerms.map { t -> Float in
            running += t
            return running
        }
        let lastKnown = harmonic[history - 1]
        let baselError = meanAbsError(baselNext.median(), baselTruth)
        let growthTrue = harmonicTruth.last! - lastKnown
        let growthDirect = harmonicNext.last! - lastKnown
        let directError = maxAbsDifference(harmonicNext, harmonicTruth)
        let viaTermsError = maxAbsDifference(rebuilt, harmonicTruth)
        print("[Series 8 sums \(model.name)] Basel: final median \(baselNext.median().last!) (limit \(limit), truth \(baselTruth.last!)), MAE \(baselError); harmonic growth over 64 steps: true \(growthTrue), direct \(growthDirect), via terms \(rebuilt.last! - lastKnown); harmonic worst abs error direct \(directError) via terms \(viaTermsError)")

        // Observed (fp32 rows, fp16 within a few percent; Basel's truth at the end of the horizon
        // is 1.63974 and the limit π²/6 is 1.64493):
        //
        //                  Basel MAE   Basel final    harmonic growth over 64 steps        worst abs error
        //                                             (true 0.404): direct / via terms     direct / via terms
        //     TimesFM 2.5  2.4e-3      1.6325         0.383 / 0.433                        0.079 / 0.029
        //     TimesFM 3.0  2.5e-4      1.6398         0.399 / 0.409                        0.010 / 0.0046
        //
        //   * The convergent series is handled well: the median flattens near the truth (3.0's
        //     final value 1.63984 is within 1e-4 of the truth 1.63974). It does not know the limit
        //     π²/6, it just sees the flattening, and lands short of it, as the truth does.
        //   * Correction to a natural guess: the model does NOT flatten the divergent series.
        //     The harmonic sums keep climbing, and the direct forecast rises by 0.383 (2.5) and
        //     0.399 (3.0) against a true 0.404. Over 64 steps ln n is close enough to a straight
        //     line that a trend-follower handles it. (Telling it apart from a bounded curve would
        //     take thousands of terms.)
        //   * Forecasting the TERMS in log space and summing them still wins on the worst error:
        //     2.7× better on 2.5 (0.079 → 0.029) and 2.2× better on 3.0 (0.010 → 0.0046). On 2.5
        //     it overshoots the growth (0.433 vs 0.404); on 3.0 it lands within 0.005.
        //   * Same lesson as the squares experiment: turn the data into the increments, forecast
        //     those, and let arithmetic rebuild the total.
        #expect(baselError < 0.01, "Basel sums should be forecast to within 0.01")
        #expect(abs(growthDirect - growthTrue) / growthTrue < 0.15, "The harmonic climb should be roughly right")
        #expect(viaTermsError < directError, "Forecasting the terms then summing should beat the direct forecast")
        switch model.version {
        case .v25: #expect(baselError < 0.01)
        case .v3: #expect(baselError < 1e-3)
        }
    }

    // MARK: 9. Alternating series

    @Test("9. Alternating series: Leibniz partial sums oscillate down onto π", arguments: TimesFMVariant.available)
    func testAlternatingSeries(model: TimesFMVariant) throws {
        let forecaster = try model.load()

        // ── The idea. π = 4·(1 − 1/3 + 1/5 − 1/7 + …). The partial sums jump above π, then below,
        // above, below, closing in. A classic calculus fact (the alternating series error bound)
        // says that after k terms the sum is off by less than the next term, about 4/(2k+3), and
        // it alternates sides. That's a rare gift: we know how wide an honest interval should be
        // and how the median should zig-zag. Does the model reproduce the zig-zag, land on π, and
        // draw a band about that wide?
        let history = 100
        let horizon = 40
        var total = 0.0
        let sums: [Float] = (0 ..< history + horizon).map { k -> Float in
            let sign: Double = k % 2 == 0 ? 1 : -1
            total += 4 * sign / Double(2 * k + 1)
            return Float(total)
        }
        let truth = Array(sums[history...])
        let pi = Float.pi
        let next = forecast(forecaster, Array(sums[0 ..< history]), horizon: horizon)
        let median = next.median()

        // Does the median sit on the same side of π as the truth, step by step?
        let sameSide = (0 ..< horizon).filter { (median[$0] - pi) * (truth[$0] - pi) > 0 }.count
        // The true error is about 4/(2k+3) (for k = 100 to 139). Compare with the band half-width.
        let boundFirst = 4 / Float(2 * history + 3)
        let boundLast = 4 / Float(2 * (history + horizon - 1) + 3)
        let halfWidthFirst = (next.high()[0] - next.low()[0]) / 2
        let halfWidthLast = (next.high()[horizon - 1] - next.low()[horizon - 1]) / 2
        let mae = meanAbsError(median, truth)
        let meanMedian = median.reduce(0, +) / Float(horizon)
        print("[Series 9 alternating \(model.name)] MAE \(mae); sits on the right side of π at \(sameSide)/\(horizon) steps; mean median \(meanMedian) (π = \(pi)); true error bound \(boundFirst)→\(boundLast) vs band half-width \(halfWidthFirst)→\(halfWidthLast); first medians \(Array(median.prefix(4))) truth \(Array(truth.prefix(4)))")

        // Observed (40 steps; fp32 rows, fp16 within a few percent; the honest mathematical
        // error bound 4/(2k+3) runs 0.0197 → 0.0142 over the horizon):
        //
        //                  MAE       right side of π   mean median   band half-width (first → last)
        //     TimesFM 2.5  7.1e-4    40 / 40           3.14184       0.0017 → 0.0043
        //     TimesFM 3.0  6.7e-4    40 / 40           3.14177       0.0016 → 0.0018
        //     (π = 3.14159)
        //
        //   * Both models reproduce the zig-zag: the median is on the correct side of π at all
        //     40 of 40 steps, and its first values (3.15157, 3.13465, 3.15017, 3.13296 on 2.5)
        //     track the truth (3.15149, 3.13179, 3.15130, 3.13198) to about 1e-3. The average of
        //     the medians lands within 3e-4 of π.
        //   * The bands are much narrower than the textbook bound (0.0016–0.0043 vs 0.014–0.020).
        //     That is not a bug: the textbook bound is the worst case, the actual error here is
        //     only about half the next term, and the model's own error (MAE 7e-4) is smaller
        //     still. The band tracks the model's *own* confidence, not the maths' guarantee.
        //   * 2.5's band widens with the horizon (0.0017 → 0.0043), the usual shape for a
        //     forecast; 3.0's stays nearly flat (0.0016 → 0.0018).
        #expect(sameSide >= horizon - 4, "The median should sit on the right side of π almost always")
        #expect(mae < 0.005, "The zig-zag should be tracked closely")
        #expect(abs(meanMedian - pi) < 0.005, "The zig-zag should average out onto π")
    }

    // MARK: 10. Taylor polynomial vs the real function

    @Test("10. Taylor polynomial vs sin x: it continues the pattern it's shown, not the function", arguments: TimesFMVariant.available)
    func testTaylorVsSine(model: TimesFMVariant) throws {
        let forecaster = try model.load()

        // ── The idea. The degree-5 Taylor polynomial  T(x) = x − x³/6 + x⁵/120  hugs sin x near
        // 0, but is a cubic-ish curve, not a wave, so past x ≈ 2.5 it peels away and shoots off.
        // We sample both at 64 points x = 0, 0.05, …, 3.15 and forecast 32 more points
        // (x up to 4.7), where sin x dives to −1 and the polynomial heads the other way.
        //   (a) forecast sin x from sin x's history   -> should follow the wave
        //   (b) forecast T(x) from T's history        -> continues a polynomial-looking curve
        // Then check (b) against BOTH the polynomial's truth and sin's truth.
        let history = 64
        let horizon = 32
        func sine(_ i: Int) -> Float { Float(sin(0.05 * Double(i))) }
        func taylor(_ i: Int) -> Float {
            let x = 0.05 * Double(i)
            let cubic: Double = pow(x, 3) / 6
            let quintic: Double = pow(x, 5) / 120
            return Float(x - cubic + quintic)
        }
        let sineHistory = (0 ..< history).map(sine)
        let taylorHistory = (0 ..< history).map(taylor)
        let sineTruth = (history ..< history + horizon).map(sine)
        let taylorTruth = (history ..< history + horizon).map(taylor)

        let fromSine = forecast(forecaster, sineHistory, horizon: horizon).median()
        let fromTaylor = forecast(forecaster, taylorHistory, horizon: horizon).median()
        let sineError = meanAbsError(fromSine, sineTruth)
        let taylorError = meanAbsError(fromTaylor, taylorTruth)
        let taylorVsSine = meanAbsError(fromTaylor, sineTruth)
        print("[Series 10 taylor \(model.name)] sin from sin: MAE \(sineError); T from T: MAE vs polynomial truth \(taylorError), MAE vs sin truth \(taylorVsSine); polynomial truth ends at \(taylorTruth.last!), sin at \(sineTruth.last!), forecast-from-T ends at \(fromTaylor.last!)")

        // Observed (32 steps; fp32 rows, fp16 within a few percent; the polynomial's truth ends at
        // 7.04 and sin's at −1.00):
        //
        //                  sin from sin   T from T (vs T's truth)   T from T (vs sin's truth)   T forecast ends at
        //     TimesFM 2.5  0.172          0.945                     2.23                        2.92
        //     TimesFM 3.0  0.033          0.272                     2.88                        5.64
        //
        //   * The model forecasts the pattern it is shown, not the function it came from. T's
        //     history rises, peaks near x = 1.6, and bottoms out near x = 3.1 just before the
        //     truth turns upward and shoots to 7.04. From those 64 points nothing announces the
        //     coming climb. Both models underestimate it (they end at 2.9 and 5.6 vs the true
        //     7.0), and 3.0 gets much closer (0.27 vs 0.95 error).
        //   * Scoring T's forecast against sin's truth is worse (2.2–2.9) than against T's own
        //     truth (0.3–0.9) on both models, so the forecast is not "secretly sin"; it just
        //     follows T's own trend, only too timidly.
        //   * sin from sin is easy on 3.0 (0.033) and fine on 2.5 (0.172, 5× worse). The history
        //     holds one full hump of the wave, and the forecast has to continue it into the
        //     trough.
        //   * Take-away: a polynomial approximation is only good on the interval it was built for,
        //     and a forecaster that only sees its output cannot know where that interval ends.
        #expect(sineError < 0.3, "sin should be continued as a wave")
        #expect(taylorVsSine > taylorError, "T's forecast should follow T's own truth more than sin's")
        switch model.version {
        case .v25: #expect(taylorError < 2.0)
        case .v3: #expect(taylorError < 0.6, "3.0 should track T's continuation roughly")
        }
    }

    // MARK: 11. Fixed-point iteration

    @Test("11. Fixed-point iteration: sequences that settle, slowly or fast", arguments: TimesFMVariant.available)
    func testFixedPointIteration(model: TimesFMVariant) throws {
        let forecaster = try model.load()

        // ── The idea. Repeatedly apply a function: xₙ₊₁ = cos(xₙ). It spirals in on the one point
        // where cos x = x (0.7390851...), overshooting left and right with each step (a damped
        // oscillation, like experiment 9 but geometric). Newton's method for √2 (xₙ₊₁ = (xₙ +
        // 2/xₙ)/2) converges much faster: the digits double each step, so after five iterations
        // the sequence is flat to float precision.
        // We feed 32 iterations of each and ask for 16 more.
        let history = 32
        let horizon = 16
        var x = 0.0
        let cosSequence: [Float] = (0 ..< history + horizon).map { _ in
            defer { x = cos(x) }
            return Float(x)
        }
        var y = 1.0
        let newton: [Float] = (0 ..< history + horizon).map { _ in
            defer { y = (y + 2 / y) / 2 }
            return Float(y)
        }
        let cosNext = forecast(forecaster, Array(cosSequence[0 ..< history]), horizon: horizon)
        let newtonNext = forecast(forecaster, Array(newton[0 ..< history]), horizon: horizon)
        let limit = Float(0.7390851332)
        let cosWorst = cosNext.median().map { abs($0 - limit) }.max()!
        let newtonWorst = newtonNext.median().map { abs($0 - Float(2).squareRoot()) }.max()!
        print("[Series 11 fixed point \(model.name)] cos: median range \(cosNext.median().min()!)…\(cosNext.median().max()!) (limit \(limit)), worst |median − limit| \(cosWorst), mean 80% band \(meanBandWidth(cosNext)); Newton: median range \(newtonNext.median().min()!)…\(newtonNext.median().max()!) (√2 = \(Float(2).squareRoot())), mean 80% band \(meanBandWidth(newtonNext))")

        // Observed (16 steps; fp32 rows, fp16 within a few percent; the truth sits within about
        // 3e-6 of the limit at n ≥ 32, so "distance from the limit" IS the error):
        //
        //                  cos: worst |median − 0.73909|   mean 80% band   Newton: worst |median − √2|   mean 80% band
        //     TimesFM 2.5  0.0036                          0.0108          0.0010                        0.0037
        //     TimesFM 3.0  0.0109                          0.0290          0.0014                        0.0124
        //
        //   * Both sequences have effectively converged by n = 32, so the honest forecast is
        //     "flat at the limit". Every checkpoint is within 1.5% of it (cos) or 0.1% (√2).
        //   * A quirk worth seeing: 3.0 is 3× WORSE than 2.5 here (0.0109 vs 0.0036 on cos) and
        //     its bands are 2.7–3.4× wider. With 32 numbers, the two models disagree about how
        //     much uncertainty to attach to a series that has stopped moving: 2.5 is more
        //     confident and right. Neither is wrong in absolute terms (all under 1.5% off).
        //   * Newton's method is more accurate than cos on both models (0.001 vs 0.004 on 2.5),
        //     because it has converged to float precision from step 5 on, so the whole history
        //     tail is one value.
        //   * Lesson: a sequence that has stopped moving is not automatically the easiest input;
        //     the model still hedges a little, and different models hedge by different amounts.
        #expect(cosWorst < 0.05, "The cos iteration should be forecast to within about 5% of its limit")
        #expect(newtonWorst < 0.01, "Newton's method should be forecast to within 1% of √2")
    }

    // MARK: 12. Logistic map

    @Test("12. Logistic map: where a deterministic rule stops being forecastable", arguments: TimesFMVariant.available)
    func testLogisticMap(model: TimesFMVariant) throws {
        let forecaster = try model.load()

        // ── The idea. xₙ₊₁ = r·xₙ·(1 − xₙ) is one line of arithmetic, fully deterministic. But its
        // behavior changes with r:
        //   r = 2.8   settles to a single value (0.643)
        //   r = 3.3   settles into a two-value cycle (0.48, 0.82)
        //   r = 3.9   chaos: no repeat, and tiny differences grow exponentially
        // Chaos is where "deterministic" and "forecastable" part ways. A pattern model can
        // continue the first two easily; for the third the honest answer is a wide band, since
        // the long-run pattern really can't be extrapolated from a finite history. We score each
        // with the true continuation (computed in double precision) and with band coverage: an
        // honest 80% band should hold about 80% of the true values.
        let history = 128
        let horizon = 32
        var maes: [Double: Float] = [:]
        var averages: [Double: Float] = [:]
        var coverages: [Double: Float] = [:]
        for r in [2.8, 3.3, 3.9] {
            var x = 0.2
            let sequence: [Float] = (0 ..< history + horizon).map { _ in
                defer { x = r * x * (1 - x) }
                return Float(x)
            }
            let truth = Array(sequence[history...])
            let next = forecast(forecaster, Array(sequence[0 ..< history]), horizon: horizon)
            // A forecaster that knows nothing: always predict the historical average.
            let average = sequence[0 ..< history].reduce(0, +) / Float(history)
            let averageError = meanAbsError([Float](repeating: average, count: horizon), truth)
            maes[r] = meanAbsError(next.median(), truth)
            averages[r] = averageError
            coverages[r] = coverage(next, truth: truth)
            print("[Series 12 logistic r=\(r) \(model.name)] always-average MAE \(averageError), MAE \(maes[r]!), mean 80% band \(meanBandWidth(next)), coverage \(coverages[r]!); truth range \(truth.min()!)…\(truth.max()!), median range \(next.median().min()!)…\(next.median().max()!)")
        }

        // Observed (32 steps; fp32 rows, fp16 within a few percent). "avg" is the MAE of always
        // predicting the history's mean, a forecaster that knows nothing:
        //
        //                  r = 2.8: MAE (avg)      r = 3.3: MAE (avg)      r = 3.9: MAE (avg)
        //     TimesFM 2.5  1.2e-4 (4.9e-3)         2.7e-3 (0.172)          0.291 (0.287)
        //     TimesFM 3.0  1.9e-5 (4.9e-3)         2.4e-3 (0.172)          0.282 (0.287)
        //
        //                  80% band coverage (r = 2.8 / 3.3 / 3.9)
        //     TimesFM 2.5  0.84 / 1.00 / 0.75
        //     TimesFM 3.0  1.00 / 0.97 / 0.78
        //
        //   * r = 2.8 and r = 3.3 are handled essentially perfectly: the error is 40–250× smaller
        //     than the know-nothing average (r = 2.8) and 64–73× smaller (r = 3.3, where the
        //     median reproduces both branches of the two-cycle: it ranges 0.4817…0.8241 vs a
        //     truth of 0.4794…0.8236).
        //   * r = 3.9 is chaos, and the model is no better than the average: 0.291 (2.5) and
        //     0.282 (3.0) vs 0.287 for "always predict the mean". The median only ranges from
        //     0.46…0.79 (2.5) or 0.34…0.76 (3.0), while the truth swings 0.10…0.97: it has
        //     effectively given up and forecast the middle, which is the right thing to do.
        //   * The bands are roughly honest. For chaos the 80% band contains 75% (2.5) and 78% (3.0)
        //     of the true values, close to the nominal 80%, and is wide (0.69–0.73 across, on a
        //     series whose range is 0.88). The model doesn't predict the chaos, but it
        //     communicates that it can't.
        //   * The 0.84–1.0 coverages on r = 2.8 and r = 3.3 are on bands that are tiny (2e-4 to
        //     5e-4 and about 0.017 wide), so the model is confidently right, not needlessly wide.
        #expect(maes[2.8]! < averages[2.8]! / 10, "A settled series should be forecast far better than the average")
        #expect(maes[3.3]! < averages[3.3]! / 20, "A two-cycle should be forecast far better than the average")
        #expect(maes[3.9]! > averages[3.9]! * 0.9, "Chaos should not be forecastable much better than the average")
        #expect(coverages[3.9]! > 0.6 && coverages[3.9]! < 0.95, "Chaos bands should be roughly calibrated")
    }

    // MARK: 13. Growth-rate hierarchy

    @Test("13. Growth-rate hierarchy: ln n, n², 2ⁿ and n! under raw and log forecasting", arguments: TimesFMVariant.available)
    func testGrowthHierarchy(model: TimesFMVariant) throws {
        let forecaster = try model.load()

        // ── The idea. In calculus you rank growth: ln n ≪ n² ≪ 2ⁿ ≪ n!. Each bends differently:
        //   ln n   barely rises (concave)
        //   n²     a parabola
        //   2ⁿ     a straight line once you take logs (log 2ⁿ = n·log 2)
        //   n!     faster than any exponential; log(n!) ≈ n·log n − n bends up gently
        // Forecast n = 2…25 (24 values), then 8 more (n = 26…33; 33! is near float32's ceiling
        // of 3.4e38). Compare forecasting the raw values against forecasting log(values) and
        // exponentiating back.
        let history = 24
        let horizon = 8
        let logFactorial: (Int) -> Double = { n in (1 ... n).map { log(Double($0)) }.reduce(0, +) }
        let sequences: [(name: String, value: (Int) -> Double)] = [
            ("ln n", { log(Double($0)) }),
            ("n²", { Double($0 * $0) }),
            ("2ⁿ", { pow(2, Double($0)) }),
            ("n!", { exp(logFactorial($0)) }),
        ]
        var rawErrors: [String: Float] = [:]
        var logErrors: [String: Float] = [:]
        for (name, value) in sequences {
            let series = (2 ..< 2 + history).map { Float(value($0)) }
            let truth = (2 + history ..< 2 + history + horizon).map { Float(value($0)) }
            let raw = forecast(forecaster, series, horizon: horizon).median()
            let logged = forecast(forecaster, series.map { Float(log(Double($0))) }, horizon: horizon)
                .median().map { Float(exp(Double($0))) }
            rawErrors[name] = worstRelativeError(raw, truth)
            logErrors[name] = worstRelativeError(logged, truth)
            print("[Series 13 growth \(name) \(model.name)] worst relative error: raw \(rawErrors[name]!), log \(logErrors[name]!); raw forecast starts \(Array(raw.prefix(3))), truth starts \(Array(truth.prefix(3)))")
        }

        // ── A fairer log test. 24 values is less than one 32-step patch, and float32 overflows past
        // 33!. The log route doesn't need the huge numbers themselves, only their logs, which we can
        // compute in double precision for any n. So repeat the log route with 64 values (n = 2…65)
        // and 16 steps (n = 66…81). The error is |exp(predicted log − true log) − 1|, the relative
        // error of the forecast value, computed without ever forming n!.
        let longHistory = 64
        let longHorizon = 16
        let logValue: [(name: String, log: (Int) -> Double)] = [
            ("ln n", { log(log(Double($0))) }),
            ("n²", { 2 * log(Double($0)) }),
            ("2ⁿ", { Double($0) * log(2) }),
            ("n!", { lgamma(Double($0) + 1) }),
        ]
        var longErrors: [String: Double] = [:]
        for (name, logOf) in logValue {
            let series = (2 ..< 2 + longHistory).map { Float(logOf($0)) }
            let truthLog = (2 + longHistory ..< 2 + longHistory + longHorizon).map { logOf($0) }
            let predicted = forecast(forecaster, series, horizon: longHorizon).median()
            let worst = zip(predicted, truthLog).map { abs(exp(Double($0) - $1) - 1) }.max()!
            longErrors[name] = worst
            print("[Series 13 growth long-log \(name) \(model.name)] worst relative error with 64 values: \(worst)")
        }

        // Observed (worst relative error; fp32 rows, fp16 within a few percent):
        //
        // 24 values in, 8 out (n = 2…25 → 26…33):
        //                 ln n            n²              2ⁿ                n!
        //                 raw / log       raw / log       raw / log         raw / log
        //     TimesFM 2.5 0.017 / 0.023   0.092 / 0.102   0.9993 / 0.98     NaN / 3.4
        //     TimesFM 3.0 0.004 / 0.017   0.013 / 0.029   0.53 / 1.4e-6     1e20 clip (1.0) / 0.31
        //
        // 64 values in, 16 out, log route only (n = 2…65 → 66…81; error of exp(pred − true log)):
        //     TimesFM 2.5 0.025           0.252           0.42 … 0.43       3.8 … 3.9
        //     TimesFM 3.0 0.008           0.050           3.7e-6            0.30
        //
        //   * ln n and n² are gentle. Raw is already good (a few percent), and taking the log
        //     makes them slightly worse, not better (0.017 → 0.023 for ln n on 2.5). Log helps
        //     when the growth is exponential; it is not a universal improvement.
        //   * 2ⁿ is the case log was made for: on 3.0 the log route is exact (1.4e-6, and 3.7e-6
        //     with the long history), where the raw route is 53% off. 2.5 is 98% off on the log
        //     route with 24 values (only 24 values is shorter than one 32-step patch, so this is
        //     partly a context confound) and still 42% off with 64, so the confound is not the
        //     whole story: 2.5 does not turn a straight log line into a straight forecast.
        //   * n! is the case that breaks everything. Its raw values reach 4e26 (n = 26), and
        //     the raw forecasts are NaN on 2.5 and ±1e20 (3.0's value clip) on 3.0, so no raw
        //     number is meaningful. In log space 3.0 is 30% off (its worst step) and 2.5 is off
        //     by 3.4–3.9× (i.e. 340–390%). Growth faster than exponential really is hard: log(n!)
        //     ≈ n·log n − n bends upward, so even the log series needs a curve, not a line.
        //   * Ranking the log-route error ln n < n² < 2ⁿ < n! is NOT the growth ranking on 2.5
        //     (its log error for 2ⁿ, 0.98, is the worst of the first three); on 3.0 the
        //     linear-detrending step makes 2ⁿ the easiest of all.
        #expect(rawErrors["ln n"]! < 0.05 && rawErrors["n²"]! < 0.2, "Gentle growth is fine raw")
        #expect(!(rawErrors["n!"]! < 0.5), "n! raw should not be usable (NaN or clipped)")
        switch model.version {
        case .v25:
            #expect(longErrors["ln n"]! < 0.1, "ln n stays easy on 2.5")
            #expect(longErrors["2ⁿ"]! > 0.2, "2.5 does not turn a straight log line into a straight forecast")
        case .v3:
            #expect(logErrors["2ⁿ"]! < 1e-4, "3.0 detrends an exactly linear log-series")
            #expect(longErrors["2ⁿ"]! < 1e-4)
            #expect(longErrors["n!"]! < 0.6, "3.0's log route on n! is roughly right, unlike raw")
        }
    }

    // MARK: 14. Ratio test

    @Test("14. Ratio test: the ratio aₙ₊₁/aₙ, and where it stops being decisive", arguments: TimesFMVariant.available)
    func testRatioTest(model: TimesFMVariant) throws {
        let forecaster = try model.load()

        // ── The idea. The ratio test says: if aₙ₊₁/aₙ settles to a limit L, then Σaₙ converges when
        // L < 1 and diverges when L > 1. (When L = 1 the test is silent.) It's also the ratio
        // transform from the Fibonacci walkthrough: the ratios of a growing sequence are much
        // flatter than the sequence itself. We forecast 16 more RATIOS from 64 and see where they
        // head:
        //   aₙ = 2ⁿ/n      ratio 2n/(n+1) → 2     (> 1: diverges)
        //   aₙ = n²/2ⁿ     ratio (n+1)²/(2n²) → ½ (< 1: converges)
        //   aₙ = 1/n       ratio n/(n+1) → 1      (silent; the series diverges)
        //   aₙ = 1/n²      ratio (n/(n+1))² → 1   (silent; the series converges)
        // The last two have the same limit but opposite fates: the ratio genuinely cannot tell
        // them apart, so the model can't either. That's not a model failure; it's a fact about
        // ratios.
        let history = 64
        let horizon = 16
        let ratios: [(name: String, limit: Double, ratio: (Int) -> Double)] = [
            ("2ⁿ/n", 2, { n in 2 * Double(n) / Double(n + 1) }),
            ("n²/2ⁿ", 0.5, { n in Double((n + 1) * (n + 1)) / Double(2 * n * n) }),
            ("1/n", 1, { n in Double(n) / Double(n + 1) }),
            ("1/n²", 1, { n in pow(Double(n) / Double(n + 1), 2) }),
        ]
        var lastMedians: [String: Float] = [:]
        var maes: [String: Float] = [:]
        for (name, limit, ratio) in ratios {
            let series = (1 ... history).map { Float(ratio($0)) }
            let truth = (history + 1 ... history + horizon).map { Float(ratio($0)) }
            let next = forecast(forecaster, series, horizon: horizon)
            lastMedians[name] = next.median().last!
            maes[name] = meanAbsError(next.median(), truth)
            print("[Series 14 ratio \(name) \(model.name)] last known \(series.last!), true ratio at step 16 \(truth.last!), forecast median at step 16 \(next.median().last!) (limit \(limit)), MAE \(maes[name]!), 80% band at step 16 \(next.low().last!)…\(next.high().last!)")
        }

        // Observed (ratios forecast 16 steps ahead from 64; fp32 rows, fp16 within a few percent):
        //
        //                  MAE by ratio:  2ⁿ/n     n²/2ⁿ    1/n      1/n²
        //     TimesFM 2.5                 3.7e-3   7.2e-3   1.8e-3   2.5e-3
        //     TimesFM 3.0                 5.0e-4   6.5e-4   2.5e-4   6.6e-4
        //
        //                  forecast median at step 16 (true value; limit)
        //                  2ⁿ/n                n²/2ⁿ               1/n                   1/n²
        //                  (1.9753; 2)         (0.5126; 0.5)       (0.9877; 1)           (0.9755; 1)
        //     TimesFM 2.5  1.9789              0.5244              0.9894                0.9754
        //     TimesFM 3.0  1.9741              0.5118              0.9870                0.9740
        //
        //   * Ratios are an easy, flat transform: every MAE is under 0.008 on 2.5 and under
        //     0.0007 on 3.0, 4–11× smaller. On 3.0 the forecast is within 0.002 of the truth at
        //     step 16 on all four.
        //   * The growing sequence's ratio (2ⁿ/n) heads to just under 2, the shrinking one (n²/2ⁿ)
        //     to just over 0.5, so the ratio test's verdicts (>1 diverges, <1 converges) are
        //     visible in the forecast.
        //   * The silent cases: the forecast ratio for the divergent 1/n and the convergent 1/n²
        //             //     are both a hair under 1 (0.9894 / 0.9754 on 2.5) at step 16, and the 80% band for
        //     1/n there (0.978…0.992 on 2.5) is entirely below 1. The model reads BOTH as
        //     "settling just under 1", which is all the ratio can say: for 1/n the true ratio is
        //     n/(n+1) < 1 forever, yet the series diverges. That's the test's blind spot, not
        //     the model's.
        //   * Lesson: forecasting a *transformed* view answers only the questions that view can
        //     answer. To tell 1/n from 1/n² you need the sums (experiment 8), not the ratios.
        for name in ["2ⁿ/n", "n²/2ⁿ", "1/n", "1/n²"] { #expect(maes[name]! < 0.02, "Ratios should be forecast to within 0.02") }
        #expect(lastMedians["2ⁿ/n"]! > 1.9 && lastMedians["2ⁿ/n"]! < 2.05, "2ⁿ/n heads to 2")
        #expect(lastMedians["n²/2ⁿ"]! > 0.45 && lastMedians["n²/2ⁿ"]! < 0.6, "n²/2ⁿ heads to ½")
        #expect(lastMedians["1/n"]! < 1 && lastMedians["1/n²"]! < 1, "The two silent cases both look like 'just under 1'")
        switch model.version {
        case .v25: break
        case .v3: for name in ["2ⁿ/n", "n²/2ⁿ", "1/n", "1/n²"] { #expect(maes[name]! < 2e-3) }
        }
    }
}
