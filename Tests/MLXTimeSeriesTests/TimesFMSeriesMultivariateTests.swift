import Foundation
import MLX
import Testing

@testable import MLXTimeSeries

// MARK: - Building intuition: four experiments where series and multivariate meet
//
// Read TimesFMMultivariateTests (experiments 1–6: what two variates can and can't do) and
// TimesFMSeriesTests (experiments 7–14: sequences and series with exact answers) first.
//
// This file combines them. Each experiment feeds TimesFM TWO related mathematical sequences, where
// the relationship is exact: one variate is the running total of the other, or a function of it.
// Because the relationship is exact, we know precisely what "using the other variate well" would
// look like, and can measure whether the model does it.
//
//   - TimesFM 2.5 forecasts each variate on its own. Passing both changes nothing (a control).
//   - TimesFM 3.0's variate attention lets each variate see the other.
//
// (Remember: variates are not covariates. Both are forecast together; you can't supply either's
// future.)
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
//   #   Experiment                     Idea
//   ──  ─────────────────────────────  ──────────────────────────────────────────────────────
//   15  Terms and partial sums         B is the running total of A. Does the model know?
//   16  Taylor polynomial vs sin x     Two curves that agree, then part ways.
//   17  Convergent + divergent, joint  Two series with opposite fates in one input.
//   18  A series and its remainder     A + B = the limit, exactly; derive B instead of forecasting.
//
// Conventions: `forecast(...)` returns median, 10% and 90% quantiles per step; `median(v)` picks
// variate v. Errors are measured against the exact continuation.

@Suite(
    "TimesFM series-as-multivariate experiments",
    .enabled(if: !TimesFMVariant.available.isEmpty),
    .serialized)
struct TimesFMSeriesMultivariateTests {

    /// Partial sums S₁…Sₙ of the series with the given terms (computed in double precision).
    static func partialSums(_ term: (Int) -> Double, upTo n: Int) -> [Float] {
        var total = 0.0
        return (1 ... n).map { k -> Float in
            total += term(k)
            return Float(total)
        }
    }

    // MARK: 15. Terms and partial sums

    @Test("15. Terms and partial sums: B is the running total of A", arguments: TimesFMVariant.available)
    func testTermsAndPartialSums(model: TimesFMVariant) throws {
        let forecaster = try model.load()

        // ── The idea. Variate A holds the terms aₙ = 1/n² and variate B holds their partial sums
        // Sₙ = a₁ + … + aₙ (which climb to π²/6 = 1.645). B is *defined* from A, so in principle
        // the two variates carry the same information twice. We forecast 32 steps from 128 and
        // compare THREE ways of getting B's future:
        //   (1) B alone (the baseline);
        //   (2) B forecast jointly with A (does 3.0 exploit the link?);
        //   (3) forecast A's terms alone, then add them up from the last known sum. This one is
        //       "use the relationship yourself" and mirrors the differencing lesson.
        let history = 128
        let horizon = 32
        let terms = (1 ... history + horizon).map { Float(1 / Double($0 * $0)) }
        let sums = Self.partialSums({ 1 / Double($0 * $0) }, upTo: history + horizon)
        let truth = Array(sums[history...])

        let alone = forecast(forecaster, Array(sums[0 ..< history]), horizon: horizon).median()
        let joint = forecast(forecaster, [Array(terms[0 ..< history]), Array(sums[0 ..< history])], horizon: horizon)
        // Terms are positive and tiny (0.00006 at n = 128): forecast them in log space.
        let logTerms = terms[0 ..< history].map { Float(log(Double($0))) }
        let nextTerms = forecast(forecaster, logTerms, horizon: horizon).median().map { Float(exp(Double($0))) }
        var running = sums[history - 1]
        let viaTerms = nextTerms.map { t -> Float in
            running += t
            return running
        }
        let aloneError = maxAbsDifference(alone, truth)
        let jointError = maxAbsDifference(joint.median(1), truth)
        let viaTermsError = maxAbsDifference(viaTerms, truth)
        let mixingChange = maxAbsDifference(alone, joint.median(1))
        print("[Series 15 terms+sums \(model.name)] worst abs error in S: alone \(aloneError), joint \(jointError), via terms \(viaTermsError); joint changed B by up to \(mixingChange); final S: truth \(truth.last!), alone \(alone.last!), joint \(joint.median(1).last!), via terms \(viaTerms.last!)")

        // Observed (worst absolute error in S over 32 steps; fp32 rows, fp16 within a few percent;
        // the truth ends at 1.63870):
        //
        //                  B alone     B with A (joint)   via terms (forecast A, then add up)
        //     TimesFM 2.5  5.3e-3      5.3e-3             9.0e-5
        //     TimesFM 3.0  5.2e-4      2.9e-4             2.7e-5
        //
        //   * 2.5 is a control: passing A changes B by exactly 0.0, so the "joint" column is
        //     literally the "alone" column.
        //   * 3.0's joint forecast moves B by up to 6.3e-4 and improves it about 1.8× (5.2e-4 →
        //     2.9e-4). Mixing helps, modestly, on a relationship this clean.
        //   * The big win is not mixing, it is doing the maths yourself. Forecasting the terms
        //     and summing them beats the direct forecast by 59× on 2.5 and 19× on 3.0, and beats
        //     3.0's joint forecast too (2.7e-5 vs 2.9e-4, about 10×). Same lesson as
        //     differencing: forecast the increments, let arithmetic do the accumulation.
        //   * Hypothesis (not tested here) for why it works: the terms are a smooth, nearly
        //     straight line in log space, and the running total only ever adds their (tiny)
        //     errors instead of forecasting a level near a limit.
        //
        // Hypothesis (not tested here): 3.0's gain from seeing A is that A's shape (a decaying
        // curve) tells the attention layer that B should flatten.
        #expect(viaTermsError < aloneError / 5, "Summing forecast terms should beat forecasting the sum directly")
        #expect(viaTermsError < 1e-3)
        switch model.version {
        case .v25:
            #expect(mixingChange == 0, "2.5 treats variates independently")
        case .v3:
            #expect(jointError < aloneError * 1.5, "3.0 joint forecast should not be much worse than solo")
        }
    }

    // MARK: 16. Taylor polynomial vs the real function

    @Test("16. Taylor vs sin x, together: does the model notice the curves part ways?", arguments: TimesFMVariant.available)
    func testTaylorAndSineTogether(model: TimesFMVariant) throws {
        let forecaster = try model.load()

        // ── The idea. Same two curves as experiment 10, now as a pair: variate 0 is the degree-5
        // Taylor polynomial T(x), variate 1 is sin x, sampled at x = 0, 0.05, …, 3.15. Up to
        // x ≈ 2 they're nearly the same; past 2.5 they diverge. If 3.0 mixes them, does seeing the
        // bending sine help it forecast the polynomial to *also* bend (wrongly!), or the sine to
        // keep rising (wrongly!)? Compare each variate's joint forecast with its solo one.
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

        let soloSine = forecast(forecaster, sineHistory, horizon: horizon).median()
        let soloTaylor = forecast(forecaster, taylorHistory, horizon: horizon).median()
        let joint = forecast(forecaster, [taylorHistory, sineHistory], horizon: horizon)
        let sineSolo = meanAbsError(soloSine, sineTruth)
        let sineJoint = meanAbsError(joint.median(1), sineTruth)
        let taylorSolo = meanAbsError(soloTaylor, taylorTruth)
        let taylorJoint = meanAbsError(joint.median(0), taylorTruth)
        let sineChange = maxAbsDifference(soloSine, joint.median(1))
        let taylorChange = maxAbsDifference(soloTaylor, joint.median(0))
        print("[Series 16 taylor+sin \(model.name)] sin MAE solo \(sineSolo) joint \(sineJoint); T MAE (vs T truth) solo \(taylorSolo) joint \(taylorJoint); largest change from mixing: sin \(sineChange), T \(taylorChange)")

        // Observed (mean absolute error over 32 steps; T's truth ends at 7.04 and sin's at −1.0;
        // fp32 rows, fp16 within a few percent):
        //
        //                  sin solo   sin joint   T solo    T joint   biggest shift (sin / T)
        //     TimesFM 2.5  0.172      0.172       0.945     0.945     0.0 / 0.0
        //     TimesFM 3.0  0.033      0.037       0.272     0.132     0.013 / 0.93
        //
        //   * 2.5 is again a control: identical numbers solo and joint.
        //   * On 3.0, pairing moves the POLYNOMIAL a lot (up to 0.93 at one step) and halves its
        //     error (0.27 → 0.13), while barely nudging the sine (0.033 → 0.037, slightly worse).
        //   * T's true continuation (a quintic) shoots up to 7.04 while the sine turns over. The
        //     model has no way to know that from the numbers, so treat the improvement as a fact
        //     about this pair, not as "3.0 understands Taylor polynomials". (Experiment 10:
        //     T-from-T forecasts on their own underestimate the climb, ending at 2.9 on 2.5 and
        //     5.6 on 3.0.)
        //   * The sine, whose shape is already the "cleaner" signal, gets nothing useful from T.
        //
        // Hypothesis (not tested here): information flows more usefully from the smooth, bending
        // series into the one that is still ambiguous than the reverse.
        switch model.version {
        case .v25:
            #expect(sineChange == 0 && taylorChange == 0, "2.5 treats variates independently")
        case .v3:
            #expect(taylorChange > 0.1, "3.0 should visibly move T when it can see sin")
            #expect(taylorJoint < taylorSolo, "In this pair, mixing should have helped T")
            #expect(sineJoint < sineSolo * 2, "Mixing should not wreck the sine")
        }
    }

    // MARK: 17. Convergent and divergent together

    @Test("17. Convergent and divergent together: does one series leak into the other?", arguments: TimesFMVariant.available)
    func testConvergentAndDivergentTogether(model: TimesFMVariant) throws {
        let forecaster = try model.load()

        // ── The idea. Variate 0 is Σ1/n² (flattens toward 1.645), variate 1 is the harmonic series
        // Σ1/n (keeps creeping up, 5.4 → 5.9 over the horizon). Their fates are opposite and their
        // sizes differ by about 3×. Passing them together is mostly an independence check like
        // experiment 2, on curves that are visually similar: does mixing them pull the flat one
        // upward or the rising one flat? (128 in, 64 out, as in experiment 8.)
        let history = 128
        let horizon = 64
        let basel = Self.partialSums({ 1 / Double($0 * $0) }, upTo: history + horizon)
        let harmonic = Self.partialSums({ 1 / Double($0) }, upTo: history + horizon)
        let baselTruth = Array(basel[history...])
        let harmonicTruth = Array(harmonic[history...])

        let soloBasel = forecast(forecaster, Array(basel[0 ..< history]), horizon: horizon).median()
        let soloHarmonic = forecast(forecaster, Array(harmonic[0 ..< history]), horizon: horizon).median()
        let joint = forecast(forecaster, [Array(basel[0 ..< history]), Array(harmonic[0 ..< history])], horizon: horizon)
        let growthTrue = harmonicTruth.last! - harmonic[history - 1]
        let baselSolo = meanAbsError(soloBasel, baselTruth)
        let baselJoint = meanAbsError(joint.median(0), baselTruth)
        let growthSolo = soloHarmonic.last! - harmonic[history - 1]
        let growthJoint = joint.median(1).last! - harmonic[history - 1]
        let baselChange = maxAbsDifference(soloBasel, joint.median(0))
        let harmonicChange = maxAbsDifference(soloHarmonic, joint.median(1))
        print("[Series 17 converge+diverge \(model.name)] Basel MAE solo \(baselSolo) joint \(baselJoint); harmonic growth over horizon: true \(growthTrue), solo \(growthSolo), joint \(growthJoint); largest change from mixing: Basel \(baselChange), harmonic \(harmonicChange)")

        // Observed (fp32 rows, fp16 within a few percent):
        //
        //                  Basel MAE solo → joint   harmonic growth (true 0.404): solo → joint
        //     TimesFM 2.5  2.4e-3 → 2.4e-3          0.383 → 0.383
        //     TimesFM 3.0  2.5e-4 → 2.2e-4          0.399 → 0.400
        //
        //   * 2.5 is an exact control (changes of 0.0).
        //   * 3.0 shifts the two forecasts by at most 1.6e-4 (Basel) and 4.3e-3 (harmonic), tiny
        //     compared with how far apart the two series are. There is NO leak of one fate into
        //     the other: the convergent series still flattens and the divergent one still
        //     climbs, by 0.40 vs 0.40 true.
        //   * Compare with experiment 2, where 3.0 moved a random walk by 23% of its std when
        //     it was paired with an unrelated series. Here the pair are both smooth, positive,
        //     monotone curves, and mixing is nearly neutral. Hypothesis: mixing does the most
        //     harm to a variate whose own history is noisy or ambiguous, and little to one whose
        //     history is already clean.
        //   * Surprise for those who expect "the model can't see divergence": the harmonic series
        //     is *not* flattened. Its forecast growth is within 1–6% of the truth on every
        //     checkpoint. Over 64 steps the log curve is nearly a straight line, which is easy.
        //     (A 64-step window cannot tell a log from a bounded curve; the true test would be
        //     thousands of terms.)
        #expect(abs(growthSolo - growthTrue) / growthTrue < 0.15, "Harmonic growth should be roughly right")
        #expect(abs(growthJoint - growthTrue) / growthTrue < 0.15, "Harmonic growth should be roughly right when paired")
        #expect(baselJoint < 0.01, "Basel forecast should stay accurate when paired")
        switch model.version {
        case .v25:
            #expect(baselChange == 0 && harmonicChange == 0, "2.5 treats variates independently")
        case .v3:
            #expect(baselChange < 0.01 && harmonicChange < 0.05, "3.0 mixing should be small on two smooth curves")
        }
    }

    // MARK: 18. A series and its remainder

    @Test("18. A series and its remainder: A + B = the limit, so derive B", arguments: TimesFMVariant.available)
    func testSeriesAndRemainder(model: TimesFMVariant) throws {
        let forecaster = try model.load()

        // ── The idea. A is the partial sums of Σ1/n². The limit is L = π²/6, so the REMAINDER
        // B = L − Sₙ (how far the sum still has to go) is 1/n-ish and shrinks to 0. Like
        // experiment 4's conserved total, A + B = L exactly, always. Three ways to get the future:
        //   (1) forecast A and B jointly;
        //   (2) forecast A alone, then derive B̂ = L − Â;
        //   (3) forecast B alone, then derive Â = L − B̂.
        // Which is best? Does the joint forecast at least keep A + B = L?
        let history = 128
        let horizon = 32
        let limit = Float(Double.pi * Double.pi / 6)
        let sums = Self.partialSums({ 1 / Double($0 * $0) }, upTo: history + horizon)
        let remainder = sums.map { limit - $0 }
        // Note: limit − Sₙ ≈ 1/n, a positive shrinking value (0.0078 at n = 128).
        let sumTruth = Array(sums[history...])
        let remainderTruth = Array(remainder[history...])

        let joint = forecast(forecaster, [Array(sums[0 ..< history]), Array(remainder[0 ..< history])], horizon: horizon)
        let soloSums = forecast(forecaster, Array(sums[0 ..< history]), horizon: horizon).median()
        let soloRemainder = forecast(forecaster, Array(remainder[0 ..< history]), horizon: horizon).median()

        let constraintGap = zip(joint.median(0), joint.median(1)).map { abs($0 + $1 - limit) }.max()!
        let derivedRemainder = soloSums.map { limit - $0 }
        let derivedSums = soloRemainder.map { limit - $0 }
        let remainderJoint = meanAbsError(joint.median(1), remainderTruth)
        let remainderFromA = meanAbsError(derivedRemainder, remainderTruth)
        let remainderAlone = meanAbsError(soloRemainder, remainderTruth)
        let sumsJoint = meanAbsError(joint.median(0), sumTruth)
        let sumsFromB = meanAbsError(derivedSums, sumTruth)
        let sumsAlone = meanAbsError(soloSums, sumTruth)
        print("[Series 18 remainder \(model.name)] max |A^+B^−L| joint \(constraintGap); error in B (remainder): joint \(remainderJoint), from A \(remainderFromA), direct alone \(remainderAlone); error in A (sums): joint \(sumsJoint), from B \(sumsFromB), direct alone \(sumsAlone)")

        // Observed (mean absolute error over 32 steps; fp32 rows, fp16 within a few percent):
        //
        //                  constraint gap    A (sums): joint / from B / alone      B (remainder): joint / from A / alone
        //     TimesFM 2.5  6.8e-3            2.0e-3 / 7.3e-4 / 2.0e-3              7.3e-4 / 2.0e-3 / 7.3e-4
        //     TimesFM 3.0  3.9e-4            7.5e-5 / 9.2e-5 / 3.3e-4              1.4e-4 / 3.3e-4 / 9.2e-5
        //
        //   * The best way to forecast A (the sums) is NOT to forecast A. On 2.5, forecasting B
        //     (the remainder) and computing L − B̂ gives 7.3e-4 vs 2.0e-3 direct: 2.7× better.
        //     Same recipe as experiment 4 and the differencing lessons. Hypothesis (not tested
        //     here): the remainder is a clean shrinking curve, while the sums sit near a limit
        //     with only tiny changes riding on a large level.
        //   * 2.5's joint = solo, exactly, for both variates. Its constraint gap, |Â + B̂ − L|
        //     of up to 6.8e-3, is the price of not connecting them.
        //   * 3.0 is the only model whose joint forecast keeps A + B = L tightly (gap 3.9e-4,
        //     17× tighter than 2.5) and its joint A error (7.5e-5) is the best of all the A
        //     numbers, 4.4× better than 3.0 alone. Its B in the joint is worse than B alone
        //     (1.4e-4 vs 9.2e-5), so the mixing does not strictly dominate.
        //   * Take-away: on either model, pick the variate that is the easier, cleaner series,
        //     forecast that one, and let subtraction give you the other. Where 3.0 mixing helps,
        //     it helps A, which is exactly the variate that is hard alone.
        #expect(sumsFromB < sumsAlone, "Deriving A from a forecast of B beats forecasting A directly")
        #expect(constraintGap < 0.02, "Even 2.5's independent forecasts satisfy A + B = L to about 1%")
        switch model.version {
        case .v25:
            #expect(sumsFromB < sumsAlone / 2, "2.5: the derived route is at least 2× better")
            #expect(abs(remainderJoint - remainderAlone) < 1e-6, "2.5 joint equals solo")
        case .v3:
            #expect(constraintGap < 2e-3, "3.0 keeps A + B = L much more tightly")
            #expect(sumsJoint < sumsAlone, "3.0 joint should help the sums")
        }
    }
}
