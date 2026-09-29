import Foundation
import MLX
import Testing

@testable import MLXTimeSeries

// MARK: - Building intuition: six multivariate experiments with TimesFM
//
// Read TimesFMFibonacciTests and TimesFMIntuitionTests first. Those are single-series. This file
// asks what changes when you hand the model TWO series at once.
//
// ── What "multivariate" means here ──────────────────────────────────────────────────────────
// The input is shaped [batch, variates, time]. We always use batch 1 and 2 variates, e.g.
// "temperature" and "power". Every variate has the same length and every variate gets its own
// forecast.
//
//   - TimesFM 2.5 treats the variates as two completely unrelated series. Putting A next to B
//     cannot change B's forecast. (Experiments 2 and 5 check that literally.)
//   - TimesFM 3.0 adds *variate attention*: at every layer, the variates can look at each other.
//     Whether that helps depends on the data, and the experiments below probe when.
//
// IMPORTANT: variates are NOT covariates. You cannot tell the model "here is A's future, now
// forecast B". Both variates are forecast together, from their pasts only.
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
//   #   Experiment            Idea
//   ──  ────────────────────  ─────────────────────────────────────────────────────────────
//   1   Lagged copy           B is A delayed by 8 steps, so A's tail IS B's future.
//   2   Independence control  An unrelated companion series should not change a forecast.
//   3   Different scales      Temperature ≈ 20 and power ≈ 5000, side by side.
//   4   Conserved total       A + B = 100: the model doesn't know, so derive B from A instead.
//   5   Order swap            [A, B] vs [B, A].
//   6   Noisy twin            A clean copy of a signal next to a noisy copy of it.
//
// Conventions: `forecast(...)` (TimesFMWalkthroughSupport.swift) returns median, 10% and 90%
// quantiles per step; `median(v)` picks variate v. Synthetic series have a known continuation,
// so we can score every forecast. "Random" data uses a seeded generator, identical on every run.

@Suite(
    "TimesFM multivariate experiments",
    .enabled(if: !TimesFMVariant.available.isEmpty),
    .serialized)
struct TimesFMMultivariateTests {

    // MARK: 1. Lagged copy

    @Test("1. Lagged copy: B is A delayed 8 steps; can the model read B's future off A?", arguments: TimesFMVariant.available)
    func testLaggedCopy(model: TimesFMVariant) throws {
        let forecaster = try model.load()

        // ── The idea. Take pure random noise n(t). Let A(t) = n(t + 8) and B(t) = n(t). Then B is
        // A delayed by 8 steps: B(t) = A(t − 8). B alone is white noise, so it can't be forecast
        // at all. But B's next 8 values are exactly the last 8 values of A, sitting right there
        // in the history. A person would copy them. Can TimesFM?
        //   - 2.5 never looks across variates, so it's a control: it can only say "noise".
        //   - 3.0's variate attention could in principle notice the lag.
        let lag = 8
        let T = 256
        var random = SeededRandom(seed: 101)
        let noise: [Float] = (0 ..< T + 2 * lag).map { _ in Float(random.normal()) }
        let a: [Float] = (0 ..< T + lag).map { noise[$0 + lag] }
        let b: [Float] = (0 ..< T + lag).map { noise[$0] }
        let truth = Array(b[T ..< T + lag])  // equals a[T-lag ..< T]

        let alone = forecast(forecaster, Array(b[0 ..< T]), horizon: lag)
        let together = forecast(forecaster, [Array(a[0 ..< T]), Array(b[0 ..< T])], horizon: lag)
        let aloneError = meanAbsError(alone.median(), truth)
        let togetherError = meanAbsError(together.median(1), truth)
        let guessZeroError = meanAbsError([Float](repeating: 0, count: lag), truth)
        let change = maxAbsDifference(alone.median(), together.median(1))
        print("[MV1 lagged \(model.name)] B alone MAE \(aloneError), with A MAE \(togetherError), always-zero MAE \(guessZeroError), largest change \(change), truth \(truth), alone \(alone.median()), together \(together.median(1))")

        // Observed (8 steps of white noise, truth values between −2.3 and +1.9; fp32 rows, fp16
        // within a few percent; the pure-noise MAE of "always predict 0" is 0.990):
        //
        //                  B alone MAE   B with A MAE   largest change from adding A
        //     TimesFM 2.5  0.974         0.974          0.0 (exactly)
        //     TimesFM 3.0  0.966         0.952          0.041
        //
        //   * Neither model copies. The truth swings between −2.3 and +1.9, and every forecast
        //     stays within about ±0.13 of zero, so both models correctly treat the series as noise
        //     they can't predict and hedge near the mean. The error is only 1–4% better than
        //     guessing zero.
        //   * 2.5 is the control: adding A changes B's forecast by exactly 0.0.
        //   * 3.0 does react to A (changes of up to 0.041, so the variates are exchanging
        //     information), but the reaction is a small shift, not a copy. Hypothesis (not
        //     tested here): variate attention picks up things that happen at the same time
        //     step (shared level, shared shape) and has no mechanism for "A's value 8 steps
        //     ago is B's value now", because it has no notion of position.
        //   * Lesson: if a relationship lives in a lag, build the lagged column yourself; the
        //     model will not discover it. (Compare the differencing lessons: transform first.)
        #expect(aloneError > 0.85 * guessZeroError, "White noise is unpredictable; solo forecast should be no better than zero")
        #expect(togetherError > 0.5 * guessZeroError, "Neither model should be able to copy the lagged values")
        switch model.version {
        case .v25:
            #expect(change == 0, "2.5 treats variates independently")
        case .v3:
            #expect(change > 0, "3.0 should react to the other variate")
            #expect(change < 0.5, "...but only by a small shift, not by copying")
        }
    }

    // MARK: 2. Independence control

    @Test("2. Independence control: an unrelated companion series shouldn't change a forecast", arguments: TimesFMVariant.available)
    func testIndependence(model: TimesFMVariant) throws {
        let forecaster = try model.load()

        // ── The idea. Before trusting variate attention to *help*, check it doesn't *hurt*. A is
        // a clean seasonal series (period 24). B is a random walk with nothing to do with A.
        // Forecast each alone, then both together, and compare every variate's forecast.
        let T = 256
        let h = 32
        var random = SeededRandom(seed: 102)
        var walk: Float = 0
        let a: [Float] = (0 ..< T + h).map { t in
            let x = Double(t)
            let cycle: Double = 3 * sin(2 * Double.pi * x / 24)
            return Float(cycle + 0.01 * x)
        }
        let b: [Float] = (0 ..< T + h).map { _ in
            walk += Float(random.normal())
            return walk
        }
        let soloA = forecast(forecaster, Array(a[0 ..< T]), horizon: h)
        let soloB = forecast(forecaster, Array(b[0 ..< T]), horizon: h)
        let joint = forecast(forecaster, [Array(a[0 ..< T]), Array(b[0 ..< T])], horizon: h)

        let gapA = maxAbsDifference(soloA.median(), joint.median(0)) / standardDeviation(Array(a[0 ..< T]))
        let gapB = maxAbsDifference(soloB.median(), joint.median(1)) / standardDeviation(Array(b[0 ..< T]))
        let truthA = Array(a[T ..< T + h])
        let maeSolo = meanAbsError(soloA.median(), truthA)
        let maeJoint = meanAbsError(joint.median(0), truthA)
        print("[MV2 independence \(model.name)] gap/std A \(gapA), B \(gapB); A MAE solo \(maeSolo) joint \(maeJoint)")

        // Observed (gap = biggest difference between a variate's solo and joint median, as a
        // fraction of that variate's own std; fp32 rows, fp16 within a few percent):
        //
        //                  gap/std A (seasonal)   gap/std B (random walk)   A MAE solo → joint
        //     TimesFM 2.5  0.0                    0.0                       0.0344 → 0.0344
        //     TimesFM 3.0  0.012                  0.231                     0.0179 → 0.0207
        //
        //   * 2.5: exactly 0.0, on both variates. Independence is by construction.
        //   * 3.0 is NOT neutral to unrelated companions. The random walk's forecast moves by up
        //     to 23% of its own std, and the seasonal series' forecast becomes about 16% worse
        //     (MAE 0.0179 → 0.0207), a small but visible cost for adding a series that has
        //     nothing to do with it. 3.0 is still more accurate than 2.5 on A in absolute terms
        //     (0.021 vs 0.034), so the cost is small next to the model's advantage here.
        //   * Hypothesis (not tested here): a random walk moves most because its own history
        //     is ambiguous (a level could continue anywhere), so it is the variate most open to
        //     influence, while a clean seasonal series has a strong own shape to hold on to.
        //   * Practical rule: don't bundle unrelated series into one 3.0 call unless you have
        //     measured that it's harmless for your data. Call them separately, or in groups that
        //     really are related.
        switch model.version {
        case .v25:
            #expect(gapA == 0 && gapB == 0, "2.5 treats variates independently")
        case .v3:
            #expect(gapA < 0.1, "The clean seasonal series should barely move")
            #expect(gapB > 0.05, "3.0 should move the ambiguous random walk")
            #expect(maeJoint < maeSolo * 2, "Mixing should not wreck the seasonal forecast")
        }
    }

    // MARK: 3. Different scales

    @Test("3. Different scales: temperature ≈ 20 and power ≈ 5000 side by side", arguments: TimesFMVariant.available)
    func testDifferentScales(model: TimesFMVariant) throws {
        let forecaster = try model.load()

        // ── The idea. TimesFM rescales each variate on its own (running mean and std), so the
        // *units* of one variate shouldn't leak into another. Temperature swings around 20 by ±5
        // once a day; power swings around 5000 by ±1500 twice a day. We test this two ways:
        //   (a) joint forecast against each variate forecast alone, and
        //   (b) joint forecast with power divided by 250 (so it's ~20 too), then scaled back.
        // If normalisation works per variate, (b) should match (a).
        let T = 240
        let h = 48
        let temperature: [Float] = (0 ..< T + h).map { t in
            Float(20 + 5 * sin(2 * Double.pi * Double(t) / 24))
        }
        let power: [Float] = (0 ..< T + h).map { t in
            Float(5000 + 1500 * sin(2 * Double.pi * Double(t) / 12))
        }
        let joint = forecast(forecaster, [Array(temperature[0 ..< T]), Array(power[0 ..< T])], horizon: h)
        let rescaled = forecast(
            forecaster, [Array(temperature[0 ..< T]), power[0 ..< T].map { $0 / 250 }], horizon: h)
        let soloTemperature = forecast(forecaster, Array(temperature[0 ..< T]), horizon: h)
        let soloPower = forecast(forecaster, Array(power[0 ..< T]), horizon: h)

        let stdT = standardDeviation(Array(temperature[0 ..< T]))
        let stdP = standardDeviation(Array(power[0 ..< T]))
        let unitGapT = maxAbsDifference(joint.median(0), rescaled.median(0)) / stdT
        let unitGapP = maxAbsDifference(joint.median(1), rescaled.median(1).map { $0 * 250 }) / stdP
        let soloGapT = maxAbsDifference(joint.median(0), soloTemperature.median()) / stdT
        let soloGapP = maxAbsDifference(joint.median(1), soloPower.median()) / stdP
        let maeT = meanAbsError(joint.median(0), Array(temperature[T ..< T + h])) / stdT
        let maeP = meanAbsError(joint.median(1), Array(power[T ..< T + h])) / stdP
        print("[MV3 scales \(model.name)] rescale gap/std T \(unitGapT) P \(unitGapP); joint-vs-solo gap/std T \(soloGapT) P \(soloGapP); MAE/std T \(maeT) P \(maeP)")

        // Observed (all gaps are as a fraction of that variate's own std; fp32 rows, fp16 within
        // a few percent):
        //
        //                  rescale gap T / P      joint vs solo gap T / P    MAE/std T / P
        //     TimesFM 2.5  0.0 / 9.2e-7           0.0 / 0.0                  0.0066 / 0.0119
        //     TimesFM 3.0  1.1e-6 / 2.5e-6        0.0094 / 0.0229            0.0050 / 0.0108
        //
        //   * Per-variate normalisation is confirmed on both models: dividing power by 250
        //     (so the two variates are the same size) changes the answers by at most 2.5e-6 of the
        //     std, which is float32 rounding. Units don't leak.
        //   * 2.5's joint equals its solo forecasts exactly. 3.0's joint differs from solo by 1%
        //     (temperature) to 2% (power) of std, small and consistent with experiment 2's
        //     "mixing has a small cost" finding for two clean series.
        //   * Both forecasts are excellent: mean error about 0.5–1.2% of each series' own std.
        //     Two clean sinusoids at different scales and periods are close to the easiest
        //     possible multivariate input.
        #expect(unitGapT < 1e-4 && unitGapP < 1e-4, "Rescaling one variate should not change either forecast")
        #expect(maeT < 0.05 && maeP < 0.05, "Two clean sinusoids should be forecast to within 5% of their std")
        switch model.version {
        case .v25:
            #expect(soloGapT == 0 && soloGapP == 0, "2.5 treats variates independently")
        case .v3:
            #expect(soloGapT < 0.1 && soloGapP < 0.1, "3.0 mixing should be small on clean series")
        }
    }

    // MARK: 4. Conserved total

    @Test("4. Conserved total: the model doesn't know A + B = 100, so derive B from A", arguments: TimesFMVariant.available)
    func testConservedTotal(model: TimesFMVariant) throws {
        let forecaster = try model.load()

        // ── The idea. A is a market share (%) drifting slowly around 30 with a little noise.
        // B = 100 − A is the rest of the market. The two always add to exactly 100, but nothing in
        // the model knows that: it forecasts each variate separately (2.5) or mixed (3.0). So the
        // forecasts A^ + B^ can drift away from 100.
        // The fix is not a smarter model but a smarter *setup*: forecast A only and DERIVE
        // B^ = 100 − A^. The constraint then holds by construction.
        let T = 240
        let h = 60
        var random = SeededRandom(seed: 104)
        let a: [Float] = (0 ..< T + h).map { t in
            let slow: Double = 5 * sin(2 * Double.pi * Double(t) / 120)
            return Float(30 + slow + 0.3 * random.normal())
        }
        let b: [Float] = a.map { 100 - $0 }
        let joint = forecast(forecaster, [Array(a[0 ..< T]), Array(b[0 ..< T])], horizon: h)
        let soloA = forecast(forecaster, Array(a[0 ..< T]), horizon: h)

        let sumGap = zip(joint.median(0), joint.median(1)).map { abs($0 + $1 - 100) }.max()!
        let truthA = Array(a[T ..< T + h])
        let truthB = Array(b[T ..< T + h])
        let jointErrorB = meanAbsError(joint.median(1), truthB)
        let derivedB = soloA.median().map { 100 - $0 }
        let derivedErrorB = meanAbsError(derivedB, truthB)
        let derivedSumGap = zip(soloA.median(), derivedB).map { abs($0 + $1 - 100) }.max()!
        let jointErrorA = meanAbsError(joint.median(0), truthA)
        print("[MV4 total \(model.name)] max |A^+B^−100| \(sumGap); derived gap \(derivedSumGap); A MAE joint \(jointErrorA); B MAE joint \(jointErrorB) vs derived \(derivedErrorB)")

        // Observed (60 steps; A has noise std 0.3, so about 0.24 is the noise floor for MAE;
        // fp32 rows, fp16 within a few percent):
        //
        //                  max |Â + B̂ − 100|   A MAE joint   B MAE joint   B MAE derived
        //     TimesFM 2.5  0.335               0.288         0.264         0.288
        //     TimesFM 3.0  0.084               0.282         0.282         0.284
        //
        //   * Independent 2.5 forecasts drift off the constraint by up to 0.335 (on a total of
        //     100), because nothing ties them together. 3.0's joint forecast is about 4× tighter
        //     (0.084), which is variate attention doing something useful.
        //   * Deriving B̂ = 100 − Â closes the gap exactly (up to float rounding, about 1e-5),
        //     on both models.
        //   * Deriving is a *consistency* fix, not an accuracy one. On 2.5, the derived B is
        //     actually about 9% worse than the independent B forecast (0.288 vs 0.264); on 3.0
        //     it is a wash (0.284 vs 0.282). The noise floor dominates all of these numbers.
        //     Choose "derive" when you need the constraint to hold (money, shares, budgets), not
        //     because you expect a lower error.
        #expect(derivedSumGap < 1e-3, "Deriving B from A satisfies the constraint by construction")
        #expect(derivedErrorB < jointErrorB * 1.3, "Deriving should not cost much accuracy")
        switch model.version {
        case .v25:
            #expect(sumGap > 0.05, "2.5's independent forecasts should visibly violate the constraint")
        case .v3:
            #expect(sumGap < 0.25, "3.0's joint forecast should hold the constraint more tightly")
        }
    }

    // MARK: 5. Order swap

    @Test("5. Order swap: [A, B] vs [B, A]", arguments: TimesFMVariant.available)
    func testOrderSwap(model: TimesFMVariant) throws {
        let forecaster = try model.load()

        // ── The idea. Nothing about "which variate comes first" should matter. If you list your
        // series in a different order, each should get (nearly) the same forecast back. For 2.5
        // this is guaranteed (no mixing at all). For 3.0 the variate attention has no notion of
        // position, so the only difference should be numerical noise.
        let T = 256
        let h = 32
        var random = SeededRandom(seed: 105)
        let a: [Float] = (0 ..< T).map { t in
            let x = Double(t)
            return Float(4 * sin(2 * Double.pi * x / 24) + 0.02 * x)
        }
        let b: [Float] = (0 ..< T).map { t in
            Float(2 * cos(2 * Double.pi * Double(t) / 10) + 0.5 * random.normal())
        }
        let ab = forecast(forecaster, [a, b], horizon: h)
        let ba = forecast(forecaster, [b, a], horizon: h)
        let gapA = maxAbsDifference(ab.median(0), ba.median(1)) / standardDeviation(a)
        let gapB = maxAbsDifference(ab.median(1), ba.median(0)) / standardDeviation(b)
        print("[MV5 order \(model.name)] gap/std A \(gapA), B \(gapB)")

        // Observed (gap as a fraction of the variate's std; fp32 rows, fp16 within a few percent):
        //
        //                  gap/std A      gap/std B
        //     TimesFM 2.5  0.0            0.0
        //     TimesFM 3.0  1.4e-6         8.6e-7
        //
        //   * Order does not matter. 2.5 is exact; 3.0 differs by about a millionth of the std,
        //     which is float32 accumulation order. The variate attention really is
        //     permutation-invariant (no positional encoding across variates).
        //   * Practical upshot: you don't need to worry about how you sort a dashboard's series
        //     before handing them to the model.
        #expect(gapA < 1e-4 && gapB < 1e-4, "Swapping the order should only change float rounding")
    }

    // MARK: 6. Noisy twin

    @Test("6. Noisy twin: does a clean copy sharpen a noisy one?", arguments: TimesFMVariant.available)
    func testNoisyTwin(model: TimesFMVariant) throws {
        let forecaster = try model.load()

        // ── The idea. A is a clean signal. B is the same signal plus heavy noise (std 3, about as
        // big as the signal's own swing). A forecaster looking at B alone must average out the
        // noise; one that also sees A could lean on A's clean shape. We score against the clean
        // continuation, so a low error means "found the signal under the noise".
        let T = 256
        let h = 48
        var random = SeededRandom(seed: 106)
        let clean: [Float] = (0 ..< T + h).map { t in
            let x = Double(t)
            let big: Double = 4 * sin(2 * Double.pi * x / 24)
            let small: Double = 1.5 * sin(2 * Double.pi * x / 7)
            return Float(big + small)
        }
        let noisy: [Float] = clean.map { $0 + Float(3 * random.normal()) }
        let truth = Array(clean[T ..< T + h])

        let alone = forecast(forecaster, Array(noisy[0 ..< T]), horizon: h)
        let together = forecast(forecaster, [Array(clean[0 ..< T]), Array(noisy[0 ..< T])], horizon: h)
        let cleanAlone = forecast(forecaster, Array(clean[0 ..< T]), horizon: h)
        let noisyAlone = meanAbsError(alone.median(), truth)
        let noisyTogether = meanAbsError(together.median(1), truth)
        let cleanError = meanAbsError(cleanAlone.median(), truth)
        let bandAlone = meanBandWidth(alone)
        let bandTogether = meanBandWidth(together, variate: 1)
        print("[MV6 twin \(model.name)] MAE vs clean truth: noisy alone \(noisyAlone), noisy with clean twin \(noisyTogether), clean alone \(cleanError); band width noisy alone \(bandAlone) with twin \(bandTogether)")

        // Observed (MAE against the CLEAN continuation, signal std about 3; fp32 rows, fp16
        // within a few percent):
        //
        //                  noisy alone   noisy + clean twin   clean alone   band width alone → twin
        //     TimesFM 2.5  0.931         0.931                0.043         7.71 → 7.71
        //     TimesFM 3.0  0.568         0.382                0.035         7.75 → 7.24
        //
        //   * This is where variate attention clearly earns its keep. The noisy variate's error
        //     drops from 0.568 to 0.382 (33% lower) with a clean twin beside it, and its
        //     uncertainty band tightens by 7%.
        //   * 2.5 is a control (0.931 both ways, identical bands). It also finds the signal
        //     under the noise worse than 3.0 does even *alone* (0.931 vs 0.568), so part of
        //     the 3.0 advantage is a better single-series model, and the twin is a bonus on top.
        //   * Neither gets near the clean-alone error (0.035–0.043). With noise as big as the
        //     signal, a noisy variate never becomes as good as a clean one; but if you have a
        //     clean version of the same signal, hand it over as a variate on 3.0, or better
        //     yet forecast the clean one and use that.
        #expect(cleanError < 0.2, "The clean signal is easy")
        switch model.version {
        case .v25:
            #expect(abs(noisyAlone - noisyTogether) < 1e-5, "2.5 treats variates independently")
        case .v3:
            #expect(noisyTogether < noisyAlone * 0.85, "3.0 should benefit from the clean twin")
        }
    }
}
