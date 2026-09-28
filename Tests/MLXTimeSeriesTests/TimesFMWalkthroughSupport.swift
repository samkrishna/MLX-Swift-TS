import Foundation
import MLX

@testable import MLXTimeSeries

/// Shared helpers for the walkthrough tests (Fibonacci, intuition examples).

/// A tiny deterministic random number generator (SplitMix64), so every run and every model
/// sees exactly the same "random" data. Swift's own generators are seeded differently per run.
struct SeededRandom {
    private var state: UInt64

    init(seed: UInt64) { state = seed }

    /// Uniform in [0, 1).
    mutating func uniform() -> Double {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        z ^= z >> 31
        return Double(z >> 11) / Double(1 << 53)
    }

    /// Standard normal (mean 0, std 1), via Box-Muller.
    mutating func normal() -> Double {
        let u1 = max(uniform(), 1e-12)
        let u2 = uniform()
        return (-2 * log(u1)).squareRoot() * cos(2 * .pi * u2)
    }
}

/// A forecast as plain Swift arrays: `quantiles[variate][step][q]`, q = 0...8 for the
/// 10%...90% levels. `median(v)` is q = 4.
struct Forecast {
    let quantiles: [[[Float]]]

    func median(_ variate: Int = 0) -> [Float] { quantiles[variate].map { $0[4] } }
    func low(_ variate: Int = 0) -> [Float] { quantiles[variate].map { $0[0] } }  // 10%
    func high(_ variate: Int = 0) -> [Float] { quantiles[variate].map { $0[8] } }  // 90%
}

/// Forecasts `horizon` steps for one series made of one or more variates (all the same length).
///
/// The input becomes a `[1, V, T]` tensor with every value marked observed. TimesFM 2.5 treats
/// the V variates as unrelated series; TimesFM 3.0 lets them exchange information.
func forecast(_ forecaster: TimeSeriesForecaster, _ variates: [[Float]], horizon: Int) -> Forecast {
    let V = variates.count
    let T = variates[0].count
    let input = TimeSeriesInput(
        series: MLXArray(variates.flatMap { $0 }).reshaped(1, V, T),
        paddingMask: MLXArray.ones([1, V, T]),
        idMask: MLXArray.zeros([1, V], type: Int32.self))
    let q = forecaster.forecast(input: input, predictionLength: horizon).quantiles!
    // Quantiles exactly as the model returns them. TimesFM 3.0 sorts them. TimesFM 2.5 returns
    // its 9 quantile channels as-is, and they can occasionally cross (e.g. the "50%" channel
    // ending up above the "60%" one). Leaving them unsorted keeps `median` equal to
    // `prediction.mean`, the model's own point forecast.
    let flat = q.reshaped(V, horizon, 9).asArray(Float.self)
    return Forecast(
        quantiles: (0 ..< V).map { v in
            (0 ..< horizon).map { h in Array(flat[(v * horizon + h) * 9 ..< (v * horizon + h + 1) * 9]) }
        })
}

/// Convenience for a single series.
func forecast(_ forecaster: TimeSeriesForecaster, _ series: [Float], horizon: Int) -> Forecast {
    forecast(forecaster, [series], horizon: horizon)
}

/// Mean absolute error between two equal-length arrays.
func meanAbsError(_ a: [Float], _ b: [Float]) -> Float {
    zip(a, b).map { abs($0 - $1) }.reduce(0, +) / Float(a.count)
}

/// Population standard deviation.
func standardDeviation(_ x: [Float]) -> Float {
    let mean = x.reduce(0, +) / Float(x.count)
    return (x.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / Float(x.count)).squareRoot()
}
