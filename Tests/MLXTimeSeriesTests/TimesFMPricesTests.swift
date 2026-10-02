import Foundation
import MLX
import Testing

@testable import MLXTimeSeries

// MARK: - Rolling forecasts of a real price series (prices.csv)
//
// Every earlier suite used synthetic data with a known answer. This one uses a real series:
// Tests/Documents/prices.csv, 6,183 hourly bars of an FX-style rate (close ≈ 1.18…1.31) for 2023.
// Real prices are close to a random walk, so the honest question is not "is the forecast close?"
// (a forecast of "no change" is always close) but:
//
//     Does the model beat "no change", and does it call the MOVE correctly?
//
// ── Setup (chosen with the user) ───────────────────────────────────────────────────────────
//
//   Horizons   1 day = 24 bars, 5 days = 120 bars. Counted in BARS, not calendar time. The data
//              has a 49-hour gap every weekend, so a window can straddle a weekend; the model sees
//              consecutive bars and never the gap.
//   Context    512 bars of history for every forecast (about three weeks of trading).
//   Rolling    Non-overlapping: forecast the next `horizon` bars, advance by `horizon`, repeat.
//              (So there are about 236 one-day windows and 47 five-day windows.)
//   Targets    (1) `close`: the raw price, forecast directly.
//              (2) `logReturn`: forecast the hourly log returns ln(cₜ/cₜ₋₁), then rebuild the price
//                  path as last known × exp(running sum). This is the "forecast the changes, let
//                  arithmetic accumulate" transform from the Fibonacci and series experiments.
//   Models     The four checkpoints (TimesFM 2.5 fp16/fp32, 3.0 fp32/fp16). All compute in
//              float32; fp16 vs fp32 is only how the weights are stored. The volume column is not
//              used (univariate; the multivariate suites cover variates).
//
// ── What is measured, per model × target × horizon ─────────────────────────────────────────
//
//   MAE          mean |forecast − actual| over every bar of every window (price units).
//   baseline     the same MAE for "last known close, forever". A forecaster that can't beat this
//                adds nothing. `skill` = 1 − MAE/baseline (positive = better than no-change).
//   coverage     share of actual closes inside the 10%…90% band (ideal 0.80). Only for `close`,
//                since a band on a rebuilt return path isn't a simple quantile.
//
//   Price MOVEMENT (the net move over a window = value at the window's last bar − last known):
//   direction    share of windows where the forecast moved the same way (up/down) as the price.
//                `up` is the share of windows where price actually rose, i.e. what "always up"
//                would score, so direction should be read against it, not against 0.5.
//   moveCorr     Pearson correlation, across windows, of forecast net move vs actual net move.
//                0 = the forecast moves tell you nothing about what the price did.
//   moveSize     average |forecast net move| / average |actual net move|. Below 1 means the model
//                predicts smaller moves than really happen (typical, and not a flaw: the median of
//                an unpredictable series is cautious).
//   stepCorr     the same correlation, pooled over every bar-to-bar change inside the windows.
//
// ── Output ─────────────────────────────────────────────────────────────────────────────────
//
// Each model prints one line per target × horizon (`[Prices … ]`), and writes a CSV of every
// forecast to  Tests/Documents/forecasts/prices_<model>.csv  (override the folder with
// `TEST_RUNNER_PRICES_OUT_DIR=/some/dir` on the xcodebuild command line). The CSV has one row per
// forecast bar: model, target, horizon, window, step, close_date (epoch seconds), close_date_pacific
// (ISO 8601 in America/Los_Angeles, with offset), last_known, actual, forecast, low, high, actual_move,
// forecast_move (moves are relative to the last known close).
// A second file, prices_summary.csv in the same folder, has one row per model × target × horizon:
// model, target, horizon, windows, mae, no_change_mae, skill, band_coverage, direction_correct,
// price_rose, move_corr, move_size, step_corr (the same metrics as the printed lines).

/// The hourly bars from prices.csv.
private struct PriceBars {
    let times: [Double]
    let close: [Float]

    static let url = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent()
        .appending(path: "Documents/prices.csv")

    static var exists: Bool { FileManager.default.fileExists(atPath: url.path) }

    static func load() throws -> PriceBars {
        let text = try String(contentsOf: url, encoding: .utf8)
        var times: [Double] = []
        var close: [Float] = []
        for line in text.split(separator: "\n").dropFirst() {
            let fields = line.split(separator: ",")
            guard fields.count >= 2, let t = Double(fields[0]), let c = Float(fields[1]) else { continue }
            times.append(t)
            close.append(c)
        }
        return PriceBars(times: times, close: close)
    }
}

private enum PriceTarget: String, CaseIterable {
    case close
    case logReturn
}

/// One rolling window: what was known, what happened, what the model said.
private struct PriceWindow {
    let times: [Double]
    let lastKnown: Float
    let actual: [Float]
    let median: [Float]
    let low: [Float]?
    let high: [Float]?

    var actualMove: Float { actual.last! - lastKnown }
    var forecastMove: Float { median.last! - lastKnown }
}

private struct PriceMetrics {
    let windows: Int
    let mae: Float
    let baselineMAE: Float
    let coverage: Float?
    let direction: Float
    let up: Float
    let moveCorr: Float
    let moveSize: Float
    let stepCorr: Float
    var skill: Float { 1 - mae / baselineMAE }
}

private func pearson(_ x: [Float], _ y: [Float]) -> Float {
    let n = Float(x.count)
    let mx = x.reduce(0, +) / n
    let my = y.reduce(0, +) / n
    var sxy: Float = 0, sxx: Float = 0, syy: Float = 0
    for (a, b) in zip(x, y) {
        sxy += (a - mx) * (b - my)
        sxx += (a - mx) * (a - mx)
        syy += (b - my) * (b - my)
    }
    return sxy / (sxx * syy).squareRoot()
}

private func metrics(_ windows: [PriceWindow]) -> PriceMetrics {
    var absError: Float = 0, baseError: Float = 0, bars: Float = 0
    var inside: Float = 0
    var forecastSteps: [Float] = [], actualSteps: [Float] = []
    for w in windows {
        for i in 0 ..< w.actual.count {
            absError += abs(w.median[i] - w.actual[i])
            baseError += abs(w.lastKnown - w.actual[i])
            bars += 1
            if let low = w.low, let high = w.high, w.actual[i] >= low[i], w.actual[i] <= high[i] { inside += 1 }
            let previousForecast = i == 0 ? w.lastKnown : w.median[i - 1]
            let previousActual = i == 0 ? w.lastKnown : w.actual[i - 1]
            forecastSteps.append(w.median[i] - previousForecast)
            actualSteps.append(w.actual[i] - previousActual)
        }
    }
    let moved = windows.filter { $0.actualMove != 0 }
    let agree = moved.filter { ($0.forecastMove > 0) == ($0.actualMove > 0) }.count
    let up = moved.filter { $0.actualMove > 0 }.count
    let forecastMoves = windows.map(\.forecastMove)
    let actualMoves = windows.map(\.actualMove)
    let meanAbs = { (x: [Float]) in x.map { abs($0) }.reduce(0, +) / Float(x.count) }
    return PriceMetrics(
        windows: windows.count, mae: absError / bars, baselineMAE: baseError / bars,
        coverage: windows.first?.low == nil ? nil : inside / bars,
        direction: Float(agree) / Float(moved.count), up: Float(up) / Float(moved.count),
        moveCorr: pearson(forecastMoves, actualMoves),
        moveSize: meanAbs(forecastMoves) / meanAbs(actualMoves),
        stepCorr: pearson(forecastSteps, actualSteps))
}

@Suite(
    "TimesFM rolling price forecasts",
    .enabled(if: !TimesFMVariant.available.isEmpty && PriceBars.exists),
    .serialized)
struct TimesFMPricesTests {

    static let context = 512
    static let horizons = [24, 120]
    private static var summaryRows: [String: [String]] = [:]

    /// Forecasts every non-overlapping window of `horizon` bars, each from the previous 512 bars.
    private static func rollingForecasts(
        _ forecaster: TimeSeriesForecaster, bars: PriceBars, target: PriceTarget, horizon: Int
    ) -> [PriceWindow] {
        var windows: [PriceWindow] = []
        // Start at context + 1 so both targets have the same windows (returns need one extra close).
        for start in stride(from: context + 1, through: bars.close.count - horizon, by: horizon) {
            let lastKnown = bars.close[start - 1]
            let actual = Array(bars.close[start ..< start + horizon])
            let times = Array(bars.times[start ..< start + horizon])
            switch target {
            case .close:
                let f = forecast(forecaster, Array(bars.close[start - context ..< start]), horizon: horizon)
                windows.append(
                    PriceWindow(
                        times: times, lastKnown: lastKnown, actual: actual,
                        median: f.median(), low: f.low(), high: f.high()))
            case .logReturn:
                let closes = bars.close[start - context - 1 ..< start].map { Double($0) }
                let returns = (1 ..< closes.count).map { Float(log(closes[$0] / closes[$0 - 1])) }
                let f = forecast(forecaster, returns, horizon: horizon).median()
                var level = Double(lastKnown)
                let path = f.map { r -> Float in
                    level *= exp(Double(r))
                    return Float(level)
                }
                windows.append(
                    PriceWindow(
                        times: times, lastKnown: lastKnown, actual: actual, median: path, low: nil, high: nil))
            }
        }
        return windows
    }

    @Test("Rolling 1-day and 5-day forecasts of prices.csv", arguments: TimesFMVariant.available)
    func testRollingPrices(model: TimesFMVariant) throws {
        let forecaster = try model.load()
        let bars = try PriceBars.load()
        var csv = "model,target,horizon,window,step,close_date,close_date_pacific,last_known,actual,forecast,low,high,actual_move,forecast_move\n"
        var results: [String: PriceMetrics] = [:]
        var summary: [String] = []
        let pacific = ISO8601DateFormatter()
        pacific.timeZone = TimeZone(identifier: "America/Los_Angeles")!

        for target in PriceTarget.allCases {
            for horizon in Self.horizons {
                let windows = Self.rollingForecasts(forecaster, bars: bars, target: target, horizon: horizon)
                let m = metrics(windows)
                results["\(target.rawValue)-\(horizon)"] = m
                summary.append([
                    model.name, target.rawValue, "\(horizon)", "\(m.windows)", "\(m.mae)", "\(m.baselineMAE)",
                    "\(m.skill)", m.coverage.map { "\($0)" } ?? "", "\(m.direction)", "\(m.up)",
                    "\(m.moveCorr)", "\(m.moveSize)", "\(m.stepCorr)",
                ].joined(separator: ","))
                let band = m.coverage.map { "\($0)" } ?? "n/a"
                print("[Prices \(target.rawValue) \(horizon) bars \(model.name)] windows \(m.windows), MAE \(m.mae) vs last-value \(m.baselineMAE) (skill \(m.skill)), coverage \(band); net move: direction \(m.direction) (price rose in \(m.up)), moveCorr \(m.moveCorr), moveSize \(m.moveSize); stepCorr \(m.stepCorr)")

                for (w, window) in windows.enumerated() {
                    for step in 0 ..< horizon {
                        let low = window.low.map { "\($0[step])" } ?? ""
                        let high = window.high.map { "\($0[step])" } ?? ""
                        csv += "\(model.name),\(target.rawValue),\(horizon),\(w),\(step),\(Int(window.times[step])),\(pacific.string(from: Date(timeIntervalSince1970: window.times[step]))),\(window.lastKnown),\(window.actual[step]),\(window.median[step]),\(low),\(high),\(window.actual[step] - window.lastKnown),\(window.median[step] - window.lastKnown)\n"
                    }
                }
            }
        }

        let outputDirectory = ProcessInfo.processInfo.environment["PRICES_OUT_DIR"].map { URL(fileURLWithPath: $0) }
            ?? PriceBars.url.deletingLastPathComponent().appending(path: "forecasts")
        try FileManager.default.createDirectory(at: outputDirectory, withIntermediateDirectories: true)
        let file = outputDirectory.appending(path: "prices_\(model.name.replacingOccurrences(of: " ", with: "_")).csv")
        try csv.write(to: file, atomically: true, encoding: .utf8)
        print("[Prices \(model.name)] wrote \(file.path)")

        Self.summaryRows[model.name] = summary
        let summaryHeader = "model,target,horizon,windows,mae,no_change_mae,skill,band_coverage,direction_correct,price_rose,move_corr,move_size,step_corr\n"
        let summaryFile = outputDirectory.appending(path: "prices_summary.csv")
        let summaryText = summaryHeader + Self.summaryRows.keys.sorted().flatMap { Self.summaryRows[$0]! }.joined(separator: "\n") + "\n"
        try summaryText.write(to: summaryFile, atomically: true, encoding: .utf8)
        print("[Prices \(model.name)] wrote \(summaryFile.path)")

        // Observed (fp32 rows; fp16 agrees to about 3 significant digits in MAE and within 0.002
        // in the correlations; windows: 236 one-day, 47 five-day; price rose in 52.5% of one-day
        // and 57.4% of five-day windows; "no change" MAE is 0.00383 at 24 bars and 0.00704 at 120):
        //
        //                         MAE (skill vs no-change)   coverage   direction   moveCorr   moveSize   stepCorr
        //   2.5 close    24 bars  0.00390 (−1.8%)            0.70       0.449       +0.004     0.25       −0.007
        //   2.5 close   120 bars  0.00781 (−10.9%)           0.68       0.532       +0.060     0.43       +0.000
        //   2.5 logRet   24 bars  0.00397 (−3.7%)            n/a        0.513       −0.059     0.29       −0.040
        //   2.5 logRet  120 bars  0.00744 (−5.6%)            n/a        0.574       −0.043     0.66       −0.045
        //   3.0 close    24 bars  0.00395 (−3.1%)            0.71       0.479       −0.011     0.29       −0.020
        //   3.0 close   120 bars  0.00830 (−17.8%)           0.72       0.511       −0.242     0.56       −0.019
        //   3.0 logRet   24 bars  0.00393 (−2.7%)            n/a        0.453       −0.114     0.23       −0.038
        //   3.0 logRet  120 bars  0.00743 (−5.5%)            n/a        0.489       −0.105     0.55       −0.026
        //
        //   * Neither model beats "no change", at either horizon, on either target. Every skill is
        //     negative: the forecasts are 2–18% worse than repeating the last close. That is what a
        //     near-random-walk looks like, and it is the same lesson as experiment 9 (random walk)
        //     and 3 (digits of π): there is no pattern in the history to continue.
        //   * Their MOVE calls are no better than a coin flip. Direction hits 45–57% against
        //     "price rose" in 52–57% of windows, so always guessing "up" would have matched or beaten
        //     most rows. With 47 windows a correlation needs to reach about ±0.29 to stand out from
        //     chance (2 standard errors), and with 236 windows about ±0.13. Every correlation here is
        //     inside that band (the largest, 3.0 close at 120 bars, is −0.24), so the forecast move
        //     carries no usable information about the actual move. The slightly negative values would
        //     be a small mean-reversion mismatch, but they are indistinguishable from chance.
        //   * moveSize is below 1 everywhere (0.23–0.66): the models forecast far smaller moves
        //     than actually happen. For an unpredictable series that is the cautious, correct
        //     median, and it is why the forecasts are close to "no change".
        //   * The bands are about right: 70–72% of actual closes land in the 10%…90% band
        //     (nominal 80%), slightly overconfident, 2.5 more than 3.0 at 120 bars (0.68 vs 0.72).
        //     So the model's uncertainty is a more useful output here than its median.
        //   * Transform: log returns are better than raw close at 120 bars (−5.6% vs −10.9% on
        //     2.5, −5.5% vs −17.8% on 3.0) and about equal at 24 bars. That's the "forecast the
        //     changes" recipe helping, but only by being less wrong.
        //   * Caveats: one year, one instrument, weekends inside windows, a single context length
        //     (512) and a single windowing; none of this says what a different setup would do.
        //     These are descriptive measurements, not trading advice.
        for (key, m) in results {
            #expect(m.mae.isFinite && m.baselineMAE > 0, "\(key): metrics should be finite")
            #expect(m.skill < 0.1, "\(key): beating no-change by 10% on near-random prices would be surprising")
            #expect(abs(m.moveCorr) < 0.5, "\(key): forecast moves should not strongly track actual moves")
            #expect(m.moveSize < 1, "\(key): forecast moves should be smaller than actual moves")
            if let coverage = m.coverage {
                #expect(coverage > 0.55 && coverage < 0.9, "\(key): 80% band coverage should be roughly honest")
            }
        }
    }
}
