import Foundation
import MLX
import Testing

@testable import MLXTimeSeries

/// The four TimesFM checkpoints the walkthrough tests compare: 2 versions × 2 precisions.
///
/// | Variant          | Weights                                              | Stored | Computed |
/// |------------------|------------------------------------------------------|--------|----------|
/// | TimesFM 2.5 fp16 | kunal732/timesfm-2.5-200m-transformers-mlx (HF cache) | fp16   | fp32     |
/// | TimesFM 2.5 fp32 | converted/timesfm25-fp32                             | fp32   | fp32     |
/// | TimesFM 3.0 fp32 | converted/timesfm3-fp32                              | fp32   | fp32     |
/// | TimesFM 3.0 fp16 | converted/timesfm3-fp16                              | fp16   | fp32     |
///
/// Each fp16 file is exactly its fp32 sibling rounded to float16 (checked tensor by tensor), and
/// both versions compute in float32. So within a version, any difference between fp16 and fp32
/// is purely the one-time rounding of the stored weights.
///
/// Make the converted ones with:
///
///     python Scripts/convert_ts_model.py --hf-path google/timesfm-2.5-200m-pytorch \
///         --model-type timesfm --mlx-path converted/timesfm25-fp32 --dtype float32
///     python Scripts/convert_ts_model.py --hf-path google/timesfm-3.0-pytorch \
///         --mlx-path converted/timesfm3-fp32 --dtype float32
///     python Scripts/convert_ts_model.py --hf-path google/timesfm-3.0-pytorch \
///         --mlx-path converted/timesfm3-fp16
struct TimesFMVariant: CustomTestStringConvertible, Sendable {
    enum Version: String, Sendable {
        case v25 = "2.5"
        case v3 = "3.0"
    }

    let version: Version
    /// "fp16" or "fp32": how the weights are *stored*. Both compute in float32.
    let precision: String
    /// The weights file; the variant is skipped when it's missing.
    let weightsFile: URL
    private let loader: @Sendable () throws -> TimeSeriesForecaster

    var name: String { "TimesFM \(version.rawValue) \(precision)" }
    var testDescription: String { name }
    var isAvailable: Bool { FileManager.default.fileExists(atPath: weightsFile.path) }

    /// Loads the model the way an app would (config.json + weights via the model registry).
    func load() throws -> TimeSeriesForecaster { try loader() }

    /// `converted/<folder>` under the repo root.
    private static func converted(_ folder: String) -> URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appending(path: "converted/\(folder)")
    }

    static let v25fp16 = TimesFMVariant(
        version: .v25, precision: "fp16", weightsFile: timesFM25Checkpoint,
        loader: { try loadTimesFM25Forecaster() })

    static let v25fp32: TimesFMVariant = {
        let dir = converted("timesfm25-fp32")
        return TimesFMVariant(
            version: .v25, precision: "fp32", weightsFile: dir.appending(path: "model.safetensors"),
            loader: { try TimeSeriesForecaster.loadFromDirectory(dir) })
    }()

    static let v3fp32 = TimesFMVariant(
        version: .v3, precision: "fp32", weightsFile: TimesFM3Weights.fp32.checkpoint,
        loader: { try TimeSeriesForecaster.loadFromDirectory(TimesFM3Weights.fp32.directory) })

    static let v3fp16 = TimesFMVariant(
        version: .v3, precision: "fp16", weightsFile: TimesFM3Weights.fp16.checkpoint,
        loader: { try TimeSeriesForecaster.loadFromDirectory(TimesFM3Weights.fp16.directory) })

    static let all = [v25fp16, v25fp32, v3fp32, v3fp16]

    /// The variants whose weights are on this machine.
    static let available = all.filter(\.isAvailable)
}
