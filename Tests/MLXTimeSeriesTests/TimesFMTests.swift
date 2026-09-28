import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXTimeSeries

@Suite("TimesFM Component Tests")
struct TimesFMTests {

    /// Small config so tests run fast with random weights. 2 input patches per output patch.
    let tinyConfig = TimesFMConfiguration(
        hiddenSize: 32, numLayers: 2, numHeads: 2, intermediateSize: 32, headDim: 16,
        patchLength: 4, outputPatchLength: 8, quantileHorizonLength: 16, numQuantiles: 9,
        contextLength: 256, predictionLength: 8
    )

    /// Largest elementwise |a - b|, as a scalar.
    private func maxAbsDiff(_ a: MLXArray, _ b: MLXArray) -> Float {
        MLX.abs(a - b).max().item(Float.self)
    }

    // MARK: - Configuration

    @Test("Configuration default values match TimesFM 2.5 200M")
    func testDefaultConfig() {
        let config = TimesFMConfiguration()
        #expect(config.hiddenSize == 1280)
        #expect(config.numLayers == 20)
        #expect(config.numHeads == 16)
        #expect(config.headDim == 80)
        #expect(config.numHeads * config.headDim == config.hiddenSize)
        #expect(config.intermediateSize == 1280)
        #expect(config.patchLength == 32)
        #expect(config.outputPatchLength == 128)
        #expect(config.quantileHorizonLength == 1024)
        #expect(config.numQuantiles == 9)
        #expect(config.decodeIndex == 5)
        #expect(config.contextLength == 16384)
    }

    @Test("Configuration decodes input_patch_len")
    func testConfigDecoding() throws {
        let json = """
            {"model_type": "timesfm", "hidden_size": 1280, "num_layers": 20,
             "input_patch_len": 32, "output_patch_len": 128, "rope_theta": 10000.0}
            """
        let decoded = try JSONDecoder().decode(
            TimesFMConfiguration.self, from: json.data(using: .utf8)!)
        #expect(decoded.patchLength == 32)
        #expect(decoded.outputPatchLength == 128)
        #expect(decoded.decodeIndex == 5)
    }

    @Test("Legacy config.json (patch_length = tokenizer width) decodes to patch 32")
    func testLegacyConfigDecoding() throws {
        // As published in kunal732/timesfm-2.5-200m-transformers-mlx.
        let json = """
            {
                "model_type": "timesfm", "ts_model_class": "TimesFMModel",
                "hidden_size": 1280, "num_layers": 20, "num_heads": 16,
                "intermediate_size": 1280, "head_dim": 80, "patch_length": 64,
                "quantile_horizon_length": 1024, "num_quantiles": 9,
                "context_length": 16384, "prediction_length": 128,
                "use_positional_encoding": false, "query_pre_attn_scalar": 256.0,
                "use_rope": true, "rope_theta": 10000.0, "use_horizon_ff": false
            }
            """
        let decoded = try JSONDecoder().decode(
            TimesFMConfiguration.self, from: json.data(using: .utf8)!)
        #expect(decoded.patchLength == 32)
        #expect(decoded.outputPatchLength == 128)
        #expect(decoded.hiddenSize == 1280)
    }

    // MARK: - Components

    @Test("Residual block maps inputs to output size")
    func testResidualBlockShape() {
        let block = TimesFMResidualBlock(inputDim: 8, outputDim: 32)
        #expect(block(MLXRandom.normal([2, 5, 8])).shape == [2, 5, 32])
    }

    @Test("Attention preserves shape and is finite")
    func testAttentionShape() {
        let attn = TimesFMAttention(tinyConfig)
        let out = attn(MLXRandom.normal([2, 6, 32]), numMasked: MLXArray.zeros([2], dtype: .int32))
        #expect(out.shape == [2, 6, 32])
        #expect(MLX.isNaN(out).any().item(Bool.self) == false)
    }

    @Test("Attention is causal: later tokens don't change earlier outputs")
    func testAttentionCausal() {
        let attn = TimesFMAttention(tinyConfig)
        let x = MLXRandom.normal([1, 6, 32])
        let none = MLXArray.zeros([1], dtype: .int32)
        let full = attn(x, numMasked: none)
        let prefix = attn(x[0..., 0 ..< 4, 0...], numMasked: none)
        #expect(maxAbsDiff(full[0..., 0 ..< 4, 0...], prefix) < 1e-4)
    }

    @Test("Padded leading patches are ignored and shift RoPE positions")
    func testAttentionIgnoresMaskedPatches() {
        let attn = TimesFMAttention(tinyConfig)
        let x = MLXRandom.normal([1, 4, 32])
        let junk = MLXRandom.normal([1, 2, 32]) * 100
        let clean = attn(x, numMasked: MLXArray([Int32(0)]))
        let padded = attn(MLX.concatenated([junk, x], axis: 1), numMasked: MLXArray([Int32(2)]))
        #expect(maxAbsDiff(padded[0..., 2...], clean) < 1e-4)
    }

    @Test("Cached decode step matches full forward (RoPE offset)")
    func testAttentionKVCacheMatchesFull() {
        let attn = TimesFMAttention(tinyConfig)
        let x = MLXRandom.normal([1, 6, 32])
        let none = MLXArray.zeros([1], dtype: .int32)
        let full = attn(x, numMasked: none)

        let cache = TimeSeriesKVCache()
        _ = attn(x[0..., 0 ..< 4, 0...], numMasked: none, cache: cache)
        let step = attn(x[0..., 4 ..< 6, 0...], numMasked: none, cache: cache)

        #expect(cache.offset == 6)
        #expect(maxAbsDiff(full[0..., 4 ..< 6, 0...], step) < 1e-4)
    }

    @Test("Transformer block preserves shape")
    func testTransformerBlockShape() {
        let block = TimesFMTransformerBlock(tinyConfig)
        let out = block(
            MLXRandom.normal([3, 4, 32]), numMasked: MLXArray.zeros([3], dtype: .int32),
            cache: nil)
        #expect(out.shape == [3, 4, 32])
    }

    @Test("Running stats match a naive per-patch computation, incl. large offsets")
    func testRunningStats() {
        let values: [Float] = (0 ..< 12).map { 10_000 + Float($0 * $0) }
        let mask: [Float] = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  // first two padded
        let x = MLXArray(values).reshaped(1, 3, 4)
        let m = MLXArray(mask).reshaped(1, 3, 4)

        // Split in two calls to also exercise carrying prior stats forward.
        let (mu1, sigma1, s1) = TimesFMRunningStats.empty(1)
            .accumulate(x[0..., 0 ..< 1], padMask: m[0..., 0 ..< 1])
        let (mu2, sigma2, _) = s1.accumulate(x[0..., 1...], padMask: m[0..., 1...])
        let mu = MLX.concatenated([mu1, mu2], axis: 1)
        let sigma = MLX.concatenated([sigma1, sigma2], axis: 1)

        for k in 0 ..< 3 {
            let seen = (0 ..< (4 * (k + 1))).filter { mask[$0] == 0 }.map { Double(values[$0]) }
            let mean = seen.reduce(0, +) / Double(seen.count)
            let std = (seen.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / Double(seen.count))
                .squareRoot()
            #expect(abs(Double(mu[0, k].item(Float.self)) - mean) < 1e-2)
            #expect(abs(Double(sigma[0, k].item(Float.self)) - std) < 1e-2)
        }
    }

    // MARK: - Model

    @Test("Parameter keys match the TimesFM 2.5 checkpoint layout")
    func testParameterKeys() {
        let keys = Set(TimesFMModel(tinyConfig).parameters().flattened().map { $0.0 })

        #expect(keys.contains("tokenizer.hidden_layer.bias"))
        #expect(keys.contains("stacked_xf.0.attn.qkv_proj.weight"))
        #expect(keys.contains("stacked_xf.0.attn.out.weight"))
        #expect(keys.contains("stacked_xf.0.attn.per_dim_scale.per_dim_scale"))
        #expect(keys.contains("stacked_xf.0.attn.query_ln.weight"))
        #expect(keys.contains("stacked_xf.1.pre_attn_ln.weight"))
        #expect(keys.contains("stacked_xf.1.ff0.weight"))
        #expect(keys.contains("output_projection_point.hidden_layer.weight"))
        #expect(keys.contains("output_projection_quantiles.output_layer.weight"))
        #expect(!keys.contains { $0.hasPrefix("horizon_ff_layer") })
        #expect(!keys.contains { $0.hasPrefix("stacked_xf.2.") })
    }

    @Test("Tokenizer takes values + mask; heads emit horizon × (mean + quantiles)")
    func testWeightShapes() {
        let params = Dictionary(
            uniqueKeysWithValues: TimesFMModel(tinyConfig).parameters().flattened())
        let c = tinyConfig
        let channels = c.numQuantiles + 1

        #expect(params["tokenizer.hidden_layer.weight"]?.shape == [c.hiddenSize, 2 * c.patchLength])
        #expect(params["stacked_xf.0.attn.qkv_proj.weight"]?.shape == [3 * c.hiddenSize, c.hiddenSize])
        #expect(params["output_projection_point.output_layer.weight"]?.shape == [c.outputPatchLength * channels, c.hiddenSize])
        #expect(params["output_projection_quantiles.output_layer.weight"]?.shape == [c.quantileHorizonLength * channels, c.hiddenSize])
    }

    @Test("Forecast shapes with non-patch-aligned, multivariate context")
    func testForecastShape() {
        let model = TimesFMModel(tinyConfig)
        let T = 29  // not a multiple of patchLength (4) → left-padded
        let input = TimeSeriesInput(
            series: MLXRandom.normal([2, 3, T]),
            paddingMask: MLXArray.ones([2, 3, T]),
            idMask: MLXArray.zeros([2, 3])
        )
        let pred = model.forecast(input: input, predictionLength: 8, caches: model.newCaches())
        #expect(pred.mean.shape == [2, 3, 8])
        #expect(pred.quantiles?.shape == [2, 3, 8, 9])
        #expect(MLX.isNaN(pred.mean).any().item(Bool.self) == false)
    }

    @Test("Horizons beyond one output patch are decoded autoregressively")
    func testForecastAutoregressive() {
        let model = TimesFMModel(tinyConfig)
        let input = TimeSeriesInput.univariate(MLXRandom.normal([32]))
        let long = model.forecast(input: input, predictionLength: 20, caches: model.newCaches())
        let short = model.forecast(input: input, predictionLength: 8, caches: model.newCaches())

        #expect(long.predictionLength == 20)
        #expect(long.mean.shape == [1, 1, 20])
        // The first output patch comes from the prefill pass either way.
        #expect(maxAbsDiff(long.mean[0..., 0..., 0 ..< 8], short.mean) < 1e-4)
    }

    @Test("newCaches returns one cache per layer")
    func testNewCaches() {
        #expect(TimesFMModel(tinyConfig).newCaches().count == tinyConfig.numLayers)
    }

    @Test("sanitize renames norm scale, keeps per_dim_scale, drops horizon_ff_layer")
    func testSanitize() {
        let model = TimesFMModel(tinyConfig)
        let w = MLXArray.ones([4])
        let out = model.sanitize(weights: [
            "stacked_xf.0.pre_attn_ln.scale": w,
            "stacked_xf.0.attn.query_ln.scale": w,
            "stacked_xf.0.attn.per_dim_scale.per_dim_scale": w,
            "tokenizer.hidden_layer.weight": w,
            "horizon_ff_layer.hidden_layer.weight": w,
        ])
        #expect(Set(out.keys) == [
            "stacked_xf.0.pre_attn_ln.weight",
            "stacked_xf.0.attn.query_ln.weight",
            "stacked_xf.0.attn.per_dim_scale.per_dim_scale",
            "tokenizer.hidden_layer.weight",
        ])
    }
}

// MARK: - Golden values vs. upstream reference

/// Local TimesFM 2.5 checkpoint. Override with TIMESFM25_CHECKPOINT; tests skip if absent.
///
/// Precision: fp16. This is kunal732's MLX conversion, which stores every weight as float16,
/// rounded from Google's float32 release. The model still computes in float32
/// (`TimesFMModel.inferenceDtype`).
let timesFM25Checkpoint: URL = {
    if let path = ProcessInfo.processInfo.environment["TIMESFM25_CHECKPOINT"] {
        return URL(fileURLWithPath: path)
    }
    return FileManager.default.homeDirectoryForCurrentUser.appending(
        path: ".cache/huggingface/hub/models--kunal732--timesfm-2.5-200m-transformers-mlx/"
            + "snapshots/98aa3b03fc637a1f57386b59c3f38b054b1a251a/model.safetensors")
}()

/// Generated by Scripts/timesfm25_golden.py from google-research/timesfm's torch
/// `forecast_naive`, using this checkpoint's fp16 weights (converted to float32). So these
/// tests check the Swift code against Google's code, not against Google's exact fp32 weights.
private let timesFM25Golden = URL(fileURLWithPath: #filePath)
    .deletingLastPathComponent()
    .appending(path: "Fixtures/timesfm25_golden.json")

/// Config exactly as published in kunal732/timesfm-2.5-200m-transformers-mlx (legacy keys).
private let timesFM25HubConfig = """
    {"model_type": "timesfm", "ts_model_class": "TimesFMModel", "hidden_size": 1280,
     "num_layers": 20, "num_heads": 16, "intermediate_size": 1280, "head_dim": 80,
     "patch_length": 64, "quantile_horizon_length": 1024, "num_quantiles": 9,
     "context_length": 16384, "prediction_length": 128, "use_positional_encoding": false,
     "query_pre_attn_scalar": 256.0, "use_rope": true, "rope_theta": 10000.0,
     "use_horizon_ff": false}
    """

/// Loads the local checkpoint the same way ModelArena does: a directory holding
/// `config.json` + `model.safetensors`, passed to `TimeSeriesForecaster.loadFromDirectory`.
/// Weights are evaluated before the temporary directory is removed.
func loadTimesFM25Forecaster() throws -> TimeSeriesForecaster {
    let dir = FileManager.default.temporaryDirectory
        .appending(path: "timesfm25-test-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: dir) }
    try timesFM25HubConfig.write(
        to: dir.appending(path: "config.json"), atomically: true, encoding: .utf8)
    try FileManager.default.createSymbolicLink(
        at: dir.appending(path: "model.safetensors"), withDestinationURL: timesFM25Checkpoint)
    let forecaster = try TimeSeriesForecaster.loadFromDirectory(dir)
    eval(forecaster.model)
    return forecaster
}

private struct TimesFMGolden: Decodable {
    struct Case: Decodable {
        let name: String
        let horizon: Int
        let input: [Float]
        let forecast: [[Float]]  // [horizon, 10]
    }
    let cases: [Case]
}

@Suite(
    "TimesFM 2.5 golden values",
    .enabled(if: FileManager.default.fileExists(atPath: timesFM25Checkpoint.path)),
    .serialized)
struct TimesFMGoldenTests {

    @Test("Checkpoint loads with every parameter present")
    func testCheckpointLoadsStrictly() throws {
        let model = TimesFMModel(TimesFMConfiguration())
        let weights = model.sanitize(weights: try loadArrays(url: timesFM25Checkpoint))
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: [.all])
    }

    @Test("Forecasts match the upstream torch reference (loaded like ModelArena)")
    func testMatchesReference() throws {
        let forecaster = try loadTimesFM25Forecaster()

        let golden = try JSONDecoder().decode(
            TimesFMGolden.self, from: Data(contentsOf: timesFM25Golden))
        for c in golden.cases {
            let pred = forecaster.forecast(
                input: .univariate(MLXArray(c.input)), predictionLength: c.horizon)
            let expected = MLXArray(c.forecast.flatMap { $0 }).reshaped(c.horizon, 10)
            let std = MLX.std(MLXArray(c.input)).item(Float.self)

            // Channels 1...9 are the quantiles; the median (channel 5) is the point forecast.
            let quantileErr = MLX.abs(
                pred.quantiles!.reshaped(c.horizon, 9) - expected[0..., 1...]
            ).max().item(Float.self) / std
            let medianErr = MLX.abs(pred.mean.reshaped(c.horizon) - expected[0..., 5])
                .max().item(Float.self) / std

            #expect(pred.mean.shape == [1, 1, c.horizon])
            #expect(quantileErr < 1e-3, "\(c.name): quantile err/std \(quantileErr)")
            #expect(medianErr < 1e-3, "\(c.name): median err/std \(medianErr)")
        }
    }
}
