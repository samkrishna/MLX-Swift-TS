import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXTimeSeries

@Suite("TimesFM 3 Component Tests")
struct TimesFM3Tests {

    /// Small config so tests run fast with random weights. Keeps upstream's 2 rolls per output.
    let tinyConfig = TimesFM3Configuration(
        inputPatchLength: 4, outputPatchLength: 8, modelDims: 32, hiddenDims: 32,
        numLayers: 2, numHeads: 2, maxContextLength: 64)

    private func maxAbsDiff(_ a: MLXArray, _ b: MLXArray) -> Float {
        MLX.abs(a - b).max().item(Float.self)
    }

    // MARK: - Configuration

    @Test("Configuration defaults match TimesFM 3.0")
    func testDefaultConfig() {
        let c = TimesFM3Configuration()
        #expect(c.inputPatchLength == 32)
        #expect(c.outputPatchLength == 64)
        #expect(c.modelDims == 1280)
        #expect(c.headDim == 80)
        #expect(c.numLayers == 20)
        #expect(c.numQuantiles == 9)
        #expect(c.medianIndex == 4)
        #expect(c.rolls == 2)
        #expect(c.maxContextLength == 15360)
    }

    @Test("Converter's config.json decodes and routes to TimesFM3Model")
    func testConfigDecoding() throws {
        // As written by Scripts/convert_ts_model.py for google/timesfm-3.0-pytorch.
        let json = """
            {"model_type": "timesfm3", "ts_model_class": "TimesFM3Model",
             "input_patch_len": 32, "output_patch_len": 64, "model_dims": 1280,
             "hidden_dims": 1280, "num_layers": 20, "num_heads": 16,
             "quantiles": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
             "use_variate_attention": true, "use_linear_detrending": true,
             "linear_detrending_threshold": 0.5, "value_clip": 1e+20, "max_variates": 32,
             "max_context_length": 15360}
            """.data(using: .utf8)!
        let decoded = try JSONDecoder().decode(TimesFM3Configuration.self, from: json)
        #expect(decoded.numHeads == 16)
        #expect(decoded.linearDetrendingThreshold == 0.5)
        #expect(decoded.valueClip == 1e20)

        let (model, base) = try TimeSeriesTypeRegistry.shared.load(configData: json)
        #expect(base.modelType == "timesfm3")
        #expect(model is TimesFM3Model)
    }

    // MARK: - Components

    @Test("Sequence attention is causal")
    func testSeqAttentionCausal() {
        let attn = TimesFM3Attention(tinyConfig, useRoPE: true, causal: true)
        let x = MLXRandom.normal([2, 6, 32])
        let noMask = MLXArray.zeros([2, 6], type: Bool.self)
        let full = attn(x, keyMasked: noMask)
        let prefix = attn(x[0..., 0 ..< 4], keyMasked: noMask[0..., 0 ..< 4])
        #expect(maxAbsDiff(full[0..., 0 ..< 4], prefix) < 1e-5)
    }

    @Test("Masked keys are ignored")
    func testAttentionIgnoresMaskedKeys() {
        let attn = TimesFM3Attention(tinyConfig, useRoPE: false, causal: false)
        let x = MLXRandom.normal([1, 5, 32])
        let mask = MLXArray([true, false, false, false, false]).reshaped(1, 5)
        let a = attn(x, keyMasked: mask)
        let changed = MLX.concatenated([MLXRandom.normal([1, 1, 32]), x[0..., 1...]], axis: 1)
        let b = attn(changed, keyMasked: mask)
        #expect(maxAbsDiff(a[0..., 1...], b[0..., 1...]) < 1e-5)
    }

    @Test("Variate attention is permutation-equivariant across variates")
    func testVariateAttentionEquivariant() {
        let attn = TimesFM3Attention(tinyConfig, useRoPE: false, causal: false)
        let x = MLXRandom.normal([1, 3, 32])
        let noMask = MLXArray.zeros([1, 3], type: Bool.self)
        let perm = MLXArray([2, 0, 1] as [Int32])
        let a = attn(x, keyMasked: noMask).take(perm, axis: 1)
        let b = attn(x.take(perm, axis: 1), keyMasked: noMask)
        #expect(maxAbsDiff(a, b) < 1e-5)
    }

    @Test("Single-key shortcut equals the value path")
    func testSingleKeyShortcut() {
        let attn = TimesFM3Attention(tinyConfig, useRoPE: false, causal: false)
        let x = MLXRandom.normal([4, 1, 32])
        let out = attn(x, keyMasked: MLXArray.zeros([4, 1], type: Bool.self))
        #expect(maxAbsDiff(out, attn.outProj(attn.valueProj(x))) < 1e-6)
    }

    @Test("Running stats match a naive per-patch computation")
    func testRunningStats() {
        let model = TimesFM3Model(tinyConfig)
        let values = MLXRandom.normal([1, 1, 3, 4]) * 5 + 1000
        let masked = MLXArray([true, true, false, false] + Array(repeating: false, count: 8))
            .reshaped(1, 1, 3, 4)
        let stats = model.runningStats(values, masked: masked)

        let flat = values.reshaped(-1)
        let observed: [[Int]] = [[2, 3], [2, 3, 4, 5, 6, 7], Array(2 ..< 12)]
        for (k, idx) in observed.enumerated() {
            let xs = flat.take(MLXArray(idx.map(Int32.init)))
            #expect(abs(stats.mu[0, 0, k].item(Float.self) - xs.mean().item(Float.self)) < 1e-3)
            #expect(
                abs(stats.sigma[0, 0, k].item(Float.self) - MLX.std(xs).item(Float.self)) < 1e-3)
            #expect(stats.n[0, 0, k].item(Float.self) == Float(idx.count))
        }
    }

    @Test("Stitching cross-fades overlapping patches")
    func testStitch() {
        let model = TimesFM3Model(tinyConfig)  // p = 4, patches of 8 overlap by 4
        // Patch 0 predicts all 1s, patch 1 all 3s.
        let preds = MLX.concatenated(
            [MLXArray.ones([1, 1, 1, 8, 1]), 3 * MLXArray.ones([1, 1, 1, 8, 1])], axis: 2)
        let out = model.stitch(preds).reshaped(-1)
        // First 4 from patch 0, then linspace(1, 0, 4) blend, then patch 1's tail.
        let w: [Float] = [1, 2.0 / 3, 1.0 / 3, 0]
        let expected: [Float] = [1, 1, 1, 1] + w.map { $0 * 1 + (1 - $0) * 3 } + [3, 3, 3, 3]
        #expect(maxAbsDiff(out, MLXArray(expected)) < 1e-6)
    }

    @Test("Linear detrending applies only when the trend dominates")
    func testDetrend() {
        let model = TimesFM3Model(tinyConfig)
        let t = MLXArray(0 ..< 64).asType(.float32)
        let trending = (2 * t + MLX.sin(t)).reshaped(1, 1, 64)
        let flat = MLX.sin(t).reshaped(1, 1, 64)
        let noMask = MLXArray.zeros([1, 1, 64], type: Bool.self)
        #expect(model.detrend(trending, masked: noMask, context: 64).apply.item(Bool.self))
        #expect(!model.detrend(flat, masked: noMask, context: 64).apply.item(Bool.self))
    }

    // MARK: - Model

    @Test("Parameter keys match the TimesFM 3.0 checkpoint layout")
    func testParameterKeys() {
        let keys = Set(TimesFM3Model(tinyConfig).parameters().flattened().map { $0.0 })
        #expect(keys.contains("pre_transformer_resblock.hidden_layer.weight"))
        #expect(keys.contains("transformer_stack.layers.0.seq_attn.query_proj.weight"))
        #expect(keys.contains("transformer_stack.layers.0.seq_attn.per_dim_scale.per_dim_scale"))
        #expect(keys.contains("transformer_stack.layers.1.var_attn.key_ln.weight"))
        #expect(keys.contains("transformer_stack.layers.1.post_var_attn_ln.weight"))
        #expect(keys.contains("output_head.bias"))
        #expect(!keys.contains { $0.contains(".bias") && !$0.hasPrefix("output_head") })
        #expect(keys.count == 3 + 2 * 22 + 2)
    }

    @Test("Without variate attention the var_attn modules are absent")
    func testNoVariateAttention() {
        var c = tinyConfig
        c.useVariateAttention = false
        let keys = TimesFM3Model(c).parameters().flattened().map { $0.0 }
        #expect(!keys.contains { $0.contains("var_attn") })
    }

    @Test("Forecast shapes with non-aligned, multivariate context and odd horizon")
    func testForecastShape() {
        let model = TimesFM3Model(tinyConfig)
        let input = TimeSeriesInput(
            series: MLXRandom.normal([2, 3, 30]), paddingMask: MLXArray.ones([2, 3, 30]),
            idMask: MLXArray.zeros([2, 3], type: Int32.self))
        let pred = model.forecast(input: input, predictionLength: 13, caches: [])
        #expect(pred.mean.shape == [2, 3, 13])
        #expect(pred.quantiles!.shape == [2, 3, 13, 9])
        #expect(MLX.isNaN(pred.quantiles!).any().item(Bool.self) == false)
        // Quantiles come back sorted and the mean is the median.
        let q = pred.quantiles!
        #expect((q[.ellipsis, 1...] .>= q[.ellipsis, ..<8]).all().item(Bool.self))
        #expect(maxAbsDiff(pred.mean, q[.ellipsis, 4]) == 0)
    }

    @Test("Contexts beyond maxContextLength are truncated")
    func testContextTruncation() {
        let model = TimesFM3Model(tinyConfig)  // maxContextLength 64
        let long = MLXRandom.normal([1, 1, 100])
        let a = model.forecast(input: .univariate(long.reshaped(-1)), predictionLength: 8, caches: [])
        let b = model.forecast(
            input: .univariate(long.reshaped(-1)[36...]), predictionLength: 8, caches: [])
        #expect(maxAbsDiff(a.quantiles!, b.quantiles!) < 1e-5)
    }
}

// MARK: - Golden values vs. upstream reference

/// A local TimesFM 3.0 conversion, made by
/// `Scripts/convert_ts_model.py --hf-path google/timesfm-3.0-pytorch --mlx-path <dir> --dtype <dtype>`.
struct TimesFM3Weights: CustomTestStringConvertible, Sendable {
    let name: String
    let directory: URL
    let dtype: DType
    /// Max forecast error, relative to the input's spread (floored at 1 for constant input).
    /// Upstream's own torch and MLX backends differ by up to ~2e-3 of that on a strongly
    /// trending series (fp32 trend fit); fp16 weights add rounding worth up to ~3e-3.
    let tolerance: Float

    var testDescription: String { name }

    /// `converted/<folder>` under the repo root, or the path in `env` if set.
    private static func directory(_ folder: String, env: String) -> URL {
        if let path = ProcessInfo.processInfo.environment[env] {
            return URL(fileURLWithPath: path)
        }
        return URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appending(path: "converted/\(folder)")
    }

    /// Google's exact weights (`--dtype float32`). Override with TIMESFM3_FP32_DIR.
    static let fp32 = TimesFM3Weights(
        name: "fp32", directory: directory("timesfm3-fp32", env: "TIMESFM3_FP32_DIR"),
        dtype: .float32, tolerance: 2.5e-3)
    /// The converter's default (`--dtype float16`). Override with TIMESFM3_FP16_DIR.
    static let fp16 = TimesFM3Weights(
        name: "fp16", directory: directory("timesfm3-fp16", env: "TIMESFM3_FP16_DIR"),
        dtype: .float16, tolerance: 5e-3)

    var checkpoint: URL { directory.appending(path: "model.safetensors") }

    /// The conversions present on this machine; a missing one is simply not tested.
    static let available = [fp32, fp16].filter {
        FileManager.default.fileExists(atPath: $0.checkpoint.path)
    }
}

/// Generated by Scripts/timesfm3_golden.py from timesfm 3.0.2's torch `TimesFM3Forecaster`,
/// using the fp32 checkpoint (Google's exact weights). Both conversions are compared to it.
private let timesFM3Golden = URL(fileURLWithPath: #filePath)
    .deletingLastPathComponent()
    .appending(path: "Fixtures/timesfm3_golden.json")

private struct TimesFM3Golden: Decodable {
    struct Case: Decodable {
        let name: String
        let horizon: Int
        let input: [[Float]]  // [variates, context]
        let forecast: [[[Float]]]  // [variates, horizon, quantiles]
    }
    let cases: [Case]
}

@Suite(
    "TimesFM 3 golden values",
    .enabled(if: !TimesFM3Weights.available.isEmpty),
    .serialized)
struct TimesFM3GoldenTests {

    @Test(
        "Checkpoint tensors match model parameters exactly (names, shapes, stored dtype)",
        arguments: TimesFM3Weights.available)
    func testCheckpointMatchesParameters(weights variant: TimesFM3Weights) throws {
        let model = TimesFM3Model(TimesFM3Configuration())
        let weights = model.sanitize(weights: try loadArrays(url: variant.checkpoint))
        let params = Dictionary(uniqueKeysWithValues: model.parameters().flattened())

        #expect(Set(weights.keys) == Set(params.keys))
        for (key, w) in weights {
            #expect(params[key]?.shape == w.shape, "\(key): \(w.shape) vs \(params[key]?.shape ?? [])")
            #expect(w.dtype == variant.dtype, "\(key) stored as \(w.dtype)")
        }
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: [.all])
    }

    @Test(
        "Forecasts match the upstream torch reference (loaded via loadFromDirectory)",
        arguments: TimesFM3Weights.available)
    func testMatchesReference(weights variant: TimesFM3Weights) throws {
        let forecaster = try TimeSeriesForecaster.loadFromDirectory(variant.directory)
        #expect(forecaster.model is TimesFM3Model)

        let golden = try JSONDecoder().decode(
            TimesFM3Golden.self, from: Data(contentsOf: timesFM3Golden))
        for c in golden.cases {
            let V = c.input.count
            let T = c.input[0].count
            let input = TimeSeriesInput(
                series: MLXArray(c.input.flatMap { $0 }).reshaped(1, V, T),
                paddingMask: MLXArray.ones([1, V, T]),
                idMask: MLXArray.zeros([1, V], type: Int32.self))
            let pred = forecaster.forecast(input: input, predictionLength: c.horizon)
            let expected = MLXArray(c.forecast.flatMap { $0.flatMap { $0 } })
                .reshaped(1, V, c.horizon, 9)

            let scale = max(MLX.std(MLXArray(c.input.flatMap { $0 })).item(Float.self), 1)
            let err = MLX.abs(pred.quantiles! - expected).max().item(Float.self) / scale
            let medianErr = MLX.abs(pred.mean - expected[.ellipsis, 4]).max().item(Float.self) / scale

            #expect(pred.mean.shape == [1, V, c.horizon])
            #expect(err < variant.tolerance, "\(variant.name) \(c.name): quantile err/std \(err)")
            #expect(
                medianErr < variant.tolerance, "\(variant.name) \(c.name): median err/std \(medianErr)")
            print(
                "TimesFM3 golden [\(variant.name)] \(c.name): quantile err/std \(err), "
                    + "median err/std \(medianErr)")
        }
    }

    @Test("Batched series forecast the same as one at a time", arguments: TimesFM3Weights.available)
    func testBatchMatchesSingle(weights variant: TimesFM3Weights) throws {
        let forecaster = try TimeSeriesForecaster.loadFromDirectory(variant.directory)
        let t = MLXArray(0 ..< 256).asType(.float32)
        let a = MLX.sin(t * 0.2) * 4
        let b = 100 + 0.3 * t + MLX.cos(t * 0.05)
        let batched = forecaster.forecast(
            input: TimeSeriesInput(
                series: MLX.stacked([a, b]).reshaped(2, 1, 256),
                paddingMask: MLXArray.ones([2, 1, 256]),
                idMask: MLXArray.zeros([2, 1], type: Int32.self)),
            predictionLength: 96)
        for (i, s) in [a, b].enumerated() {
            let single = forecaster.forecast(input: .univariate(s), predictionLength: 96)
            let diff = MLX.abs(batched.quantiles![i] - single.quantiles![0]).max().item(Float.self)
            #expect(diff < 1e-3, "\(variant.name) series \(i): \(diff)")
        }
    }
}
