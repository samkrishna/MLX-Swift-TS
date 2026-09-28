import Foundation
@preconcurrency import MLX
import MLXNN

// MARK: - Configuration

/// Configuration for the TimesFM 2.5 time series model.
///
/// Mirrors `TimesFM_2p5_200M_Definition` from google-research/timesfm.
public struct TimesFMConfiguration: Codable, Sendable {
    public var hiddenSize: Int
    public var numLayers: Int
    public var numHeads: Int
    public var intermediateSize: Int
    public var headDim: Int
    /// Input patch length (32). The tokenizer sees `2 * patchLength` features: values + mask.
    public var patchLength: Int
    /// Steps produced per forward pass by the point head (128).
    public var outputPatchLength: Int
    /// Steps produced by the quantile-spread head (1024).
    public var quantileHorizonLength: Int
    public var numQuantiles: Int
    public var contextLength: Int
    public var predictionLength: Int
    public var ropeTheta: Float
    /// Output channel used as the point forecast and fed back during decoding (median).
    public var decodeIndex: Int

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case numLayers = "num_layers"
        case numHeads = "num_heads"
        case intermediateSize = "intermediate_size"
        case headDim = "head_dim"
        case patchLength = "input_patch_len"
        case legacyPatchLength = "patch_length"
        case outputPatchLength = "output_patch_len"
        case quantileHorizonLength = "quantile_horizon_length"
        case numQuantiles = "num_quantiles"
        case contextLength = "context_length"
        case predictionLength = "prediction_length"
        case ropeTheta = "rope_theta"
        case decodeIndex = "decode_index"
    }

    /// Creates a configuration. Defaults are the published TimesFM 2.5 200M values.
    public init(
        hiddenSize: Int = 1280, numLayers: Int = 20, numHeads: Int = 16,
        intermediateSize: Int = 1280, headDim: Int = 80, patchLength: Int = 32,
        outputPatchLength: Int = 128, quantileHorizonLength: Int = 1024, numQuantiles: Int = 9,
        contextLength: Int = 16384, predictionLength: Int = 128,
        ropeTheta: Float = 10000.0, decodeIndex: Int = 5
    ) {
        self.hiddenSize = hiddenSize; self.numLayers = numLayers; self.numHeads = numHeads
        self.intermediateSize = intermediateSize; self.headDim = headDim
        self.patchLength = patchLength; self.outputPatchLength = outputPatchLength
        self.quantileHorizonLength = quantileHorizonLength; self.numQuantiles = numQuantiles
        self.contextLength = contextLength; self.predictionLength = predictionLength
        self.ropeTheta = ropeTheta; self.decodeIndex = decodeIndex
    }

    /// Decodes `config.json`, falling back to the 2.5 defaults for any missing key.
    ///
    /// Accepts both the current `input_patch_len` key and the legacy `patch_length` key
    /// that older converter output (including the published Hub repo) still carries.
    /// Keys from earlier Swift implementations (`use_rope`, `query_pre_attn_scalar`,
    /// `use_horizon_ff`, ...) are ignored: those behaviors are now fixed by the architecture.
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let d = TimesFMConfiguration()
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? d.hiddenSize
        numLayers = try c.decodeIfPresent(Int.self, forKey: .numLayers) ?? d.numLayers
        numHeads = try c.decodeIfPresent(Int.self, forKey: .numHeads) ?? d.numHeads
        intermediateSize =
            try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? d.intermediateSize
        headDim = try c.decodeIfPresent(Int.self, forKey: .headDim) ?? d.headDim
        // Older convert_ts_model.py output wrote the tokenizer input width (values + mask)
        // as `patch_length`, so the real patch length is half of it.
        if let p = try c.decodeIfPresent(Int.self, forKey: .patchLength) {
            patchLength = p
        } else if let legacy = try c.decodeIfPresent(Int.self, forKey: .legacyPatchLength) {
            patchLength = legacy / 2
        } else {
            patchLength = d.patchLength
        }
        outputPatchLength =
            try c.decodeIfPresent(Int.self, forKey: .outputPatchLength) ?? d.outputPatchLength
        quantileHorizonLength =
            try c.decodeIfPresent(Int.self, forKey: .quantileHorizonLength)
            ?? d.quantileHorizonLength
        numQuantiles = try c.decodeIfPresent(Int.self, forKey: .numQuantiles) ?? d.numQuantiles
        contextLength = try c.decodeIfPresent(Int.self, forKey: .contextLength) ?? d.contextLength
        predictionLength =
            try c.decodeIfPresent(Int.self, forKey: .predictionLength) ?? d.predictionLength
        ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? d.ropeTheta
        decodeIndex = try c.decodeIfPresent(Int.self, forKey: .decodeIndex) ?? d.decodeIndex
    }

    /// Encodes using the current key names only (`input_patch_len`, never `patch_length`),
    /// so a round trip does not re-trigger the legacy halving in `init(from:)`.
    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(hiddenSize, forKey: .hiddenSize)
        try c.encode(numLayers, forKey: .numLayers)
        try c.encode(numHeads, forKey: .numHeads)
        try c.encode(intermediateSize, forKey: .intermediateSize)
        try c.encode(headDim, forKey: .headDim)
        try c.encode(patchLength, forKey: .patchLength)
        try c.encode(outputPatchLength, forKey: .outputPatchLength)
        try c.encode(quantileHorizonLength, forKey: .quantileHorizonLength)
        try c.encode(numQuantiles, forKey: .numQuantiles)
        try c.encode(contextLength, forKey: .contextLength)
        try c.encode(predictionLength, forKey: .predictionLength)
        try c.encode(ropeTheta, forKey: .ropeTheta)
        try c.encode(decodeIndex, forKey: .decodeIndex)
    }
}

// MARK: - TimesFM Components

/// Residual block: output_layer(silu(hidden_layer(x))) + residual_layer(x)
class TimesFMResidualBlock: Module {
    @ModuleInfo(key: "hidden_layer") var hiddenLayer: Linear
    @ModuleInfo(key: "output_layer") var outputLayer: Linear
    @ModuleInfo(key: "residual_layer") var residualLayer: Linear

    /// - Parameters:
    ///   - inputDim: Feature width going in.
    ///   - outputDim: Feature width coming out (also the hidden width unless `hiddenDim` is set).
    ///   - hiddenDim: Width of the inner SiLU layer.
    ///   - bias: The tokenizer uses biases; the output heads do not.
    init(inputDim: Int, outputDim: Int, hiddenDim: Int? = nil, bias: Bool = true) {
        let hDim = hiddenDim ?? outputDim
        self._hiddenLayer.wrappedValue = Linear(inputDim, hDim, bias: bias)
        self._outputLayer.wrappedValue = Linear(hDim, outputDim, bias: bias)
        self._residualLayer.wrappedValue = Linear(inputDim, outputDim, bias: bias)
    }

    /// Applies the block over the last axis of `x`; leading axes pass through unchanged.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        outputLayer(silu(hiddenLayer(x))) + residualLayer(x)
    }
}

/// Wrapper module for per_dim_scale nested parameter.
class TimesFMPerDimScale: Module {
    @ModuleInfo(key: "per_dim_scale") var perDimScale: MLXArray

    /// Zero-initialized like the reference (softplus(0) ≈ 0.69); real values come from the
    /// checkpoint. Nested so the parameter path is `attn.per_dim_scale.per_dim_scale`.
    init(headDim: Int) {
        self._perDimScale.wrappedValue = MLXArray.zeros([headDim])
    }
}

/// TimesFM attention: fused QKV, RoPE, QK RMS norms, learned per-dimension query scale.
class TimesFMAttention: Module {
    @ModuleInfo(key: "qkv_proj") var qkvProj: Linear
    @ModuleInfo(key: "out") var wO: Linear
    @ModuleInfo(key: "per_dim_scale") var perDimScaleModule: TimesFMPerDimScale
    @ModuleInfo(key: "query_ln") var queryLN: RMSNorm
    @ModuleInfo(key: "key_ln") var keyLN: RMSNorm

    let numHeads: Int
    let headDim: Int
    let ropeTheta: Float

    /// Builds the fused QKV projection (`[q; k; v]` stacked on the output axis, matching the
    /// checkpoint), the output projection, per-head-dim QK RMS norms (eps 1e-6, as upstream),
    /// and the per-dimension query scale.
    init(_ config: TimesFMConfiguration) {
        self.numHeads = config.numHeads
        self.headDim = config.headDim
        self.ropeTheta = config.ropeTheta
        let totalDim = numHeads * headDim
        self._qkvProj.wrappedValue = Linear(config.hiddenSize, 3 * totalDim, bias: false)
        self._wO.wrappedValue = Linear(totalDim, config.hiddenSize, bias: false)
        self._perDimScaleModule.wrappedValue = TimesFMPerDimScale(headDim: config.headDim)
        self._queryLN.wrappedValue = RMSNorm(dimensions: config.headDim, eps: 1e-6)
        self._keyLN.wrappedValue = RMSNorm(dimensions: config.headDim, eps: 1e-6)
    }

    /// Causal self-attention over `L` patch tokens, optionally continuing from a KV cache.
    ///
    /// Order follows the reference exactly: project → RoPE → QK norms → scale queries by
    /// `1.442695 / sqrt(headDim) * softplus(per_dim_scale)` → attention with scale 1.
    ///
    /// - Parameters:
    ///   - x: `[N, L, hidden]`.
    ///   - numMasked: `[N]` count of fully padded leading patches per row. Their keys are
    ///     never attended to and RoPE positions start after them.
    ///   - cache: Appended to in place; `cache.offset` is the absolute index of `x[:, 0]`.
    func callAsFunction(
        _ x: MLXArray,
        numMasked: MLXArray,
        cache: TimeSeriesKVCache? = nil
    ) -> MLXArray {
        let N = x.dim(0)
        let L = x.dim(1)
        let totalDim = numHeads * headDim
        let offset = cache?.offset ?? 0

        let qkv = qkvProj(x)
        var q = qkv[.ellipsis, 0 ..< totalDim]
            .reshaped(N, L, numHeads, headDim).transposed(0, 2, 1, 3)
        var k = qkv[.ellipsis, totalDim ..< (2 * totalDim)]
            .reshaped(N, L, numHeads, headDim).transposed(0, 2, 1, 3)
        var v = qkv[.ellipsis, (2 * totalDim) ..< (3 * totalDim)]
            .reshaped(N, L, numHeads, headDim).transposed(0, 2, 1, 3)

        // RoPE is applied before the QK norms, at positions shifted past padded patches.
        let positions =
            MLXArray(Int32(offset) ..< Int32(offset + L)).reshaped(1, L).asType(.float32)
            - numMasked.reshaped(N, 1).asType(.float32)
        q = applyRoPE(q, positions: positions)
        k = applyRoPE(k, positions: positions)

        q = queryLN(q)
        k = keyLN(k)

        let scale = (1.442695041 / Float(headDim).squareRoot())
            * softplus(perDimScaleModule.perDimScale)
        q = q * scale.asType(q.dtype)

        if let cache {
            (k, v) = cache.update(keys: k, values: v)
        }

        // Causal, and padded patches are invisible. Each query may always see itself so
        // padded-patch rows (whose outputs are never used) don't become all-masked NaNs.
        let kvLen = k.dim(2)
        let qIndex = MLXArray(Int32(offset) ..< Int32(offset + L)).reshaped(1, 1, L, 1)
        let kvIndex = MLXArray(Int32(0) ..< Int32(kvLen)).reshaped(1, 1, 1, kvLen)
        let visible = (kvIndex .>= numMasked.reshaped(N, 1, 1, 1)) .|| (kvIndex .== qIndex)
        let mask = (qIndex .>= kvIndex) .&& visible

        let output = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: 1.0, mask: .array(mask)
        )

        return wO(output.transposed(0, 2, 1, 3).reshaped(N, L, -1))
    }

    /// Half-split rotary embedding (first half / second half), matching the reference.
    /// `positions` is `[N, L]`.
    private func applyRoPE(_ x: MLXArray, positions: MLXArray) -> MLXArray {
        let half = headDim / 2
        let fraction = MLXArray(0 ..< half).asType(.float32) * (2.0 / Float(headDim))
        let timescale = MLX.pow(MLXArray(ropeTheta), fraction)
        let angles = positions.expandedDimensions(axes: [1, 3]) / timescale  // [N,1,L,half]
        let cosA = MLX.cos(angles).asType(x.dtype)
        let sinA = MLX.sin(angles).asType(x.dtype)
        let x1 = x[.ellipsis, 0 ..< half]
        let x2 = x[.ellipsis, half ..< headDim]
        return MLX.concatenated([x1 * cosA - x2 * sinA, x2 * cosA + x1 * sinA], axis: -1)
    }
}

/// TimesFM transformer block with sandwich norms (pre + post per sub-layer).
class TimesFMTransformerBlock: Module {
    @ModuleInfo(key: "pre_attn_ln") var preAttnLN: RMSNorm
    @ModuleInfo(key: "post_attn_ln") var postAttnLN: RMSNorm
    @ModuleInfo(key: "pre_ff_ln") var preFFLN: RMSNorm
    @ModuleInfo(key: "post_ff_ln") var postFFLN: RMSNorm
    @ModuleInfo(key: "attn") var attn: TimesFMAttention
    @ModuleInfo(key: "ff0") var ff0: Linear
    @ModuleInfo(key: "ff1") var ff1: Linear

    /// Builds the four RMS norms (eps 1e-6), the attention layer, and the two-layer SiLU FFN.
    init(_ config: TimesFMConfiguration) {
        self._preAttnLN.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: 1e-6)
        self._postAttnLN.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: 1e-6)
        self._preFFLN.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: 1e-6)
        self._postFFLN.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: 1e-6)
        self._attn.wrappedValue = TimesFMAttention(config)
        self._ff0.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        self._ff1.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: false)
    }

    /// Runs attention then the FFN, each wrapped as `x + postNorm(sublayer(preNorm(x)))`.
    ///
    /// - Parameters:
    ///   - x: `[N, L, hidden]`.
    ///   - numMasked: Passed through to attention; see ``TimesFMAttention``.
    ///   - cache: This layer's KV cache, or `nil` for a stateless pass.
    func callAsFunction(
        _ x: MLXArray, numMasked: MLXArray, cache: TimeSeriesKVCache?
    ) -> MLXArray {
        let h = x + postAttnLN(attn(preAttnLN(x), numMasked: numMasked, cache: cache))
        return h + postFFLN(ff1(silu(ff0(preFFLN(h)))))
    }
}

// MARK: - Running statistics (RevIN)

/// Per-row running count / mean / standard deviation (population), in float32.
struct TimesFMRunningStats {
    var n: MLXArray  // [N]
    var mu: MLXArray  // [N]
    var sigma: MLXArray  // [N]

    /// Starting state for `rows` series: nothing observed yet (n = 0, mean 0, std 0).
    static func empty(_ rows: Int) -> TimesFMRunningStats {
        let z = MLXArray.zeros([rows], dtype: .float32)
        return TimesFMRunningStats(n: z, mu: z, sigma: z)
    }

    /// Fold in `patches` (`[N, K, p]`, `padMask` 1 = padded) one patch at a time.
    ///
    /// Returns the stats after each patch (`[N, K]` mean and sigma) and the final state.
    /// Equivalent to the reference's sequential `update_running_stats`; computed with
    /// cumulative sums around a shift to avoid cancellation on large-offset series.
    func accumulate(_ patches: MLXArray, padMask: MLXArray)
        -> (mu: MLXArray, sigma: MLXArray, final: TimesFMRunningStats)
    {
        let x = patches.asType(.float32)
        let legit = 1 - padMask.asType(.float32)
        let incN = legit.sum(axis: -1)  // [N, K]

        let observedMean = (x * legit).sum(axes: [1, 2]) / MLX.maximum(incN.sum(axis: -1), 1)
        // Shift by the prior mean when there is one, otherwise by this data's mean.
        let shift = MLX.where(n .> 0, mu, observedMean).reshaped(-1, 1, 1)

        let xs = (x - shift) * legit
        let s1 = xs.sum(axis: -1)  // [N, K]
        let s2 = (xs * xs).sum(axis: -1)

        let priorS1 = (n * (mu - shift.reshaped(-1))).reshaped(-1, 1)
        let priorS2 = (n * (sigma * sigma + (mu - shift.reshaped(-1)).square())).reshaped(-1, 1)

        let cumN = n.reshaped(-1, 1) + incN.cumsum(axis: 1)
        let safeN = MLX.maximum(cumN, 1)
        let meanShifted = (priorS1 + s1.cumsum(axis: 1)) / safeN
        let variance = MLX.maximum(
            (priorS2 + s2.cumsum(axis: 1)) / safeN - meanShifted.square(), 0)

        let hasData = cumN .> 0
        let newMu = MLX.where(hasData, meanShifted + shift.reshaped(-1, 1), 0)
        let newSigma = MLX.where(hasData, MLX.sqrt(variance), 0)

        let last = newMu.dim(1) - 1
        let final = TimesFMRunningStats(
            n: cumN[0..., last], mu: newMu[0..., last], sigma: newSigma[0..., last])
        return (newMu, newSigma, final)
    }
}

// MARK: - TimesFMModel

/// TimesFM 2.5 time series forecasting model.
///
/// Decoder-only transformer over 32-step patches. Each patch is normalized by the
/// running mean/std up to and including it, tokenized together with its padding mask,
/// and every position predicts the next 128 steps × 10 channels (mean + 9 quantiles).
/// Horizons beyond 128 are decoded autoregressively by feeding back the median.
public class TimesFMModel: Module, TimeSeriesModel {

    @ModuleInfo(key: "tokenizer") var tokenizer: TimesFMResidualBlock
    @ModuleInfo(key: "stacked_xf") var layers: [TimesFMTransformerBlock]
    @ModuleInfo(key: "output_projection_point") var outputPoint: TimesFMResidualBlock
    /// Quantile-spread head (1024 steps). Only used by the reference's optional
    /// continuous-quantile mode; kept so checkpoints load completely.
    @ModuleInfo(key: "output_projection_quantiles") var outputQuantiles: TimesFMResidualBlock

    let config: TimesFMConfiguration

    /// Output channels: mean + quantiles.
    var numChannels: Int { config.numQuantiles + 1 }

    /// Builds the model with the checkpoint's module layout.
    ///
    /// The tokenizer reads `2 * patchLength` features (values + mask). Both output heads
    /// read the same transformer output: the point head emits `outputPatchLength` steps and
    /// the quantile head `quantileHorizonLength` steps, each with `numQuantiles + 1` channels.
    public init(_ config: TimesFMConfiguration) {
        self.config = config

        self._tokenizer.wrappedValue = TimesFMResidualBlock(
            inputDim: 2 * config.patchLength, outputDim: config.hiddenSize)
        self._layers.wrappedValue = (0 ..< config.numLayers).map { _ in
            TimesFMTransformerBlock(config)
        }
        let channels = config.numQuantiles + 1
        self._outputPoint.wrappedValue = TimesFMResidualBlock(
            inputDim: config.hiddenSize, outputDim: config.outputPatchLength * channels,
            hiddenDim: config.hiddenSize, bias: false)
        self._outputQuantiles.wrappedValue = TimesFMResidualBlock(
            inputDim: config.hiddenSize, outputDim: config.quantileHorizonLength * channels,
            hiddenDim: config.hiddenSize, bias: false)
    }

    /// One forward pass over normalized patches (prefill, or one decode step with caches).
    ///
    /// Every position's output is its forecast for the `outputPatchLength` steps after that
    /// patch, still in that patch's normalized space. Callers use the last position.
    ///
    /// - Parameters:
    ///   - patches: `[N, K, p]` normalized values (padded positions zeroed).
    ///   - padMask: `[N, K, p]`, 1 = padded.
    ///   - numMasked: `[N]` fully padded leading patches.
    /// - Returns: Normalized point output `[N, K, outputPatchLength, channels]`.
    func callAsFunction(
        patches: MLXArray, padMask: MLXArray, numMasked: MLXArray,
        caches: [TimeSeriesKVCache?]
    ) -> MLXArray {
        let dtype = tokenizer.hiddenLayer.weight.dtype
        let tokens = MLX.concatenated([patches, padMask], axis: -1).asType(dtype)
        var hidden = tokenizer(tokens)
        for (i, layer) in layers.enumerated() {
            hidden = layer(hidden, numMasked: numMasked, cache: caches[i])
        }
        let N = patches.dim(0)
        let K = patches.dim(1)
        return outputPoint(hidden).reshaped(N, K, config.outputPatchLength, numChannels)
    }

    /// Forecasts `predictionLength` steps for every series in `input`, following upstream's
    /// `forecast_naive`: no extra input normalization, flip-invariance, or quantile fixes.
    ///
    /// Steps:
    /// 1. Flatten `[B, V, T]` to `N = B * V` independent series. Keep the last
    ///    `contextLength` steps, then left-pad with masked zeros to whole patches.
    /// 2. Prefill: normalize each patch by the running stats through that patch, run the
    ///    model, and denormalize the last position's output. That gives the first
    ///    `outputPatchLength` steps.
    /// 3. While more steps are needed, feed the previous chunk's median back in as
    ///    `outputPatchLength / patchLength` new patches, update the stats, and decode
    ///    another chunk through the KV caches.
    ///
    /// - Parameters:
    ///   - input: Series `[B, V, T]`. `paddingMask` 1 = observed, 0 = missing.
    ///   - predictionLength: Steps to forecast. Any length works; past 128 it decodes step by step.
    ///   - caches: Fresh caches from ``newCaches()``. If the count doesn't match the layer
    ///     count, new ones are made.
    /// - Returns: `mean` is the median (channel `decodeIndex`) `[B, V, H]`; `quantiles` are
    ///   the 0.1...0.9 channels `[B, V, H, numQuantiles]`.
    public func forecast(
        input: TimeSeriesInput,
        predictionLength: Int,
        caches: [TimeSeriesKVCache?]
    ) -> TimeSeriesPrediction {
        let p = config.patchLength
        let o = config.outputPatchLength
        let perStep = o / p

        let B = input.series.dim(0)
        let V = input.series.dim(1)
        let N = B * V

        var series = input.series.reshaped(N, -1).asType(.float32)
        var padMask = (1 - input.paddingMask.reshaped(N, -1)).asType(.float32)

        // Keep the most recent context, then left-pad to a whole number of patches.
        if series.dim(1) > config.contextLength {
            let start = series.dim(1) - config.contextLength
            series = series[0..., start...]
            padMask = padMask[0..., start...]
        }
        let padLen = (p - series.dim(1) % p) % p
        if padLen > 0 {
            series = MLX.concatenated([MLXArray.zeros([N, padLen]), series], axis: 1)
            padMask = MLX.concatenated([MLXArray.ones([N, padLen]), padMask], axis: 1)
        }
        let K = series.dim(1) / p
        let patches = series.reshaped(N, K, p)
        let patchMask = padMask.reshaped(N, K, p)
        let numMasked = patchMask[0..., 0..., p - 1].sum(axis: 1).asType(.int32)

        let layerCaches =
            caches.count == config.numLayers ? caches : newCaches()

        // Prefill: normalize each patch by the running stats through that patch.
        let (ctxMu, ctxSigma, ctxStats) = TimesFMRunningStats.empty(N)
            .accumulate(patches, padMask: patchMask)
        let normed = MLX.where(
            patchMask .> 0, 0, revin(patches, mu: ctxMu, sigma: ctxSigma))
        let out = self(
            patches: normed, padMask: patchMask, numMasked: numMasked, caches: layerCaches)
        var chunks = [denormalize(out[0..., K - 1], mu: ctxMu[0..., K - 1], sigma: ctxSigma[0..., K - 1])]

        // Autoregressive decode: feed the median back as `perStep` new patches.
        var stats = ctxStats
        let decodeSteps = max(predictionLength - 1, 0) / o
        for _ in 0 ..< decodeSteps {
            let newPatches = chunks.last![0..., 0..., config.decodeIndex].reshaped(N, perStep, p)
            let newMask = MLXArray.zeros([N, perStep, p], dtype: .float32)
            let (mu, sigma, next) = stats.accumulate(newPatches, padMask: newMask)
            stats = next
            let stepOut = self(
                patches: revin(newPatches, mu: mu, sigma: sigma), padMask: newMask,
                numMasked: numMasked, caches: layerCaches)
            chunks.append(
                denormalize(
                    stepOut[0..., perStep - 1], mu: mu[0..., perStep - 1],
                    sigma: sigma[0..., perStep - 1]))
        }

        let full = MLX.concatenated(chunks, axis: 1)[0..., 0 ..< predictionLength]  // [N, H, C]
        let H = full.dim(1)
        let median = full[0..., 0..., config.decodeIndex].reshaped(B, V, H)
        let quantiles = full[0..., 0..., 1...].reshaped(B, V, H, config.numQuantiles)

        return TimeSeriesPrediction(mean: median, quantiles: quantiles, predictionLength: H)
    }

    /// Normalize `[N, K, p]` by per-patch stats `[N, K]` (upstream `revin`, forward).
    /// A near-zero sigma divides by 1 so constant series don't blow up.
    private func revin(_ x: MLXArray, mu: MLXArray, sigma: MLXArray) -> MLXArray {
        let safeSigma = MLX.where(sigma .< 1e-6, 1, sigma)
        return (x - mu.expandedDimensions(axis: -1)) / safeSigma.expandedDimensions(axis: -1)
    }

    /// Denormalize one position's output `[N, o, C]` with its stats `[N]` (upstream `revin`,
    /// reverse). Returns float32 regardless of the model dtype.
    private func denormalize(_ x: MLXArray, mu: MLXArray, sigma: MLXArray) -> MLXArray {
        x.asType(.float32) * sigma.reshaped(-1, 1, 1) + mu.reshaped(-1, 1, 1)
    }

    /// Maps checkpoint keys onto this module tree. Handles both the upstream torch release
    /// (`*_ln.scale`) and the converted HF-transformers checkpoint (`*_ln.weight`, plus an
    /// unused `horizon_ff_layer`).
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var result = [String: MLXArray]()
        for (key, value) in weights {
            // horizon_ff_layer comes from the HF transformers port; TimesFM 2.5 doesn't use it.
            if key.hasPrefix("horizon_ff_layer.") { continue }
            // Upstream RMSNorm params are "_ln.scale"; MLXNN expects "_ln.weight".
            // per_dim_scale is a learned attention parameter, not a norm — leave it.
            let newKey: String
            if key.hasSuffix("_ln.scale") {
                newKey = String(key.dropLast("scale".count)) + "weight"
            } else {
                newKey = key
            }
            result[newKey] = value
        }
        return result
    }

    /// float16 drifts noticeably once horizons exceed one output patch: the fed-back
    /// median amplifies its rounding error (~9% of series std at 256 steps vs ~0.02% in
    /// float32 against the reference), so run in float32.
    public var inferenceDtype: DType { .float32 }

    /// One KV cache per transformer layer; every TimesFM layer attends over time.
    public func newCaches() -> [TimeSeriesKVCache?] {
        (0 ..< config.numLayers).map { _ in TimeSeriesKVCache() }
    }
}
