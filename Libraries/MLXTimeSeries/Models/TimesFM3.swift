import Foundation
@preconcurrency import MLX
import MLXNN

// MARK: - Configuration

/// Configuration for the TimesFM 3.0 time series model.
///
/// Mirrors `TimesFM3MlxConfig` from upstream's `timesfm3.mlx` package (timesfm 3.0.2).
public struct TimesFM3Configuration: Codable, Sendable {
    /// Context patch length (32).
    public var inputPatchLength: Int
    /// Steps each patch predicts (64), i.e. `rolls` input patches.
    public var outputPatchLength: Int
    public var modelDims: Int
    public var hiddenDims: Int
    public var numLayers: Int
    public var numHeads: Int
    public var quantiles: [Float]
    /// Mix information across the variates of one series at every layer.
    public var useVariateAttention: Bool
    /// Remove a per-variate linear trend when it explains enough of the variance.
    public var useLinearDetrending: Bool
    public var linearDetrendingThreshold: Float
    public var valueClip: Float
    /// Longer contexts are truncated to their most recent `maxContextLength` steps.
    public var maxContextLength: Int

    enum CodingKeys: String, CodingKey {
        case inputPatchLength = "input_patch_len"
        case outputPatchLength = "output_patch_len"
        case modelDims = "model_dims"
        case hiddenDims = "hidden_dims"
        case numLayers = "num_layers"
        case numHeads = "num_heads"
        case quantiles
        case useVariateAttention = "use_variate_attention"
        case useLinearDetrending = "use_linear_detrending"
        case linearDetrendingThreshold = "linear_detrending_threshold"
        case valueClip = "value_clip"
        case maxContextLength = "max_context_length"
    }

    /// Creates a configuration. Defaults are the published TimesFM 3.0 values.
    public init(
        inputPatchLength: Int = 32, outputPatchLength: Int = 64, modelDims: Int = 1280,
        hiddenDims: Int = 1280, numLayers: Int = 20, numHeads: Int = 16,
        quantiles: [Float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        useVariateAttention: Bool = true, useLinearDetrending: Bool = true,
        linearDetrendingThreshold: Float = 0.5, valueClip: Float = 1e20,
        maxContextLength: Int = 15360
    ) {
        self.inputPatchLength = inputPatchLength; self.outputPatchLength = outputPatchLength
        self.modelDims = modelDims; self.hiddenDims = hiddenDims
        self.numLayers = numLayers; self.numHeads = numHeads; self.quantiles = quantiles
        self.useVariateAttention = useVariateAttention
        self.useLinearDetrending = useLinearDetrending
        self.linearDetrendingThreshold = linearDetrendingThreshold
        self.valueClip = valueClip; self.maxContextLength = maxContextLength
    }

    /// Decodes `config.json`, falling back to the 3.0 defaults for any missing key.
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let d = TimesFM3Configuration()
        inputPatchLength =
            try c.decodeIfPresent(Int.self, forKey: .inputPatchLength) ?? d.inputPatchLength
        outputPatchLength =
            try c.decodeIfPresent(Int.self, forKey: .outputPatchLength) ?? d.outputPatchLength
        modelDims = try c.decodeIfPresent(Int.self, forKey: .modelDims) ?? d.modelDims
        hiddenDims = try c.decodeIfPresent(Int.self, forKey: .hiddenDims) ?? d.hiddenDims
        numLayers = try c.decodeIfPresent(Int.self, forKey: .numLayers) ?? d.numLayers
        numHeads = try c.decodeIfPresent(Int.self, forKey: .numHeads) ?? d.numHeads
        quantiles = try c.decodeIfPresent([Float].self, forKey: .quantiles) ?? d.quantiles
        useVariateAttention =
            try c.decodeIfPresent(Bool.self, forKey: .useVariateAttention)
            ?? d.useVariateAttention
        useLinearDetrending =
            try c.decodeIfPresent(Bool.self, forKey: .useLinearDetrending)
            ?? d.useLinearDetrending
        linearDetrendingThreshold =
            try c.decodeIfPresent(Float.self, forKey: .linearDetrendingThreshold)
            ?? d.linearDetrendingThreshold
        valueClip = try c.decodeIfPresent(Float.self, forKey: .valueClip) ?? d.valueClip
        maxContextLength =
            try c.decodeIfPresent(Int.self, forKey: .maxContextLength) ?? d.maxContextLength
    }

    public var headDim: Int { modelDims / numHeads }
    public var numQuantiles: Int { quantiles.count }
    /// Input patches covered by one output patch (64 / 32 = 2).
    public var rolls: Int { outputPatchLength / inputPatchLength }
    /// Quantile used as the point forecast and for the horizon RevIN refinement (0.5).
    public var medianIndex: Int { numQuantiles / 2 }
}

// MARK: - Components

/// `output_layer(relu(hidden_layer(x))) + residual_layer(x)`, no biases.
class TimesFM3ResidualBlock: Module {
    @ModuleInfo(key: "hidden_layer") var hiddenLayer: Linear
    @ModuleInfo(key: "output_layer") var outputLayer: Linear
    @ModuleInfo(key: "residual_layer") var residualLayer: Linear

    init(inputDim: Int, outputDim: Int) {
        self._hiddenLayer.wrappedValue = Linear(inputDim, outputDim, bias: false)
        self._outputLayer.wrappedValue = Linear(outputDim, outputDim, bias: false)
        self._residualLayer.wrappedValue = Linear(inputDim, outputDim, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        outputLayer(relu(hiddenLayer(x))) + residualLayer(x)
    }
}

/// Holds the learned per-dimension query scale, nested so its path is
/// `*_attn.per_dim_scale.per_dim_scale` as in the checkpoint.
class TimesFM3PerDimScale: Module {
    @ParameterInfo(key: "per_dim_scale") var perDimScale: MLXArray

    init(headDim: Int) {
        self._perDimScale.wrappedValue = MLXArray.zeros([headDim])
    }
}

/// Multi-head attention with optional RoPE, per-head QK RMSNorm and per-dimension query scale.
class TimesFM3Attention: Module {
    @ModuleInfo(key: "query_proj") var queryProj: Linear
    @ModuleInfo(key: "key_proj") var keyProj: Linear
    @ModuleInfo(key: "value_proj") var valueProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear
    @ModuleInfo(key: "query_ln") var queryLN: RMSNorm
    @ModuleInfo(key: "key_ln") var keyLN: RMSNorm
    @ModuleInfo(key: "per_dim_scale") var perDimScale: TimesFM3PerDimScale

    let numHeads: Int
    let headDim: Int
    let useRoPE: Bool
    let causal: Bool

    /// Sequence attention uses RoPE and a causal mask; variate attention uses neither.
    init(_ config: TimesFM3Configuration, useRoPE: Bool, causal: Bool) {
        let d = config.modelDims
        self.numHeads = config.numHeads
        self.headDim = config.headDim
        self.useRoPE = useRoPE
        self.causal = causal
        self._queryProj.wrappedValue = Linear(d, d, bias: false)
        self._keyProj.wrappedValue = Linear(d, d, bias: false)
        self._valueProj.wrappedValue = Linear(d, d, bias: false)
        self._outProj.wrappedValue = Linear(d, d, bias: false)
        self._queryLN.wrappedValue = RMSNorm(dimensions: headDim, eps: TimesFM3Model.rmsEps)
        self._keyLN.wrappedValue = RMSNorm(dimensions: headDim, eps: TimesFM3Model.rmsEps)
        self._perDimScale.wrappedValue = TimesFM3PerDimScale(headDim: headDim)
    }

    /// Attention over the `L` axis of `x` (`[N, L, d]`).
    ///
    /// Order follows the reference: project → RoPE → QK norms → per-dim query scale. Logits are
    /// then multiplied by `sqrt(headDim)` (upstream's `rescale_logits=False` path).
    ///
    /// - Parameter keyMasked: `[N, L]`, true = this position can't be attended to.
    func callAsFunction(_ x: MLXArray, keyMasked: MLXArray) -> MLXArray {
        let N = x.dim(0)
        let L = x.dim(1)
        // One key: softmax weight is exactly 1, so attention reduces to the value path.
        if L == 1 {
            return outProj(valueProj(x))
        }

        var q = queryProj(x).reshaped(N, L, numHeads, headDim)
        var k = keyProj(x).reshaped(N, L, numHeads, headDim)
        let v = valueProj(x).reshaped(N, L, numHeads, headDim)
        if useRoPE {
            q = applyRoPE(q)
            k = applyRoPE(k)
        }
        q = queryLN(q)
        k = keyLN(k)
        let scale = (1.442695041 / Float(headDim).squareRoot()) * softplus(perDimScale.perDimScale)
        q = q * scale

        var attend = logicalNot(keyMasked).reshaped(N, 1, 1, L)
        if causal {
            let qi = MLXArray(Int32(0) ..< Int32(L)).reshaped(1, 1, L, 1)
            let ki = MLXArray(Int32(0) ..< Int32(L)).reshaped(1, 1, 1, L)
            attend = attend .&& (qi .>= ki)
        } else {
            attend = MLX.broadcast(attend, to: [N, 1, L, L])
        }
        // Additive -1e9 bias (not a boolean mask) so fully masked rows stay finite, as upstream.
        let bias = MLX.where(attend, Float(0), Float(-1e9)).asType(q.dtype)

        let out = MLXFast.scaledDotProductAttention(
            queries: q.transposed(0, 2, 1, 3), keys: k.transposed(0, 2, 1, 3),
            values: v.transposed(0, 2, 1, 3), scale: Float(headDim).squareRoot(),
            mask: .array(bias))
        return outProj(out.transposed(0, 2, 1, 3).reshaped(N, L, numHeads * headDim))
    }

    /// Half-split rotary embedding at positions `0..<L` on `[N, L, H, hd]`.
    private func applyRoPE(_ x: MLXArray) -> MLXArray {
        let L = x.dim(1)
        let half = headDim / 2
        let fraction = MLXArray(0 ..< half).asType(.float32) * (2.0 / Float(headDim))
        let timescale = MLX.pow(MLXArray(Float(10000)), fraction)
        let angles =
            MLXArray(0 ..< L).asType(.float32).reshaped(1, L, 1, 1) / timescale.reshaped(1, 1, 1, half)
        let cosA = MLX.cos(angles).asType(x.dtype)
        let sinA = MLX.sin(angles).asType(x.dtype)
        let x1 = x[.ellipsis, 0 ..< half]
        let x2 = x[.ellipsis, half ..< headDim]
        return MLX.concatenated([x1 * cosA - x2 * sinA, x2 * cosA + x1 * sinA], axis: -1)
    }
}

/// One layer: sequence attention, then variate attention, then FFN, each as
/// `x + postNorm(sublayer(preNorm(x)))`.
class TimesFM3MixingLayer: Module {
    @ModuleInfo(key: "pre_seq_attn_ln") var preSeqAttnLN: RMSNorm
    @ModuleInfo(key: "post_seq_attn_ln") var postSeqAttnLN: RMSNorm
    @ModuleInfo(key: "seq_attn") var seqAttn: TimesFM3Attention
    @ModuleInfo(key: "pre_var_attn_ln") var preVarAttnLN: RMSNorm?
    @ModuleInfo(key: "post_var_attn_ln") var postVarAttnLN: RMSNorm?
    @ModuleInfo(key: "var_attn") var varAttn: TimesFM3Attention?
    @ModuleInfo(key: "pre_ff_ln") var preFFLN: RMSNorm
    @ModuleInfo(key: "post_ff_ln") var postFFLN: RMSNorm
    @ModuleInfo(key: "ff0") var ff0: Linear
    @ModuleInfo(key: "ff1") var ff1: Linear

    init(_ config: TimesFM3Configuration) {
        let d = config.modelDims
        let eps = TimesFM3Model.rmsEps
        self._preSeqAttnLN.wrappedValue = RMSNorm(dimensions: d, eps: eps)
        self._postSeqAttnLN.wrappedValue = RMSNorm(dimensions: d, eps: eps)
        self._seqAttn.wrappedValue = TimesFM3Attention(config, useRoPE: true, causal: true)
        if config.useVariateAttention {
            self._preVarAttnLN.wrappedValue = RMSNorm(dimensions: d, eps: eps)
            self._postVarAttnLN.wrappedValue = RMSNorm(dimensions: d, eps: eps)
            self._varAttn.wrappedValue = TimesFM3Attention(config, useRoPE: false, causal: false)
        }
        self._preFFLN.wrappedValue = RMSNorm(dimensions: d, eps: eps)
        self._postFFLN.wrappedValue = RMSNorm(dimensions: d, eps: eps)
        self._ff0.wrappedValue = Linear(d, config.hiddenDims, bias: false)
        self._ff1.wrappedValue = Linear(config.hiddenDims, d, bias: false)
    }

    /// - Parameters:
    ///   - x: `[b, v, n, d]`.
    ///   - patchMasked: `[b, v, n]`, true = patch can't be attended to.
    func callAsFunction(_ x: MLXArray, patchMasked: MLXArray) -> MLXArray {
        let (b, v, n, d) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))

        // Sequence attention over n, batched across (b, v).
        let sa = seqAttn(
            preSeqAttnLN(x).reshaped(b * v, n, d), keyMasked: patchMasked.reshaped(b * v, n)
        ).reshaped(b, v, n, d)
        var h = postSeqAttnLN(sa) + x

        // Variate attention over v, batched across (b, n).
        if let varAttn, let preVarAttnLN, let postVarAttnLN {
            let vaIn = preVarAttnLN(h).transposed(0, 2, 1, 3).reshaped(b * n, v, d)
            let vaMask = patchMasked.transposed(0, 2, 1).reshaped(b * n, v)
            let va = varAttn(vaIn, keyMasked: vaMask).reshaped(b, n, v, d).transposed(0, 2, 1, 3)
            h = postVarAttnLN(va) + h
        }

        return postFFLN(ff1(relu(ff0(preFFLN(h))))) + h
    }
}

/// The `transformer_stack` module: `layers.N.*` in the checkpoint.
class TimesFM3TransformerStack: Module {
    @ModuleInfo(key: "layers") var layers: [TimesFM3MixingLayer]

    init(_ config: TimesFM3Configuration) {
        self._layers.wrappedValue = (0 ..< config.numLayers).map { _ in TimesFM3MixingLayer(config) }
    }

    func callAsFunction(_ x: MLXArray, patchMasked: MLXArray) -> MLXArray {
        var h = x
        for layer in layers {
            h = layer(h, patchMasked: patchMasked)
        }
        return h
    }
}

// MARK: - TimesFMModel 3

/// TimesFM 3.0 time series forecasting model.
///
/// A port of upstream's `TimesFM3Mlx.decode` (target-only path, no covariates). Unlike 2.5 it
/// is not autoregressive: every horizon patch is appended as a masked placeholder and the whole
/// horizon comes out of one forward pass. The variates of a series (`V` in `[B, V, T]`) are
/// forecast jointly and exchange information through variate attention.
public class TimesFM3Model: Module, TimeSeriesModel {

    /// Upstream's RMSNorm epsilon (float32 machine epsilon).
    static let rmsEps: Float = 1.1920929e-07
    /// Upstream's RevIN guard: a sigma below this divides by 1 instead.
    static let divTolerance: Float = 1e-6

    @ModuleInfo(key: "pre_transformer_resblock") var preTransformerResblock: TimesFM3ResidualBlock
    @ModuleInfo(key: "transformer_stack") var transformerStack: TimesFM3TransformerStack
    @ModuleInfo(key: "output_head") var outputHead: Linear

    let config: TimesFM3Configuration

    /// The input block reads `[values, rolled future values, masks for both]` per patch:
    /// `2 * (inputPatchLength + outputPatchLength)` features. The head emits
    /// `outputPatchLength × numQuantiles` per patch.
    public init(_ config: TimesFM3Configuration) {
        self.config = config
        self._preTransformerResblock.wrappedValue = TimesFM3ResidualBlock(
            inputDim: 2 * (config.inputPatchLength + config.outputPatchLength),
            outputDim: config.modelDims)
        self._transformerStack.wrappedValue = TimesFM3TransformerStack(config)
        self._outputHead.wrappedValue = Linear(
            config.modelDims, config.outputPatchLength * config.numQuantiles, bias: true)
    }

    /// Forecasts `predictionLength` steps (upstream `TimesFM3Forecaster.predict` with defaults).
    ///
    /// Steps:
    /// 1. Keep the last `maxContextLength` steps, left-pad with masked zeros to whole patches.
    /// 2. Fit a linear trend per variate; subtract it where it explains enough variance.
    /// 3. Append masked horizon patches, normalize every patch by the running stats through
    ///    it, and run one forward pass.
    /// 4. Re-estimate the stats at horizon patches from the model's own median predictions
    ///    (iterative CPM RevIN), denormalize, stitch the overlapping 64-step patch forecasts,
    ///    add the trend back, and sort the quantiles.
    ///
    /// - Parameters:
    ///   - input: Series `[B, V, T]`. `paddingMask` 1 = observed, 0 = missing (masked).
    ///   - predictionLength: Steps to forecast; any length, still one forward pass.
    ///   - caches: Unused; TimesFM 3 decodes in a single pass.
    /// - Returns: `mean` is the median `[B, V, H]`; `quantiles` are `[B, V, H, numQuantiles]`,
    ///   sorted ascending.
    public func forecast(
        input: TimeSeriesInput,
        predictionLength: Int,
        caches: [TimeSeriesKVCache?]
    ) -> TimeSeriesPrediction {
        let p = config.inputPatchLength
        let horizon = predictionLength
        let B = input.series.dim(0)
        let V = input.series.dim(1)

        var values = input.series.asType(.float32)
        var masked = input.paddingMask .== 0

        // 1. Truncate, then left-pad to whole patches.
        if values.dim(2) > config.maxContextLength {
            let start = values.dim(2) - config.maxContextLength
            values = values[0..., 0..., start...]
            masked = masked[0..., 0..., start...]
        }
        let padLen = (p - values.dim(2) % p) % p
        if padLen > 0 {
            values = MLX.concatenated([MLXArray.zeros([B, V, padLen]), values], axis: 2)
            masked = MLX.concatenated(
                [MLXArray.ones([B, V, padLen], type: Bool.self), masked], axis: 2)
        }
        let context = values.dim(2)
        let numCtxPatches = context / p

        // 2. Linear detrend (context part).
        let trend = detrend(values, masked: masked, context: context)
        if config.useLinearDetrending {
            let t = trendTime(from: -(context - 1), through: 0, context: context)
            let detrended = values - (trend.slope.expandedDimensions(axis: -1) * t
                + trend.intercept.expandedDimensions(axis: -1))
            values = MLX.where(trend.apply.expandedDimensions(axis: -1), detrended, values)
        }
        values = MLX.where(masked, Float(0), values)

        // 3. Horizon placeholders: masked zeros, one extra patch per extra roll.
        let extractLength = min(2 * p, config.outputPatchLength)
        let overlap = extractLength - p
        let numForecastPatches = max(Int(ceil(Double(horizon - overlap) / Double(p))), 1)
        let numHorPatches = numForecastPatches + config.rolls - 1
        let paddedH = numHorPatches * p
        values = MLX.concatenated([values, MLXArray.zeros([B, V, paddedH])], axis: 2)
        masked = MLX.concatenated(
            [masked, MLXArray.ones([B, V, paddedH], type: Bool.self)], axis: 2)

        let n = numCtxPatches + numHorPatches
        let logits = forwardLogits(
            values: values.reshaped(B, V, n, p), masked: masked.reshaped(B, V, n, p),
            numCtxPatches: numCtxPatches)  // [B, V, n, outputPatchLength, Q]

        // 4. Each forecast patch starts at the last context patch; stitch their overlaps.
        let first = numCtxPatches - 1
        let patchPreds = logits[
            0..., 0..., first ..< (first + numForecastPatches), 0 ..< extractLength]
        var out = stitch(patchPreds)[0..., 0..., 0 ..< horizon]  // [B, V, H, Q]

        if config.useLinearDetrending {
            let t = trendTime(from: 1, through: horizon, context: context)
            let future = trend.slope.expandedDimensions(axis: -1) * t
                + trend.intercept.expandedDimensions(axis: -1)
            let added = MLX.where(trend.apply.expandedDimensions(axis: -1), future, Float(0))
            out = out + added.expandedDimensions(axis: -1)
        }

        let quantiles = MLX.sorted(out, axis: -1)
        let median = quantiles[.ellipsis, config.medianIndex]
        return TimeSeriesPrediction(mean: median, quantiles: quantiles, predictionLength: horizon)
    }

    /// The model proper: normalize, embed, transform, project, denormalize.
    ///
    /// - Parameters:
    ///   - values: `[b, v, n, p]`, masked positions zeroed.
    ///   - masked: `[b, v, n, p]`, true = masked. Patches from `numCtxPatches` on are the horizon.
    /// - Returns: Denormalized per-patch forecasts `[b, v, n, outputPatchLength, numQuantiles]`.
    func forwardLogits(values: MLXArray, masked: MLXArray, numCtxPatches: Int) -> MLXArray {
        let (b, v, n, p) = (values.dim(0), values.dim(1), values.dim(2), values.dim(3))
        let o = config.outputPatchLength
        let Q = config.numQuantiles

        let stats = runningStats(values, masked: masked)
        let normed = MLX.where(masked, Float(0), revin(values, mu: stats.mu, sigma: stats.sigma))

        // Rolled "future" values only exist for past-future covariates; for targets they are
        // always masked, so that half of the input is zeros with an all-ones mask.
        let embedIn = MLX.concatenated(
            [
                normed, MLXArray.zeros([b, v, n, o]),
                masked.asType(.float32), MLXArray.ones([b, v, n, o]),
            ], axis: -1)
        let x = preTransformerResblock(embedIn.asType(outputHead.weight.dtype))

        // Only leading, fully masked patches (left padding) are hidden from attention.
        let patchMasked = masked.all(axis: -1)
        let leading = patchMasked.asType(.int32).cumprod(axis: 2) .> 0

        let raw = outputHead(transformerStack(x, patchMasked: leading)).asType(.float32)

        let refined = refineHorizonStats(
            raw: raw, stats: stats, numCtxPatches: numCtxPatches)
        let denorm = MLX.clip(
            raw * refined.sigma.expandedDimensions(axis: -1)
                + refined.mu.expandedDimensions(axis: -1),
            min: -config.valueClip, max: config.valueClip)
        return denorm.reshaped(b, v, n, o, Q)
    }

    // MARK: Normalization

    struct Stats {
        var n: MLXArray
        var mu: MLXArray
        var sigma: MLXArray
    }

    /// Running population count / mean / std through each patch (upstream `get_running_stats`).
    ///
    /// Computed with cumulative sums around each row's observed mean rather than upstream's
    /// sequential merge; the same statistics, without a per-patch loop.
    func runningStats(_ values: MLXArray, masked: MLXArray) -> Stats {
        let legit = logicalNot(masked).asType(.float32)
        let incN = legit.sum(axis: -1)  // [b, v, n]
        let rowN = MLX.maximum(incN.sum(axis: -1, keepDims: true), 1)
        let shift = ((values * legit).sum(axes: [-2, -1]).expandedDimensions(axis: -1) / rowN)
            .expandedDimensions(axis: -1)  // [b, v, 1, 1]

        let xs = (values - shift) * legit
        let cumN = incN.cumsum(axis: -1)
        let safeN = MLX.maximum(cumN, 1)
        let mean = xs.sum(axis: -1).cumsum(axis: -1) / safeN
        let variance = MLX.maximum(
            (xs * xs).sum(axis: -1).cumsum(axis: -1) / safeN - mean.square(), 0)

        let hasData = cumN .> 0
        return Stats(
            n: cumN,
            mu: MLX.where(hasData, mean + shift.squeezed(axis: -1), Float(0)),
            sigma: MLX.where(hasData, MLX.sqrt(variance), Float(0)))
    }

    /// `(x - mu) / sigma` with per-patch stats `[b, v, n]` broadcast over the last axis.
    func revin(_ x: MLXArray, mu: MLXArray, sigma: MLXArray) -> MLXArray {
        let safeSigma = MLX.where(sigma .< Self.divTolerance, Float(1), sigma)
        return (x - mu.expandedDimensions(axis: -1)) / safeSigma.expandedDimensions(axis: -1)
    }

    /// Iterative CPM RevIN (upstream `cpm_iterative_revin_refine`).
    ///
    /// Horizon patches have no observed data, so their running stats would stay frozen at the
    /// context's. Instead, walk the horizon patches in order and fold in the model's own median
    /// prediction for each one, taken from the last patch whose forecast covers it (the anchor
    /// is refreshed every `rolls` patches).
    ///
    /// - Returns: Stats `[b, v, n]` equal to the running stats on context patches and the
    ///   refined stats on horizon patches.
    func refineHorizonStats(raw: MLXArray, stats: Stats, numCtxPatches: Int) -> Stats {
        let (b, v, n) = (raw.dim(0), raw.dim(1), raw.dim(2))
        let p = config.inputPatchLength
        let rolls = config.rolls
        let median = raw.reshaped(b, v, n, rolls, p, config.numQuantiles)[
            .ellipsis, config.medianIndex]  // [b, v, n, rolls, p]

        // State after the last context patch: its own stats, and its median forecast.
        let last = numCtxPatches - 1
        var carry = Stats(
            n: stats.n[0..., 0..., last], mu: stats.mu[0..., 0..., last],
            sigma: stats.sigma[0..., 0..., last])
        var anchor = denormalizeClipped(median[0..., 0..., last], carry)
        var blockOffset = 0

        var mus = [stats.mu[0..., 0..., 0 ..< numCtxPatches]]
        var sigmas = [stats.sigma[0..., 0..., 0 ..< numCtxPatches]]
        for i in numCtxPatches ..< n {
            carry = merge(carry, anchor[0..., 0..., blockOffset])
            blockOffset = (blockOffset + 1) % rolls
            if blockOffset == 0 {
                anchor = denormalizeClipped(median[0..., 0..., i], carry)
            }
            mus.append(carry.mu.expandedDimensions(axis: -1))
            sigmas.append(carry.sigma.expandedDimensions(axis: -1))
        }
        return Stats(
            n: stats.n, mu: MLX.concatenated(mus, axis: 2),
            sigma: MLX.concatenated(sigmas, axis: 2))
    }

    /// `x * sigma + mu`, clipped, for `[b, v, rolls, p]` with stats `[b, v]`.
    private func denormalizeClipped(_ x: MLXArray, _ s: Stats) -> MLXArray {
        MLX.clip(
            x * s.sigma.reshaped(s.sigma.shape + [1, 1]) + s.mu.reshaped(s.mu.shape + [1, 1]),
            min: -config.valueClip, max: config.valueClip)
    }

    /// Merge a fully observed patch `x` (`[b, v, p]`) into running stats (upstream
    /// `update_running_stats` with no masked values).
    private func merge(_ s: Stats, _ x: MLXArray) -> Stats {
        let incN = Float(x.dim(-1))
        let incMu = x.mean(axis: -1)
        let incVar = (x - incMu.expandedDimensions(axis: -1)).square().mean(axis: -1)
        let newN = s.n + incN
        let newMu = (s.n * s.mu + incMu * incN) / newN
        let newVar =
            (s.n * s.sigma.square() + incN * incVar + s.n * (s.mu - newMu).square()
                + incN * (incMu - newMu).square()) / newN
        return Stats(n: newN, mu: newMu, sigma: MLX.sqrt(newVar))
    }

    // MARK: Trend and stitching

    struct Trend {
        var slope: MLXArray  // [b, v]
        var intercept: MLXArray  // [b, v]
        var apply: MLXArray  // [b, v] bool
    }

    /// Times `from...through` scaled by `1 / context`, as `[1, 1, count]`.
    private func trendTime(from start: Int, through end: Int, context: Int) -> MLXArray {
        (MLXArray(Int32(start) ..< Int32(end + 1)).asType(.float32) / Float(context))
            .reshaped(1, 1, -1)
    }

    /// Least-squares line over observed context values, applied only when the detrended
    /// std is below `linearDetrendingThreshold` × the original std (upstream `_detrend`).
    func detrend(_ values: MLXArray, masked: MLXArray, context: Int) -> Trend {
        let b = values.dim(0)
        let v = values.dim(1)
        guard config.useLinearDetrending else {
            let z = MLXArray.zeros([b, v])
            return Trend(slope: z, intercept: z, apply: MLXArray.zeros([b, v], type: Bool.self))
        }
        let t = trendTime(from: -(context - 1), through: 0, context: context)
        let obs = logicalNot(masked)
        let y = values
        func sum(_ a: MLXArray) -> MLXArray { MLX.where(obs, a, Float(0)).sum(axis: -1) }

        let nV = obs.asType(.float32).sum(axis: -1)
        let sumT = sum(MLX.broadcast(t, to: values.shape))
        let sumT2 = sum(MLX.broadcast(t * t, to: values.shape))
        let sumY = sum(y)
        let sumTY = sum(t * y)
        let det = nV * sumT2 - sumT.square()
        let singular = det .== 0
        let safeDet = MLX.where(singular, Float(1), det)
        let safeN = MLX.maximum(nV, 1)
        let slope = MLX.where(singular, Float(0), (nV * sumTY - sumT * sumY) / safeDet)
        let intercept = MLX.where(
            singular, MLX.where(nV .> 0, sumY / safeN, Float(0)), (sumY - slope * sumT) / safeN)

        let detr = y - (slope.expandedDimensions(axis: -1) * t + intercept.expandedDimensions(axis: -1))
        let meanY = sumY / safeN
        let stdOrig = MLX.sqrt(MLX.maximum(sum(y * y) / safeN - meanY.square(), 0))
        let meanD = sum(detr) / safeN
        let stdDet = MLX.sqrt(MLX.maximum(sum(detr * detr) / safeN - meanD.square(), 0))
        return Trend(
            slope: slope, intercept: intercept,
            apply: stdDet .< config.linearDetrendingThreshold * stdOrig)
    }

    /// Join overlapping patch forecasts `[b, v, F, outputPatchLength, Q]` into one sequence
    /// `[b, v, (F + rolls - 1) * p, Q]`, cross-fading each overlap linearly from the earlier
    /// patch to the later one (upstream `stitch_patches`).
    func stitch(_ preds: MLXArray) -> MLXArray {
        let (b, v, F, total, Q) = (preds.dim(0), preds.dim(1), preds.dim(2), preds.dim(3), preds.dim(4))
        let p = config.inputPatchLength
        let overlap = total - p
        if F == 1 {
            return preds[0..., 0..., 0]
        }
        // linspace(1, 0, overlap)
        let w = (1 - MLXArray(0 ..< overlap).asType(.float32) / Float(max(overlap - 1, 1)))
            .reshaped(1, 1, 1, overlap, 1)
        let first = preds[0..., 0..., 0, 0 ..< p]
        let prev = preds[0..., 0..., 0 ..< (F - 1)]
        let next = preds[0..., 0..., 1...]
        let blended = w * prev[0..., 0..., 0..., p...] + (1 - w) * next[0..., 0..., 0..., 0 ..< overlap]
        let middles = next[0..., 0..., 0..., overlap ..< p]
        let chunks = MLX.concatenated([blended, middles], axis: 3).reshaped(b, v, (F - 1) * p, Q)
        let tail = preds[0..., 0..., F - 1, p...]
        return MLX.concatenated([first, chunks, tail], axis: 2)
    }

    // MARK: TimeSeriesModel

    /// Checkpoint names already match this module tree.
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        weights
    }

    /// Upstream computes in float32; keep it for reference parity.
    public var inferenceDtype: DType { .float32 }

    /// Single-pass decoding: no KV caches.
    public func newCaches() -> [TimeSeriesKVCache?] { [] }
}
