"""Generate TimesFM 3.0 golden forecasts from the upstream torch reference,
using a converted MLX checkpoint's weights. The fixture is made from the fp32 conversion
(bit-identical to Google's release), so it is Google's exact output.
The Swift tests compare both the fp32 and fp16 conversions against it.

Usage (needs `pip install "timesfm[torch]==3.0.2"`):
    python Scripts/timesfm3_golden.py converted/timesfm3-fp32 Tests/MLXTimeSeriesTests/Fixtures/timesfm3_golden.json
"""
import json, sys, tempfile
from pathlib import Path
import numpy as np
from safetensors.numpy import load_file, save_file
from timesfm3.torch import TimesFM3Forecaster

mlx_dir = Path(sys.argv[1]); out = sys.argv[2]

# Upstream's loader wants its own config.json layout; rebuild it from ours.
c = json.load(open(mlx_dir / "config.json"))
upstream_cfg = {
    "input_patch_len": c["input_patch_len"], "output_patch_len": c["output_patch_len"],
    "quantiles": c["quantiles"], "use_variate_attention": c["use_variate_attention"],
    "use_linear_detrending": c["use_linear_detrending"],
    "linear_detrending_threshold": c["linear_detrending_threshold"], "value_clip": c["value_clip"],
    "use_stitching": True, "use_iterative_cpm_revin": True, "use_frozen_running_stats": False,
    "input_transform": "identity",
    "residual_block_config": {"activation": "relu", "dropout": 0.0, "hidden_dims": c["model_dims"],
                              "identity_skip": False, "output_dims": c["model_dims"],
                              "prenorm": "none", "use_bias": False},
    "transformer_config": {"num_layers": c["num_layers"], "use_remat": False, "transformer": {
        "attention_norm": "rms", "causal_attention": True, "debug_no_masking": False,
        "deterministic": True, "feedforward_norm": "rms", "ff_activation": "relu",
        "hidden_dims": c["hidden_dims"], "max_variates": c["max_variates"],
        "model_dims": c["model_dims"], "num_heads": c["num_heads"],
        "paired_token_skip_second": False, "qk_norm": "rms", "training": False,
        "use_bias": False, "use_memory_efficient_attention": True, "use_rope_seq": True,
        "use_rope_var": False, "use_sdpa": True, "v_norm": "none"}},
}
tmp = Path(tempfile.mkdtemp())
json.dump(upstream_cfg, open(tmp / "config.json", "w"))
save_file({k: v.astype(np.float32) for k, v in load_file(mlx_dir / "model.safetensors").items()},
          str(tmp / "model.safetensors"))
fc = TimesFM3Forecaster.from_pretrained(str(tmp), device="cpu")

t = np.arange(600, dtype=np.float64)
rng = np.random.default_rng(0)
base = np.sin(2 * np.pi * t[:320] / 40)
cases = {
    "sine_trend_512_h256": (np.sin(2*np.pi*t[:512]/48) * 3 + 0.02*t[:512], 256),
    "unaligned_200_h100": (np.cos(2*np.pi*t[:200]/24) + 0.5*np.sin(2*np.pi*t[:200]/7), 100),
    "offset_scale_256_h128": (1000 + 50*np.sin(2*np.pi*t[:256]/32), 128),
    "strong_trend_300_h96": (0.1*t[:300] + np.sin(2*np.pi*t[:300]/12), 96),
    "constant_64_h64": (np.full(64, 7.0), 64),
    # Three coupled variates, so variate attention actually mixes information.
    "multivariate_3x320_h128": (np.stack([base, 0.8*np.roll(base, 5) + 0.1*rng.normal(size=320),
                                          2 + base**2]), 128),
}
res = {"quantiles": c["quantiles"], "median_index": 4,
       "note": "TimesFM3Forecaster.predict defaults (sorted quantiles, no znorm/symmetric/positive); "
               "forecast is [variates, horizon, quantiles]",
       "cases": []}
for name, (x, h) in cases.items():
    x = x.astype(np.float32)
    r = fc.predict(x, horizon=h, return_quantiles=True)
    q = np.atleast_3d(r.quantiles) if x.ndim == 2 else r.quantiles[None]  # [V, h, 9]
    xs = np.atleast_2d(x)
    res["cases"].append({"name": name, "horizon": h,
                         "input": [[round(float(v), 6) for v in row] for row in xs],
                         "forecast": [[[round(float(v), 5) for v in step] for step in var] for var in q]})
    print(name, q.shape, "median[:5]", np.round(q[0, :5, 4], 4))
json.dump(res, open(out, "w"))
