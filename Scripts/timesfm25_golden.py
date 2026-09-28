"""Generate TimesFM 2.5 golden forecasts from the upstream torch reference,
using the MLX checkpoint's own (fp16) weights so the comparison isolates code differences.

Usage (needs `pip install "timesfm[torch] @ git+https://github.com/google-research/timesfm.git"`):
    python Scripts/timesfm25_golden.py <model.safetensors> Tests/MLXTimeSeriesTests/Fixtures/timesfm25_golden.json
"""
import json, sys
import numpy as np, torch
from safetensors.numpy import load_file
from timesfm.timesfm_2p5.timesfm_2p5_torch import TimesFM_2p5_200M_torch_module

ckpt = sys.argv[1]; out = sys.argv[2]
w = load_file(ckpt)
sd = {}
for k, v in w.items():
    if k.startswith("horizon_ff_layer."): continue
    if k.endswith("_ln.weight"): k = k[: -len("weight")] + "scale"
    sd[k] = torch.from_numpy(v.astype(np.float32))
m = TimesFM_2p5_200M_torch_module()
m.load_state_dict(sd, strict=True); m.eval()

t = np.arange(600, dtype=np.float64)
cases = {
    "sine_trend_512_h256": (np.sin(2*np.pi*t[:512]/48) * 3 + 0.02*t[:512], 256),
    "unaligned_200_h128": (np.cos(2*np.pi*t[:200]/24) + 0.5*np.sin(2*np.pi*t[:200]/7), 128),
    "offset_scale_256_h128": (1000 + 50*np.sin(2*np.pi*t[:256]/32), 128),
}
res = {"quantiles": [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9], "decode_index": 5,
       "note": "channel 0 = mean, 1..9 = quantiles 0.1..0.9; forecast_naive semantics (no normalize_inputs/flip)",
       "cases": []}
for name, (x, h) in cases.items():
    x = x.astype(np.float32)
    f = m.forecast_naive(h, [x])[0]  # [h, 10]
    res["cases"].append({"name": name, "horizon": h, "input": [round(float(v), 6) for v in x],
                         "forecast": [[round(float(v), 5) for v in row] for row in f]})
    print(name, f.shape, "median[:5]", np.round(f[:5, 5], 4))
json.dump(res, open(out, "w"))
