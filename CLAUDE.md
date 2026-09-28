Read LESSONS.md before starting work.

## Build & test
- Build/test only with xcodebuild, e.g.
  `xcodebuild test -scheme mlxtoto-Package -destination 'platform=macOS' -only-testing:MLXTimeSeriesTests/<Suite>`
  `swift build` / `swift test` fail compiling mlx-swift's .metal files.
- On a fresh Xcode, run `xcodebuild -downloadComponent MetalToolchain` first.
- Checkpoint-backed suites (TimesFM 2.5/3 golden, Fibonacci) skip silently when weights
  are missing. After moving/renaming anything under converted/ or changing a checkpoint
  path, grep for the old path and confirm xcodebuild output shows the suite's printed
  golden/Fibonacci lines, not a skip.
- Pass env vars to tests as `TEST_RUNNER_<NAME>=...` on the xcodebuild command line.

## Porting or fixing a model
- Before writing tests, diff the checkpoint's safetensors header shapes against
  `model.parameters().flattened()` shapes; every mismatch is a bug.
- Validate forecasts against the upstream reference run on the checkpoint's own
  weights (see Scripts/timesfm25_golden.py). Random-weight unit tests only prove shapes.
- If upstream ships a second backend (e.g. timesfm3.mlx alongside torch), run it on the
  golden cases first and set test tolerances above that backend-to-backend difference
  (TimesFM 3 strong_trend: 1.7e-3 of series std).
