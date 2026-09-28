Read LESSONS.md before starting work.

## Build & test
- Build/test only with xcodebuild, e.g.
  `xcodebuild test -scheme mlxtoto-Package -destination 'platform=macOS' -only-testing:MLXTimeSeriesTests/<Suite>`
  `swift build` / `swift test` fail compiling mlx-swift's .metal files.
- On a fresh Xcode, run `xcodebuild -downloadComponent MetalToolchain` first.

## Porting or fixing a model
- Before writing tests, diff the checkpoint's safetensors header shapes against
  `model.parameters().flattened()` shapes; every mismatch is a bug.
- Validate forecasts against the upstream reference run on the checkpoint's own
  weights (see Scripts/timesfm25_golden.py). Random-weight unit tests only prove shapes.
