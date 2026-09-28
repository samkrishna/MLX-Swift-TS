# Lessons

Written by /aar-loop after each session's After Action Review. Read this file before starting a new task in this project. Every entry should be concrete and checkable, never vague.

## 2026-09-28 -- Build and test MLX-Swift-TS only with xcodebuild (xcodebuild test -scheme mlxtoto-Package -destination 'platform=macOS' -only-testing:MLXTimeSeriesTests/<Suite>); swift build/swift test fail compiling mlx-swift's .metal files, and on a fresh Xcode 27 install run 'xcodebuild -downloadComponent MetalToolchain' first.
- Expected: swift build --build-tests would compile and run the TimesFM tests.
- Actual: swift build failed with CompileMetalFile errors in mlx-swift; xcodebuild then failed with 'missing Metal Toolchain'. The user downloaded it (838.9 MB, Metal Toolchain 27A266a) by running `! xcodebuild -downloadComponent MetalToolchain` in the session, after which xcodebuild tests passed.
- Status: The Metal Toolchain is now installed on this machine (Xcode 27.0.0), so the download step only applies after a fresh Xcode install or on another machine.
- Why: mlx-swift's Metal shaders only compile through Xcode's build system, and Xcode 27 ships without the Metal Toolchain. Fix applied: CLAUDE.md "Build & test" section.
- tags: build,xcodebuild,metal,mlx,fix-applied

## 2026-09-28 -- When porting or fixing a model, first diff the checkpoint's safetensors header shapes against model.parameters().flattened() shapes, then validate forecasts against the upstream reference run on the checkpoint's own weights (Scripts/timesfm25_golden.py pattern); tiny random-weight unit tests only prove shapes and inherit the code's assumptions.
- Expected: 14 new TimesFM unit tests passing meant the model was sound enough to build on.
- Actual: They passed while the model was structurally wrong: tokenizer read one 64-step patch instead of 32 values + 32 mask, heads were chained not parallel, RoPE half-split and order wrong, per-dim scale and FFN activation wrong. Found only by reading the checkpoint header ([1280, 64] tokenizer) and upstream source.
- Why: The tests used a config derived from the code under test, so they encoded the same wrong tokenizer-width assumption; nothing independent was compared. Fix applied: CLAUDE.md "Porting or fixing a model" section.
- tags: testing,model-port,timesfm,golden,fix-applied
