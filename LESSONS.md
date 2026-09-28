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

## 2026-09-28 -- Checkpoint-backed test suites (TimesFM 2.5/3 golden, both Fibonacci walkthroughs) use .enabled(if: fileExists) and skip silently when weights are missing; after moving/renaming anything under converted/ or changing a checkpoint path, grep for the old path and confirm xcodebuild output shows the suite's printed golden/Fibonacci lines. Pass env vars to tests as TEST_RUNNER_<NAME>=... on the xcodebuild command line.
- Expected: Renaming converted/timesfm3 to converted/timesfm3-fp16 was a pure file move.
- Actual: The golden suite's default path still pointed at converted/timesfm3, so it would have skipped instead of failing; caught only by grepping for the old path. Separately, TIMESFM3_MLX_DIR only reached the test process when passed as TEST_RUNNER_TIMESFM3_MLX_DIR.
- Why: Suites gate on file existence, so a wrong path turns a meaningful test into a silent skip; xcodebuild only forwards TEST_RUNNER_-prefixed env vars to the test runner. Fix applied: CLAUDE.md "Build & test" section.
- tags: testing,xcodebuild,golden,paths,fix-applied

## 2026-09-28 -- If upstream ships a second backend (timesfm3.mlx alongside torch in timesfm 3.0.2), run it on the golden cases before judging the port and set tolerances above that backend-to-backend difference; for TimesFM 3 strong_trend_300_h96 it is 1.7e-3 of series std (fp32 linear-trend fit), everywhere else ~1e-5.
- Expected: The Swift port's 1.7e-3 error on strong_trend might indicate a bug.
- Actual: Upstream's own MLX backend differs from its torch backend by the same 1.7e-3 on that case; Swift matched torch to ~5e-6 on all other cases. Tolerance set to 2.5e-3 (fp32) / 5e-3 (fp16) with that justification.
- Why: Running the independent upstream backend first separated reference noise from port bugs before any tolerance was chosen. Worked; lock it in. Fix applied: CLAUDE.md "Porting or fixing a model" section.
- tags: testing,model-port,timesfm3,golden,fix-applied

## 2026-09-28 -- fp16 weight rounding cost is input-dependent: for TimesFM 3 it is 0.2-0.4% of series std on smooth series but 5.5% of the median on raw Fibonacci F(0..31) (80% interval ~14M wide). Before quoting an fp16-vs-fp32 figure, measure it on at least one high-uncertainty input, not only smooth ones.
- Expected: fp16 shifts TimesFM 3 forecasts by ~0.3% of series std (told to the user, from 3 smooth series).
- Actual: TimesFM3FibonacciTests showed fp32 3,433,327 vs fp16 3,243,824 on raw Fibonacci, a 5.5% gap; log and ratio gaps were ~1e-7 and ~3e-5.
- Why: Where the model is uncertain, its median sits on a flat part of the output distribution, so tiny weight changes move it far; the first estimate only sampled confident inputs. Context only, no file fix.
- tags: precision,fp16,timesfm3
