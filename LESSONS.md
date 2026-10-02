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

## 2026-09-28 -- Numbers written into test comments (the 'Observed' lines in TimesFMIntuitionTests.swift, Fibonacci tests) must be copied from an xcodebuild run of that exact test on that exact checkpoint; anything estimated beforehand or from a different variant has been wrong.
- Expected: Observed figures and explanations drafted alongside the 11 intuition tests would match the final runs.
- Actual: Several were wrong after the first full run: squares raw 2.5 worst error 2.9%, seasonality 2.5 log 0.99%, prime-gaps off-by-one (history included the gap being forecast), and my explanation for 2.5's narrow random-walk interval was wrong (128 steps fit one pass; 2.5 is just overconfident far ahead).
- Why: Comments were written from expectation and exploration-harness output, not from the parameterized test's own output across all four variants; a sorting step in the forecast helper had also changed the 2.5 median. Fix applied: CLAUDE.md.
- tags: testing,comments,timesfm,fix-applied

## 2026-09-28 -- When a change adds or renames a converter model type, flag, or converted/<folder> that tests depend on, update README.md (Converting Models, Running the Tests, Project Structure) in the same commit, and run or explicitly mark unverified every command you document.
- Expected: README.md, the project's front door, described how to convert the models the tests need.
- Actual: After the whole TimesFM 3 port the README still showed only Toto conversion: no --dtype (fp16 default), no --model-type, no timesfm25-fp32/timesfm3-fp32/timesfm3-fp16 folder names, no test-running instructions; commands lived only in TimesFM-Pedagogy.md and Swift doc comments. Found only when the user asked. The README fix (commit 326a5af) documented the 2.5 Hub-ID converter command without having re-run it; re-run afterwards into a scratch folder, it produced a model.safetensors byte-identical (same sha256) to converted/timesfm25-fp32, config.json identical.
- Why: Docs were added where the work was happening (pedagogy doc, test comments) and nothing prompted a README check; README is not touched by tests or builds. Fix applied: CLAUDE.md.
- tags: docs,readme,converter,fix-applied

## 2026-10-02 -- A test that writes generated files into the repo tree (TimesFMPricesTests writes Tests/Documents/forecasts/*.csv, about 3 MB each) needs its output folder in .gitignore in the same change that adds the writer; otherwise git status shows it untracked and a broad git add would commit about 12 MB of CSVs.
- Expected: The new suite would leave the repo clean apart from the code and prices.csv.
- Actual: The first run left Tests/Documents/forecasts/ untracked on pv-exp; the user had to say 'Ignore the generated forecasts' and later 'Add the forecasts directory to .gitignore' before it was handled.
- Why: The output folder was chosen next to the test data for convenience and nothing in the workflow asked whether generated files should be tracked. Fix applied: CLAUDE.md Build & test bullet.
- tags: git,testing,generated-files,fix-applied

## 2026-10-02 -- Define every metric and baseline term in one clause the first time it appears in a chat report (MAE = mean absolute error; 'no change' = repeat the last known close for every future bar; skill = 1 - MAE / no-change MAE); the user asked 'What is MAE?' and 'what does no change mean?' within one session.
- Expected: The prices results summary (MAE, skill, 'no change', move correlation) would be readable as written.
- Actual: The user stopped to ask what MAE meant, then what 'no change' meant. The test header comment and the pedagogy doc defined the terms, but the chat replies used them bare, and I used 'no change' and 'last-value baseline' for the same thing.
- Why: Definitions lived in the file comments I wrote, not in the replies the user read, and I switched between two names for one baseline. Context only, no file fix.
- tags: communication,metrics,timesfm-prices

## 2026-10-02 -- Verify a generated derived column with an independent implementation before reporting it: the Swift close_date_pacific column (ISO8601DateFormatter, America/Los_Angeles) was checked against Python zoneinfo on all 90,432 rows across the four CSVs with 0 mismatches, covering both -08:00 and -07:00 offsets.
- Expected: The new Pacific ISO column would be correct, since the formatter is a standard API.
- Actual: The check found 0 mismatches; first window is 2023-02-01T02:00:00-08:00 from epoch 1675245600, and sampled rows in July/August show -07:00.
- Why: Worked as planned; locking in the pattern. Date and timezone output is easy to get subtly wrong (offset, DST, epoch seconds vs milliseconds), and a second implementation in a different language costs one short script. Context only, no file fix.
- tags: testing,timezones,verification
