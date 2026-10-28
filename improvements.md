# Improvements
What to fix next, one short line each, bugs first. Remove a line once it is done, add what is found at the end. No history or measurements (those go into commit messages). Speed work: [performance.md](performance.md).

## Bugs
- Ternary Bonsai 27B at a simulated 10 GB stalled once more (first run after a cold build), 12 later runs did not: [improvements-models.md](improvements-models.md).
- `constant Shared = Created` with `Created` a method of the same type is typed `Created` (HeaderTokens.ValueType takes the identifier as the type), its test's `Code` call then fails linking with `UnresolvedLink: Strict/Created.Code` (Runtime/check_tests.py, LazyFixture).
- Native builds take far more memory than their output: the tests of Math/Matrix.strict (2 MB of MLIR, one test runner of 12848 lines) peak at 7.7 GB in NativeCompiler's function body emission, the Math folder at 20 GB (16 GB CI runners die, Windows swaps for 6 min), so CI tests Math file by file. Emitting in batches of 16 functions did not lower it.

## Next
- GPU speed and memory: [improvements-gpu.md](improvements-gpu.md); Ternary Bonsai and Qwen3.8-Flash-Next: [improvements-models.md](improvements-models.md).
- Game engine (iOS, Android, one game loop, multiplayer, Spine, platform services): [improvements-engine.md](improvements-engine.md).
- CI natively: the Compiler folder (2743 tests, 171 s here, too long next to a bootstrap) and Math as one folder (emission memory above); the root types are in no CI job (a folder test of `.` takes the whole repository).
- Nightly CI: build and run every program's Run once a day (games and long running apps for a few frames); work too slow for a Slow test (rebuilding all content, big models) belongs in a Run, not in tests.

## Waiting
- Base library: Number `Sign`, `Wrap`, `ArcTangent2` and inverse trigonometry, Text to Numbers, fixed decimal text, `Truncated`.
- Later: Qwen3-Coder-Next 80B-A3B and Qwen3-Coder-30B with all experts, image blur on the GPU, Apple Silicon, AMD GPUs, batched training ideas.

