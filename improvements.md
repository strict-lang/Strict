# Improvements
What to fix next, one short line each, bugs first. Remove a line once it is done, add what is found at the end. No history or measurements (those go into commit messages). Speed work: [performance.md](performance.md).

## Bugs
- Building Compiler as one native test executable retains excessive memory during frontend linking; CI tests its files separately.
- Ternary Bonsai 27B at a simulated 10 GB stalled once more (first run after a cold build), 12 later runs did not: [improvements-models.md](improvements-models.md).
- A folder test whose package native library is out of date builds it in every test child at once (`NativeSources.Library`, one clang call of all sources), the children race on the rename and leave `<library>.<random>.dll` behind: build it once under a lock like `PtxCache.Waited`.
- A method without parameters named like a type leaks into other types: a `Window Number` method in DeltaEngine `Graphics3D/Route3D.strict` made `Window.Size(app.handle)` in `Games/RacingGame/RacingGame.strict` (which uses a Route3D) fail with `MethodNotFound: Number.Size, local Window hides the type Window` on the VM and natively.

## Next
- GPU speed and memory: [improvements-gpu.md](improvements-gpu.md); Ternary Bonsai and Qwen3.8-Flash-Next: [improvements-models.md](improvements-models.md).
- Game engine (iOS, Android, one game loop, Spine, platform services): [improvements-engine.md](improvements-engine.md).
- Nightly CI: build and run every program's Run once a day (games and long running apps for a few frames); work too slow for a Slow test (rebuilding all content, big models) belongs in a Run, not in tests.

## Waiting
- Base library: Number `Sign`, `Wrap`, `ArcTangent2` and inverse trigonometry, Text to Numbers, fixed decimal text, `Truncated`.
- Later: Qwen3-Coder-Next 80B-A3B and Qwen3-Coder-30B with all experts, image blur on the GPU, Apple Silicon, AMD GPUs, batched training ideas.

