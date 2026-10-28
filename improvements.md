# Improvements
What to fix next, one short line each, bugs first. Remove a line once it is done, add what is found at the end. No history or measurements (those go into commit messages). Speed work: [performance.md](performance.md).

## Bugs
- Ternary Bonsai 27B at a simulated 10 GB stalled once more (first run after a cold build), 12 later runs did not: [improvements-models.md](improvements-models.md).
- A constructor argument of the wrong type is not reported: `Holder(value).count` with `has count Number` inside `for ("a", "bc")` gives 97bc2 on the VM, `NativeLibrary(value)` with the Text value of a loop instead of the NativeSources (`outer`) crashed the native compiler with AccessViolation.
- `Members` natively has no values (`()` for every InitialValue), the VM has them.
- A GPU loop reads outside its list once a flat list passes 2^24 numbers (StrictComfy `H3Vae.Canvas` with `size` hoisted: a tiny 16x16 latent window of 19.4M numbers fails, 12x12 of 10.4M matches the CPU): index arithmetic must stay exact. A copy of Canvas with 19.35M synthetic tile numbers matches the CPU on master (2026-10-28), StrictComfy's exact failing code is needed.
- A `List(Numbers)` of GPU lists (one list per tile in `H3Vae.Canvas`) reads 0 on the GPU without an error. 4 tiles computed on the GPU read through `tiles(t)(i)` in a GPU loop are right on master (2026-10-28), StrictComfy's exact failing code is needed.

## Next
- GPU speed and memory: [improvements-gpu.md](improvements-gpu.md); Ternary Bonsai and Qwen3.8-Flash-Next: [improvements-models.md](improvements-models.md).
- Game engine (iOS, Android, one game loop, Spine, platform services): [improvements-engine.md](improvements-engine.md).
- Nightly CI: build and run every program's Run once a day (games and long running apps for a few frames); work too slow for a Slow test (rebuilding all content, big models) belongs in a Run, not in tests.

## Waiting
- Base library: Number `Sign`, `Wrap`, `ArcTangent2` and inverse trigonometry, Text to Numbers, fixed decimal text, `Truncated`.
- Later: Qwen3-Coder-Next 80B-A3B and Qwen3-Coder-30B with all experts, image blur on the GPU, Apple Silicon, AMD GPUs, batched training ideas.

