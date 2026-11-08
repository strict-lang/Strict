# Improvements
Bugs to fix, most harmful first, one short line each. Remove a line once it is done, add what is found. No history or measurements (those go into commit messages). Speed work: [performance.md](performance.md). Anything large or difficult, features and far fetched things: [ideas.md](ideas.md).

- Ben-Home-PC: two greedy runs of Qwen3-Coder-30B-A3B on the same prompt give different outputs (Ben, not investigated: atomics or reduction order in the expert path?).
- Ben-Home-PC: once in 8 Qwen3-Coder 30B runs, right after loading, a full collection followed another (19 in a row, about 40 ms each, nothing freed) until GarbageCollectionExceedsLimit: probably the own block path of strict_gpu_obtain, whose pause after a collection is skipped when it freed one reusable block.
- `strict check NeuralNetworks` reports 8 issues: GpuMemory unused stub parameters, Safetensors.Tuned, Qwen3.Forwarded and QwenSafetensors.Blocks over 12 lines, a Safetensors line over 120 characters, QwenSafetensors with 19 methods.

- GPU speed and memory: [improvements-gpu.md](improvements-gpu.md); Ternary Bonsai and Qwen3.8-Flash-Next: [improvements-models.md](improvements-models.md).
- Game engine (iOS, Android, one game loop, Spine, platform services): [improvements-engine.md](improvements-engine.md).
- A `constant` after a method (DeltaEngine had three) only surfaces as UnresolvedLink at a user of it, `strict check` should name it. `Warning: GarbageCollectionAboveTarget` is the last runtime warning, Strict has no warnings.
