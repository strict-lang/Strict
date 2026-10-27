# Improvements
What to fix next, one short line each, bugs first. Remove a line once it is done, add what is found at the end. No history or measurements (those go into commit messages). Speed work: [performance.md](performance.md).

## Bugs
- Reproduce the reported `NeuralNetworks/Batch` CPU/GPU exact-comparison mismatch on the original GPU.
- Ternary Bonsai 27B at a simulated 10 GB stalled once more (first run after a cold build), 12 later runs did not: [improvements-models.md](improvements-models.md).
- Test comparisons of recursive types compare the repeated type with `is`, without numeric tolerance.

## Next
- Measure native constant-wrapper emission and ready-path overhead.
- GPU speed and memory: [improvements-gpu.md](improvements-gpu.md); Ternary Bonsai and Qwen3.8-Flash-Next: [improvements-models.md](improvements-models.md).
- Game engine (iOS, Android, one game loop, multiplayer, Spine, platform services): [improvements-engine.md](improvements-engine.md).
- CI runs few native builds: add more while each job stays under 10 minutes.

## Waiting
- Base library: Number `Sign`, `Wrap`, `ArcTangent2` and inverse trigonometry, Text to Numbers, fixed decimal text, `Truncated`.
- Later: Qwen3-Coder-Next 80B-A3B and Qwen3-Coder-30B with all experts, image blur on the GPU, Apple Silicon, AMD GPUs, batched training ideas.
