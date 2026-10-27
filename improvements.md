# Improvements
What to fix next, one short line each, bugs first. Remove a line once it is done, add what is found at the end. No history or measurements (those go into commit messages). Speed work: [performance.md](performance.md).

## Bugs
- Concurrent builds of one target share intermediates and replacement executable names; isolate artifacts or lock each target.
- Nested `Process.RunTools` batches overwrite and reset the global child-thread share; make capacity ownership batch-local.
- Verify linker convergence against all semantic work, not only signatures and the main method count (`LinkRound.IsSettled`).
- Detect cross-thread constant dependency cycles; per-thread `RecursiveConstant` detection cannot break mutual initialization waits.
- Ternary Bonsai 27B stalls at a simulated 10 GB only with a raw cuMemAlloc holder in another process, not with a torch holder.
- Test comparisons of recursive types compare the repeated type with `is`, without numeric tolerance.
- `strict_gpu_tiled` and `strict_gpu_eights` read a tile's rows in its first row's format (a row unpacked elsewhere gives wrong numbers); checking every row hangs `strict_device_blocks` on mixed rows with 8 bit input.
- VM unit tests over 100 ms also alone: `FlashLayer.Cached` (line 51) and `Gpt2Pass.Attending` (line 65).

## Next
- Measure native constant-wrapper emission and ready-path overhead; avoid unnecessary TLS access on cached reads.
- GPU speed and memory: [improvements-gpu.md](improvements-gpu.md); Ternary Bonsai and Qwen3.8-Flash-Next: [improvements-models.md](improvements-models.md).
- Game engine (iOS, Android, one game loop, multiplayer, Spine, platform services): [improvements-engine.md](improvements-engine.md).
- CI runs few native builds: add more while each job stays under 10 minutes.

## Waiting
- Base library: Number `Sign`, `Wrap`, `ArcTangent2` and inverse trigonometry, Text to Numbers, fixed decimal text, `Truncated`.
- Later: Qwen3-Coder-Next 80B-A3B and Qwen3-Coder-30B with all experts, image blur on the GPU, Apple Silicon, AMD GPUs, batched training ideas.
