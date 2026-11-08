# Improvements
Bugs to fix, most harmful first, one short line each. Remove a line once it is done, add what is found. No history or measurements (those go into commit messages). Speed work: [performance.md](performance.md). Anything large or difficult, features and far fetched things: [ideas.md](ideas.md).

- Two greedy runs of Qwen3-Coder-30B-A3B on the same prompt give different outputs (Ben, not investigated: atomics or reduction order in the expert path?).
- Qwen3.8 27B decode on the 4090 sometimes runs at 20.5 ms per token until about token 70, then stalls once for 0.7 s and continues at 27-31 ms: GPU memory reaches 22.6 of 24 GB in 256 MB steps, the same binary was fast an hour before; find what grows and what the trim then evicts.
- Once in 8 Qwen3-Coder 30B runs, right after loading, a full collection followed another (19 in a row, about 40 ms each, nothing freed) until GarbageCollectionExceedsLimit: probably the own block path of strict_gpu_obtain, whose pause after a collection is skipped when it freed one reusable block.

- GPU speed and memory: [improvements-gpu.md](improvements-gpu.md); Ternary Bonsai and Qwen3.8-Flash-Next: [improvements-models.md](improvements-models.md).
- Game engine (iOS, Android, one game loop, Spine, platform services): [improvements-engine.md](improvements-engine.md).
- A loop that stops being GPU eligible silently runs on the CPU (FractalZoomer lost its Perturbation kernel for days, several 100 times slower): report it at build time, or test in Math that Perturbation.Strip stays a kernel.
