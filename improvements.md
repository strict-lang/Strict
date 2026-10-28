# Improvements
What to fix next, one short line each, bugs first. Remove a line once it is done, add what is found at the end. No history or measurements (those go into commit messages). Speed work: [performance.md](performance.md).

## Bugs
- Ternary Bonsai 27B at a simulated 10 GB stalled once more (first run after a cold build), 12 later runs did not: [improvements-models.md](improvements-models.md).
- A native program looping `for 42` over Moss layers with an inner `for` over 6 linear names (H3Lora.Stacked, Convrot.Rotated, TensorFile.Tuned, timings summed into mutables) fails with TensorNotFound at layer 0 (the Entry of the linear empty, one variant also found no LoRA, so a LoRA can be dropped silently); `for 1` or one more logger.Log in the inner loop pass: C:/Users/Benja/AppData/Local/Temp/claude/C--code-GitHub-strict-lang-StrictComfy/c8cc3516-b276-4c26-8866-f9137e998608/scratchpad/NestedValueBug.strict (run in ../StrictComfy with the Moss model and dmd_turbo_lora), small repros pass.

- `for Shade` over an enum (a type of only constants) fails with NotIterable, the README says it gives its constants.
- Overloads by parameter type pick the first: with `Value(name Text, default Number)` and `Value(name Text, default Text)`, `Value("b", "x")` runs the Number one ("x1"), on the VM and natively.
- `Members` natively has no values (`()` for every InitialValue), the VM has them.
- A text literal broken over two lines reports TypeNotFound for `-` at line 3 of the file instead of the line it is on.

## Next
- GPU speed and memory: [improvements-gpu.md](improvements-gpu.md); Ternary Bonsai and Qwen3.8-Flash-Next: [improvements-models.md](improvements-models.md).
- Game engine (iOS, Android, one game loop, multiplayer, Spine, platform services): [improvements-engine.md](improvements-engine.md).
- CI runs few native builds: add more while each job stays under 10 minutes.

## Waiting
- Base library: Number `Sign`, `Wrap`, `ArcTangent2` and inverse trigonometry, Text to Numbers, fixed decimal text, `Truncated`.
- Later: Qwen3-Coder-Next 80B-A3B and Qwen3-Coder-30B with all experts, image blur on the GPU, Apple Silicon, AMD GPUs, batched training ideas.

