# Improvements
What to fix next, one short line each, bugs first. Remove a line once it is done, add what is found at the end. No history or measurements (those go into commit messages). Speed work: [performance.md](performance.md).

## Bugs
- A parameter without a type ignores the type of its constant default: `Combined(other = Short)` in Examples/Labeled reads `other.label` as `()` (`HeaderTokens.ValueType`).
- Traits as members or parameters break `strict build`: the call goes to the trait's own symbol (`undefined symbol: Analytics_GameStarted`).
- A bare zero-argument method as constant initializer (`constant Shared = Created`) is evaluated on every read, `Type.Created` is cached (ConstantMethods).
- Native builds sometimes leave an llvm-split part (`Name.test-3`) next to the source: `NativeBuild.Removed` ignores a failed `File.Delete`.
- A failed host allocation shows up as `AccessViolation` instead of an out of memory Error.
- `Process.RunTools` starting clang on pool threads exited 127 without a message once (Toolchain `PartsCompiled`).
- GPU launches are not guarded on pool threads (`MlirSafety` keeps GPU loops out of parallel loops instead of running them on the CPU).
- Native `NeuralNetworks/Batch` fails 3 tests since c06d0649: the left side is now exact while the right side runs on the GPU.
- `STRICT_RC_CHECK=1` self-builds of NativeCompiler report counts too low for the last 25-30 elements of one large list, and the check's marking before a reclaim lost live objects with young collections (it collects fully now).
- Ternary Bonsai 27B stalls at a simulated 10 GB only with a raw cuMemAlloc holder in another process, not with a torch holder.
- Test comparisons expand records and lists only through eight nested types, deeper types compare without numeric tolerance.
- `Linker` forwarding is not transitive (Angle to Degrees to Number never asks Number for Floor).

## Next
- GPU speed and memory: [improvements-gpu.md](improvements-gpu.md); Ternary Bonsai and Qwen3.8-Flash-Next: [improvements-models.md](improvements-models.md).
- Game engine (iOS, Android, one game loop, multiplayer, Spine, platform services): [improvements-engine.md](improvements-engine.md).
- CI runs few native builds: add more while each job stays under 10 minutes.

## Waiting
- Base library: Number `Sign`, `Wrap`, `ArcTangent2` and inverse trigonometry, Text to Numbers, fixed decimal text, `Truncated`.
- Later: Qwen3-Coder-Next 80B-A3B and Qwen3-Coder-30B with all experts, image blur on the GPU, Apple Silicon, AMD GPUs, batched training ideas.
