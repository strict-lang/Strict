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
- Stale list slots past `used` break counts: `strict_list_push` skips counting when the slot already holds the same bits (meant for prewired flat lists) and releases any other stale value, but `strict_list_into` reuse sets `used` to 0 and leaves stale bits that may be uncounted (stores inside pool jobs pin instead of count) or point at a freed and reallocated address. `STRICT_RC_CHECK=1` self-builds show it: the last ~40 of 2673 pushes of one list end at count 0. Fix: reuse releases and clears the stale slots of other allocations (keeping flat prewiring), or push trusts only inline flat elements and never releases stale slots.
- Why the check marking before a reclaim (`strict_rc_mark_all`) lost live objects with young collections (STRICT_RC_CHECK collects fully now).
- Ternary Bonsai 27B stalls at a simulated 10 GB only with a raw cuMemAlloc holder in another process, not with a torch holder.
- Test comparisons expand records and lists only through eight nested types, deeper types compare without numeric tolerance.
- `Linker` forwarding is not transitive (Angle to Degrees to Number never asks Number for Floor).

## Next
- `strict_testbudget_extended` (runtime) is unused by current code, remove it with the next seed regeneration.
- GPU speed and memory: [improvements-gpu.md](improvements-gpu.md); Ternary Bonsai and Qwen3.8-Flash-Next: [improvements-models.md](improvements-models.md).
- Game engine (iOS, Android, one game loop, multiplayer, Spine, platform services): [improvements-engine.md](improvements-engine.md).
- CI runs few native builds: add more while each job stays under 10 minutes.

## Waiting
- Base library: Number `Sign`, `Wrap`, `ArcTangent2` and inverse trigonometry, Text to Numbers, fixed decimal text, `Truncated`.
- Later: Qwen3-Coder-Next 80B-A3B and Qwen3-Coder-30B with all experts, image blur on the GPU, Apple Silicon, AMD GPUs, batched training ideas.
