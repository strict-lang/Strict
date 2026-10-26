# Improvements
What to fix next, one short line each, bugs first. Remove a line once it is attempted: if it is done it is gone, whatever is left or newly found goes in as a new line at the end. No numbering, no history and no measurements here (those go into commit messages and package READMEs). Long running topics keep their details in their own file; bugs always stay here.

## Bugs
- `strict test Compiler/MlirModule.strict` prints "Unsupported native call Strict/Bytecode/InstructionEntry.LoadBoolean" twice: constants of linked types compiled on demand (`MlirPurity.Clamped`, `GpuSamples.Absolute`) call methods the linker never saw.
- A parameter without a type ignores the type of its constant default: `Combined(other = Short)` in Examples/Labeled reads `other.label` as `()` (`HeaderTokens.ValueType` takes the constant's name as the type).
- Traits break `strict build` as members or parameters: the call goes to the trait's own symbol (`undefined symbol: Analytics_GameStarted`); resolve it to the one implementation or specialize like generic methods (`Bytecode/Specialization.strict`).
- Ternary Bonsai 27B stalls on 10 GB cards (p95 315 ms per token against 31 with 24 GB), not from collections (5 full ones either way): check uploads of evicted weights.
- Native builds now and then leave an llvm-split part (`Name.test-3`, LLVM bitcode) next to the source: `NativeBuild.Removed` ignores a failed `File.Delete` (likely still open).
- Parallel `strict build`s after a compiler change all rebuild `Compiler/NativeCompiler` at once and fail linking (`NativeCompiler.res` missing): one build rebuilds, the others wait for it.
- A GPU loop calling a method that only wraps one summing call (`Normalization.Scale(Normalization.Squared(...))`) sums the whole group again per element (Moss sampling 4.0 -> 17.5 s): let `MlirGroups.Adding` look through such a callee, with a MlirGroups test.
- `Compiler/Bootstrap.mlir` predates the 11 attention arguments: MossGenerate built by the stage 0 compiler of `Bootstrap.cmd` sampled for over 100 s; regenerate it from a fixpoint build.
- A failed host allocation shows up as `AccessViolation` instead of an out of memory Error (H3 with 40 GB committed by other programs).
- `File.Length` on the VM returns the collection length, not the file's byte count (release verification uses `ReadBytes.Length` for now).
- `strict test Compiler/Toolchain.strict` exited 127 without a message once among 16 parallel VM tests, in `PartsCompiled` (`Process.RunTools` starting clang on pool threads).

## Next
- GPU speed and memory (attention, 16 bit lists, memory budget, loading, generated loops): [improvements-gpu.md](improvements-gpu.md).
- Ternary Bonsai and Qwen3.8-Flash-Next speed: [improvements-models.md](improvements-models.md).
- Game engine (iOS, Android, one game loop, multiplayer, Spine, platform services): [improvements-engine.md](improvements-engine.md).
- CI runs few native builds (about 4 minutes per job): add more while each job stays under 10 minutes.
- Trees as pools with 32 bit backward offsets ([Goose](https://github.com/aardappel/goose)): measure SyntaxNode allocations per `strict build Strict.strict` first.
- Build speed, goal under 10 s for any build: unchanged builds skip in 55-120 ms and runtime PTX is shared, but source rebuilds take seconds (MossGenerate 5.5-6.3 s: frontend 1.6, emission 1.3, host LLVM and linking 1.9, dependency snapshot 0.5) and fresh Boolean and example builds exceed 10/100 ms; keep unchanged dependency IR and validation, profile emission and LLVM (`Compiler/README.md`).
- 20 ms test budget: move slow integration and model tests (`Compiler/NativeFolder.Counted` 8.4 s on the VM, `Runtime/Execute.Completed` 112 ms) into explicit slow suites with their assertions, then enforce the budget.

## Waiting
- Base library: Number `Sign`, `Wrap`, `ArcTangent2` and inverse trigonometry, Text to Numbers, fixed decimal text, `Truncated`.
- Qwen3-Coder-Next 80B-A3B pruned to 18-20 GB: needs more VRAM and the `qwen3next` architecture.
- Later: Qwen3-Coder-30B with all experts, image blur on the GPU, Apple Silicon, AMD GPUs.
