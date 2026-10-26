# Improvements
What to fix next, one short line each, bugs first. Remove a line once it is attempted: if it is done it is gone, whatever is left or newly found goes in as a new line at the end. No numbering, no history and no measurements here (those go into commit messages and package READMEs). Long running topics keep their details in their own file; bugs always stay here.

## Bugs
- `strict test Compiler/MlirModule.strict` prints "Unsupported native call Strict/Bytecode/InstructionEntry.LoadBoolean" twice: constants of linked types compiled on demand (`MlirPurity.Clamped`, `GpuSamples.Absolute`) call methods the linker never saw.
- A parameter without a type ignores the type of its constant default: `Combined(other = Short)` in Examples/Labeled reads `other.label` as `()` (`HeaderTokens.ValueType` takes the constant's name as the type).
- Traits break `strict build` as members or parameters: the call goes to the trait's own symbol (`undefined symbol: Analytics_GameStarted`); resolve it to the one implementation or specialize like generic methods (`Bytecode/Specialization.strict`).
- Ternary Bonsai 27B stalls at a simulated 10 GB only with a raw cuMemAlloc holder in another process (p95 315 against 31 ms per token), not with a torch holder (p95 21-24 ms): find what differs (WDDM residency, memset pages) before blaming the streaming code.
- Native builds now and then leave an llvm-split part (`Name.test-3`, LLVM bitcode) next to the source: `NativeBuild.Removed` ignores a failed `File.Delete` (likely still open).
- Parallel `strict build`s after a compiler change all rebuild `Compiler/NativeCompiler` at once and fail linking (`NativeCompiler.res` missing): one build rebuilds, the others wait for it.
- A failed host allocation shows up as `AccessViolation` instead of an out of memory Error (H3 with 40 GB committed by other programs).
- `File.Length` on the VM returns the collection length, not the file's byte count (release verification uses `ReadBytes.Length` for now).
- `strict test Compiler/Toolchain.strict` exited 127 without a message once among 16 parallel VM tests, in `PartsCompiled` (`Process.RunTools` starting clang on pool threads).
- `strict test Compiler/ToolRunner.strict native` fails with `unsupported: text of List(IsWindow)` in `$Tests`; conditional constants infer the wrong type (also fails without timing changes).
- A method named like a base type is not what a bare name means inside its own loops: in `Bytecode/Signature`, `Signature(value).Type is Type` compared with the base type `Type` (silently false) instead of calling the method; report the ambiguity.
- Qwen3Benchmark native compilation intermittently exits with heap corruption after GPU preparation; keep the failing cold-build log and investigate in the compiler/runtime session.
- Native C object cache ignores optimization flags: switching default/small builds can reuse an object compiled at the wrong optimization level.
- Test-data discovery recursively inventories SDKs and `.git` from quoted directory fragments; separate actual fixtures from tool/VCS state before parallelizing the scan.
- Native TensorFile result buffers and GgufFile model-selection caches are shared mutable state; audit concurrent exported calls before enabling parallel loading.
- SocketChecks binds a fixed loopback port; concurrent copies need isolated ports and guaranteed cleanup on connection failure.

## Next
- GPU speed and memory (attention, 16 bit lists, memory budget, loading, generated loops): [improvements-gpu.md](improvements-gpu.md).
- Ternary Bonsai and Qwen3.8-Flash-Next speed: [improvements-models.md](improvements-models.md).
- Game engine (iOS, Android, one game loop, multiplayer, Spine, platform services): [improvements-engine.md](improvements-engine.md).
- CI runs few native builds (about 4 minutes per job): add more while each job stays under 10 minutes.
- Trees as pools with 32 bit backward offsets ([Goose](https://github.com/aardappel/goose)): measure SyntaxNode allocations per `strict build Strict.strict` first.
- Build speed: under 10 s hard limit, 2-3 s target; reduce compiler/game frontend and emission, cold GPU preparation, retain unchanged dependency IR and validation (`Compiler/README.md`).
- 20 ms test budget: move slow integration and model tests (`Compiler/NativeFolder.Counted` 8.4 s on the VM, `Runtime/Execute.Completed` 112 ms) into explicit slow suites with their assertions, then enforce the budget.

## Waiting
- Base library: Number `Sign`, `Wrap`, `ArcTangent2` and inverse trigonometry, Text to Numbers, fixed decimal text, `Truncated`.
- Qwen3-Coder-Next 80B-A3B pruned to 18-20 GB: needs more VRAM and the `qwen3next` architecture.
- Later: Qwen3-Coder-30B with all experts, image blur on the GPU, Apple Silicon, AMD GPUs.
- Later: training ideas (several step sizes evaluated in one batched pass, Muon's orthogonalized updates) and whether batched candidate evaluation carries over to diffusion and token generation.
