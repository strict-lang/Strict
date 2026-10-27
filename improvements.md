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
- `strict test Compiler/Toolchain.strict` exited 127 without a message once among 16 parallel VM tests, in `PartsCompiled` (`Process.RunTools` starting clang on pool threads).
- Qwen3Benchmark native compilation intermittently exits with heap corruption after GPU preparation; keep the failing cold-build log and investigate in the compiler/runtime session.
- GPU launches are not guarded on pool threads: `MlirSafety` keeps methods with GPU-able loops out of parallel loops; the runtime should run such loops on the CPU while the pool is busy.
- Speed up boxed numeric VM allocation/dispatch for explicit Attention prompt tests (Runtime/README.md).
- Native `NeuralNetworks/Batch` fails 3 tests since c06d0649 (passed at f38592f7): `Fours.Grouped(...)(3) is Fours.Sums(Tokens)(11)`, `Fours.Grouped((0, 8), 8, Long)(523)` and `Wholes.Gated(Tokens)(0)`, whose left side now comes out exact (`Wholes.Gated(Tokens)(0)` 7579.69 like the VM instead of the GPU's 7916.48) while the right side runs on the GPU; H3Video still runs the GPU kernels.
- `STRICT_RC_CHECK=1` self-builds of Compiler/NativeCompiler report counts too low for the last 25-30 elements of one list of about 2,700 elements (MlirModule's functions?) in a few full collections, and about 100 objects of count 0 found once on the stack; also why the check's marking before a reclaim lost live objects with young collections (it collects fully now).

- A bare zero-argument method used as a constant initializer is re-evaluated: `constant Shared = Created` logs three calls across three reads, while `constant Shared = Type.Created` is cached; recognize the implicit call in ConstantMethods.
- `strict build Strict.strict` fails `Strict.Message("help")` with TestDurationMustNotExceedTwentyMilliseconds while the 209 types test in parallel (0.3 ms alone, passes with STRICT_THREADS=1); `MlirFunction` `Counter.Text` and ToolRunner's `Echo.ExecuteAll` exceed 20 ms on the VM even alone.

## Next
- GPU speed and memory (attention, 16 bit lists, memory budget, loading, generated loops): [improvements-gpu.md](improvements-gpu.md).
- Ternary Bonsai and Qwen3.8-Flash-Next speed: [improvements-models.md](improvements-models.md).
- Game engine (iOS, Android, one game loop, multiplayer, Spine, platform services): [improvements-engine.md](improvements-engine.md).
- CI runs few native builds (about 4 minutes per job): add more while each job stays under 10 minutes.
- Trees as pools with 32 bit backward offsets ([Goose](https://github.com/aardappel/goose)): measure SyntaxNode allocations per `strict build Strict.strict` first.
- Build speed: under 10 s hard limit, 2-3 s target; profile repeated linker discovery/codegen and emission, compare moderate thread counts before changing pool defaults (Compiler/README.md).
- Profile remaining compiler/VM self-test allocation and dispatch overhead; named files now have bounded VM-only allowances while native tests retain 20 ms (Runtime/README.md).
- Native File.Exists opens and closes each file before checking its directory bit; use platform metadata and measure lookup-heavy linker work.
- WebWars generated SceneAtlas/LevelAtlas constructors dominate emission; reduce large constant/record-list lowering work without changing atlas data.
- MlirCost sums expensive alternative call paths and overestimates tiny loops; calibrate branch-sensitive work estimates against private frame initialization before relaxing thresholds further.
- Linker forwarding is not transitive (Angle to Degrees to Number never asks Number for Floor) and also forwards for single-member base types and Logger, which emission's `TypeEntry.Wrapped` never does.
- Syntax validation (`FileCompiler.Parsed`) reads and parses every source serially; run it like name validation (FileCompiler has 16 methods, the syntax checks need a type of their own).
- Small builds lower and translate the whole runtime every time; cache it as bitcode (keep the 16 split parts, one clang part is slower even for Sum).
- Source discovery inventories whole packages for every build, also for small programs; narrow it to the files the source can reach.

## Waiting
- Base library: Number `Sign`, `Wrap`, `ArcTangent2` and inverse trigonometry, Text to Numbers, fixed decimal text, `Truncated`.
- Qwen3-Coder-Next 80B-A3B pruned to 18-20 GB: needs more VRAM and the `qwen3next` architecture.
- Later: Qwen3-Coder-30B with all experts, image blur on the GPU, Apple Silicon, AMD GPUs.
- Later: training ideas (several step sizes evaluated in one batched pass, Muon's orthogonalized updates) and whether batched candidate evaluation carries over to diffusion and token generation.
