# Performance
Measured speed work, compiler front end first. One item per finding, removed when done or no longer true.

## Front end, checks, tests and the server
- Native builds after a new strict.exe reuse 0 emitted functions (MossWeb: 613 emitted again, 51.6 s instead of 39.9 s): every emission key holds CodegenStamp (modified times of root *.strict, Compiler/, Bytecode/, Expressions/ and NativeRuntime*.mlir); needs a native file content hash, leaving out validation-only files, or a stamp per file of what codegen reaches.
- Cache loading: a fresh `.strictbinary` decodes all 391 linked types (3.2 MB of bytecode down to every instruction, about 2.3 million allocations) although the links ask for 4024 methods: decode only the selected methods, share the stamp dictionary across links (0.14-0.24 s for 6648 source names), and use the cache for `strict <file>` (135 ms for Sum).
- Compiler self-build front end: decoding the homes of new types (`BinaryFile.Head`, `ZipReader.Copied` turns every byte into a Number, `BinaryLinks.Owned`), `LinkRound.Forwarded` (`Indices` again per recursion), `List.Unique` and appends of the 4500 targets per pass, and GPU preparation (`Attending.Used` scans the whole program once per kind, two or three host text copies in `Embedded`/`Referenced`).
- `strict check .` of DeltaEngine (12.7 s warm): `SourceFiles.Of` 16 % (every reached file is read again by SourceFiles, ResolveCheck, SureMethods, InputOutputs and Counts), body parsing 20 %, `ResolveCheck.Inference` 13 %, `SureMethods.Named` 13 %, `ValidateCheck.Mutations` 10 %; the 42 folders rebuild these per folder.
- Resident builds forward to the server but repeat MLIR emission (1.45 s), splitting the module (0.8 s) and link passes (0.9 s) per build of a big program: keep decoded homes or per-function MLIR text across requests.
- `strict test RealTimeStrategy native` (DeltaEngine, 73 files) after one touched file (16.6-17.3 s): struct lanes 3.2-3.9 s (lane state is texts like "3 Pair" keyed by register numbers: keep laned registers as numbers), `CallGraph.Closure` in `EmissionCache.Keys` (`Marked` copies the dictionary parameter, rebuilding the index), dependency linking (`Completed`, `Forwarded`, `Linkable` and round lists), and name validation reading every linked source again per file.
- Tests of stale files in `strict check` run serially in one process (`Execute.Failures`: 4.93 s for 109 files in Examples, 58.5 s for 150 files in Ben): verify stale files in parallel without racing on `BinaryLinks.Grown` rewriting shared dependency `.strictbinary` files.
- Trees as pools with 32 bit backward offsets ([Goose](https://github.com/aardappel/goose)): count SyntaxNode allocations of `strict build Strict.strict` first.
- `Up to date` could check a pid stamp written by each running program (like `obj/strict.server*`) instead of a 0.55 s process query.

## Native builds (emission, lowering, linking)
- `MlirPurity.IsReachingGpu` and kernel selection for every program should only gate or offload loops whose counts can reach the GPU threshold (`IsDevice` today counts any device-capable inner loop), checked on NeuralNetworks and StrictComfy.
- Every raw loop is emitted six times (one copy per packed format its lists may have at run time plus the plain one) next to its generic body: measure code size and lowering time impact.
- `MlirDefaults.ParameterName` (57 call sites, 0.31-1.68 s summed over parallel body emission) rebuilds `ParameterTexts` over all methods per call, and `MlirConstants.EscapedLength` encodes the text a second time.
- Host lowering with 0 parts changed (1.5 s per build of Strict.strict): `ReadLines` 0.39 s (pass emitter text to `MlirSplit` without the file round-trip), `Text.Combine` of each key 0.20 s, `Starts`/`Shared`/`Placed` 0.13 s, `Written` 0.08 s.
- `MlirWholeCall.IsMatched` analyses the callee from scratch at every call site (13397 calls per cold `build Strict.strict windows`).
- Programs and releases with ThinLTO cross-module importing regenerate every module importing from an edited part (64 parts mix types by hash); one module per type needs each part to declare only what it mentions.
- Web rebuilds run whole-program emcc link with Closure (2.3 s) and wasm-opt (1.6 s) even when only one string changed.
- Build speed (under 10 s hard limit, 2-3 s target; LogoApp, Blocks and Breakout 3.6-4 s): a build with tests runs the front end twice (test gate, then native), lowering and ThinLTO link are the largest phases left.
- Reachability, safety and globals/text assembly of MLIR emission run serially.

## VM
- VM loops of small Number methods (20-50x slower than native): `Machine.Bound` (4 us), `Execute` (3 us), routing (4 us), `VmValue.OfNumber` allocations and `strict_rc_reclaim` (18 %); `DynamicSample.Inflated` is still around its 100 ms limit.
- VM calls dominate game logic (a member read through a method like `Vector2.X` is two calls): DeltaEngine's multi-tick simulation and content loading tests are Slow for it.

## Compiled CPU loops, lists and texts
- Serial loops still generic (`strict_hoisted_read`/`strict_list_read`, `strict_is_failure`, `strict_list_push`): gathers `table(value)`, calls that can fail or read lists (`MersenneTwister.Normals`), one thread of a parallel chunk, and raw `Tempered` calls (f64 body has two remainders, no whole-number version used).
- Appending in a loop still creates a new view per append (~30 ns) when a list starts from a member (`MersenneTwister.Uints`) or is handed to a constructor afterwards (`MersenneTwister.Seeded`).
- Texts: `File.ReadLines` copies its lines in the runtime; `Trim`, case changes, `to Number` and sorting copy a span first; texts under 256 bytes copy their substrings.
- Short parallel loops: the 10 us probe hands a loop over even when the rest takes less than wake-up, every piece starts with a memo miss, and cheaper small loops need work kept on fewer hot workers per CCX.
- `List.Sorted`: MSVC and wasm `qsort` call the comparator per comparison (195 us on Windows, 952 us on web vs 54 us on Linux for 1187 numbers): inline comparison sort in the runtime, and measure `Sampling.Threshold` against one sort.
- Matrix and vector speed: the VM calculates in f64, accessors (`point.X`) are not counted as 32 bit values, `Axis(n).Y` still allocates its Vector3, and a block is only reused when the local is alone.
- Whole-number call guards give up (f64 version runs) when a variable read after an `if` or loop that stores it is used in `+`, `-` or `*`: a maximum over the possible stores would keep the i64 path.
- ParticleStress at 1 million particles (4.8-5.5 ms per tick): 32 bit members (72 -> 44 bytes per particle) and writing in place over the list being read.
- A `for` that drops items collects objects in parallel, but numbers and flat elements stay on one thread (no hole marker): keep a bitmap of the kept slots.
- Generated CPU loops still per element: reads of non-hoisted lists (`strict_list_read`), checks on instances, chunks with inner loops, serial loops, and f64 numbers and indices.
- Reading a packed list through `strict_list_items` (chunks without inner loops over Float8, byte, lane or quantized lists, quantized formats read by index, Float8 and quantized arguments of native library calls) replaces it with an f64 copy for good (`strict_list_unpacked`).
- A call with whole-number arguments dispatches per element between its i64 (`_whole`) and f64 versions (a vectorized loop computes both and blends: 55 ms vs 14 ms inline over 10 million numbers).

## Collector
- Lists of numbers made by generic list paths (`strict_list_add`, `strict_list_new`) are scanned by the collector (on wasm halves of numbers look like references): scan only up to their items like `strict_numbers_new` lists.
- Realtime frames: mark incrementally over several frame ends when a full collection takes longer than a few ms.
- `GarbageCollectionExceedsLimit` and `GarbageCollectionAboveTarget` could name the largest allocating call sites of the window.

## GPU
- First GPU context initialization: ~149 ms on 4090 (`STRICT_GPU_WARM=0`: `cuInit` + `cuDevicePrimaryCtxRetain` + PTX load), reduced to ~26 ms (`32.2 vs 6.43 ms`) by `strict_gpu_warm` when a `near` loop precedes the first GPU loop; remaining driver init only fires when the very first loop goes straight to GPU.
- Long-context decode attention (`strict_device_attention_decode`, `positions > 128`): split `strict_device_attention_long` from prompt kernel, streamed `merge` 4 floats at a time, dropped KV operand double-buffering and Round-1 pre-barrier, and lowered `allPartials` to `128` (`255` regs with spills -> `233` regs, `0` spills; `Qwen3-8B` `b=32` `pos149` `0.133 -> 0.061 ms`). Further gains require 4-warp (`128`-thread) or `<= 128`-reg kernel for `Qwen3-Coder 30B` (`pos > 2000`).
- Attention routing planner reads all pooled keys of a head in every block: compute moments once per head and scores as a small product kernel (dump plan lengths before trying again).
- 16 bit lists: compiled loops index in 64 bits; use 32 bit index math where counts fit.
- GPU memory: GPU-only lists cannot be spilled, so activations kept across streamed blocks take every byte from the weights.
- GPU memory: a reference in a local stays alive after its last use in a callee-saved register or spill slot (conservative stack scan).
- GPU memory: NVFP4 matrices (`TensorFile_Fours`) load as 8 bit rows (27 instead of 15.7 GB): keep e2m1 rows with their block scales on the GPU.
- Loading with LoRAs (H3 11.3-12.4 s vs 6.4-6.5 s without): read the first matrix of each block during the last merge of the previous block, and avoid round-tripping every merged matrix to the GPU and back (22 GB each way).
- WebGPU: `strict_convolution` runs at ~6 TFLOPS on a 3080 (30 peak).
- GPU kernels: bounds, reduction and tail checks remain: extend the four-wide loads (`MlirBatched`) and fuse memory passes.
- CUDA convolution: stages of 64 channels or 3 buffers once shared memory allows it (convolutions are 5.9 % of Moss).
- Uniform branches in GPU loops: `tokens_blocks` switches on row bits per block, `products_run` runs bits chains per element, `strict_device_dots` has no bfloat16 arm, prompt attention masks every tile.
- Small number types as kernel values: still on CPU for a `for` over `List(Complex)` (a list member with `value.Bits is 32` fails the build, index it instead), a `List(Complex)` result, a method returning `Complex` that has a loop or an early return, and list-backed types (`Vector2`, `Vector3`).
