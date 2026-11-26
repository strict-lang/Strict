# Performance
Measured speed work, compiler front end first. One item per finding, removed when done or no longer true.

## Front end, checks, tests and the server
- Cache loading: `BinaryFile.Current` on `Qwen3Benchmark.strictbinary` (`93` linked types) still spends time in `EntryDecoder.Of` (5x `1028 -> 787 ms` via `ByteEncoder.Integers` 1-byte varint fast path; `1.52 s` -> ~`1.16 s` across 72 links) and `0.78 s` in `Head` (`ZipReader.Copied`); `EntryDecoder.Select` now skips decoding unrequested methods (`890 ms` vs `1052 ms`), and `strict Examples/Sum.strict` already uses cache (`91.2 ms` warm). Remaining win: decode varints/names natively.
- Compiler self-build front end: decoding the homes of new types (`BinaryFile.Head`, `ZipReader.Copied` turns every byte into a Number, `BinaryLinks.Owned`), `List.Unique` and appends of the 4500 targets per pass, and GPU preparation (two or three host text copies in `Embedded`/`Referenced`; `Attending.Used` now fast-exits when `@strict_gpu_` is absent, `86 -> 39 ms` on `Bootstrap.mlir`).
- `strict check .` of DeltaEngine (7.1 s warm on 2026-11-25: validation 5-6 s, dependency cycles 0.4 s, project rules skipped by their stamps; was `12.7 s`): `SourceFiles.Of` (every reached file is read again by SourceFiles, ResolveCheck, SureMethods, InputOutputs and Counts), body parsing, `ResolveCheck.Inference`, `SureMethods.Named` and `ValidateCheck.Mutations`; the 42 folders rebuild these per folder. Warm `strict check .` of Strict is 3.4 s (was 6.5 s): validation 2.5 s is the per-folder keys, a fresh process tokenizes the mentions of every file once (~1 s, the server's Memo keeps them but its check is not faster), traversal ~0.7 s and 15000 `File.Modified` stamps ~0.4 s.
- Resident builds forward to the server but repeat MLIR emission (1.45 s), splitting the module (0.8 s) and link passes (0.9 s) per build of a big program: keep decoded homes or per-function MLIR text across requests.
- `strict test RealTimeStrategy native` (DeltaEngine, 73 files) after one touched file (16.6-17.3 s): struct lanes 3.2-3.9 s (lane state is texts like "3 Pair" keyed by register numbers: keep laned registers as numbers), `CallGraph.Closure` in `EmissionCache.Keys` (`Marked` copies the dictionary parameter, rebuilding the index), dependency linking (`Completed`, `Forwarded`, `Linkable` and round lists), and name validation reading every linked source again per file.
- Trees as pools with 32 bit backward offsets ([Goose](https://github.com/aardappel/goose)): count SyntaxNode allocations of `strict build Strict.strict` first.

## Native builds (emission, lowering, linking)
- `MlirPurity.IsReachingGpu` and kernel selection for every program should only gate or offload loops whose counts can reach the GPU threshold (`IsDevice` today counts any device-capable inner loop), checked on NeuralNetworks and StrictComfy.
- Every raw loop is emitted six times (one copy per packed format its lists may have at run time plus the plain one) next to its generic body: measure code size and lowering time impact.
- `MlirWholeCall.IsMatched` analyses the callee from scratch at every call site (13397 calls per cold `build Strict.strict windows`).
- Web rebuilds run whole-program emcc link with Closure (2.3 s) and wasm-opt (1.6 s) even when only one string changed.
- Build speed (under 10 s hard limit, 2-3 s target; LogoApp, Blocks and Breakout 3.6-4 s): a build with tests runs the front end twice (test gate, then native), lowering and ThinLTO link are the largest phases left.
- Reachability, safety and globals/text assembly of MLIR emission run serially.

## VM
- A VM call of a small Strict method costs about 6 us: `Machine.Bound` a quarter of it (one `Member` call per member slot, Number has 12 with its constants, and a `strict_list_push` per slot), the body about 150 ns per instruction (`Frame.Finish`, `strict_list_set`, `strict_rc_container`).
- VM calls dominate game logic (a member read through a method like `Vector2.X` is two calls): DeltaEngine's multi-tick simulation and content loading tests are Slow for it. `position + Vector2(1, 2) * 0.75` costs about 20 us: `Rounding.Build` 30 % (every construction fills all 10 member items, 8 of them constants through `Natives.Initial`, decodes the parameter names and rounds X and Y through `Number.Rounded`), `Machine.Bound` 16 %.

## Compiled CPU loops, lists and texts
- Serial loops still generic (`strict_hoisted_read`/`strict_list_read`, `strict_is_failure`, `strict_list_push`): gathers `table(value)`, calls that can fail or read lists (`MersenneTwister.Normals`), one thread of a parallel chunk, and raw `Tempered` calls (f64 body has two remainders, no whole-number version used).
- Appending in a loop still creates a new view per append (`strict_list_add`, about 130 ns against 15-30 ns pushed, 3 million appends 390-570 ms) when the list is handed to a call or constructor after the loop (`MersenneTwister.Seeded`).
- Texts: a substring of a plain text under 256 bytes is copied (Substring, a trimmed or cut text that is no span), Upper and Lower always copy.
- Short parallel loops: the 10 us probe hands a loop over even when the rest takes less than wake-up, every piece starts with a memo miss, and cheaper small loops need work kept on fewer hot workers per CCX.
- Matrix and vector speed: the VM calculates in f64, accessors (`point.X`) are not counted as 32 bit values, `Axis(n).Y` still allocates its Vector3, and a block is only reused when the local is alone.
- Whole-number call guards give up (f64 version runs) when a variable read after an `if` or loop that stores it is used in `+`, `-` or `*`: a maximum over the possible stores would keep the i64 path.
- ParticleStress at 1 million particles (4.8-5.5 ms per tick): 32 bit members (72 -> 44 bytes per particle) and writing in place over the list being read.
- Generated CPU loops still per element: reads of non-hoisted lists (`strict_list_read`), checks on instances, chunks with inner loops, serial loops, and f64 numbers and indices.
- Reading a packed list through `strict_list_items` (chunks without inner loops over Float8, byte, lane or quantized lists, quantized formats read by index, Float8 and quantized arguments of native library calls) replaces it with an f64 copy for good (`strict_list_unpacked`).

## Collector
- Realtime frames: mark incrementally over several frame ends when a full collection takes longer than a few ms.
- `GarbageCollectionExceedsLimit` and `GarbageCollectionAboveTarget` could name the largest allocating call sites of the window.

## GPU
- Loading with LoRAs (H3 with the turbo v4 LoRA 12.3 s vs 5.9 s without, warm, 2026-11-25): the GPU is busy 2.4 of 14 s (merge kernel 0.63 s, 22.6 GB up and 21.7 GB down in copies 1.8 s), the main thread spends 67 % in H3Block.Merge: reading each LoRA's factors serially (MossLora.Loaded -> Safetensors.Values -> TensorFile_Floats, 20 %), MossLora.Pack (15 %) and strict_gpu_tune (33 %, mostly host work in a parallel loop around the merge). Read the next matrix's factors while the GPU merges, and avoid round-tripping every merged matrix.
- Small number types as kernel values: still on CPU for a `for` over `List(Complex)` (a list member with `value.Bits is 32` fails the build, index it instead), a `List(Complex)` result, a method returning `Complex` that has a loop or an early return, and list-backed types (`Vector2`, `Vector3`).
