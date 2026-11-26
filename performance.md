# Performance
Measured speed work, compiler front end first. One item per finding, removed when done or no longer true.

## Front end, checks, tests and the server
- Cache loading: `EntryDecoder.Of` turns all bytes of a cached type into varints and names (`ByteEncoder.Integers`) also when only its method names are asked (`BinaryLinks.IsTested`, 100-170 ms of the 230 ms stale scan of Strict's 426 test files on one thread, reading and the head 65-130 ms more). Decode varints and names natively. An up to date `strict test` of Qwen3Benchmark takes 220 ms, scripts never take the cache.
- Compiler self-build front end: decoding the homes of new types (`BinaryFile.Head`, `ZipReader.Copied` turns every byte into a Number, `BinaryLinks.Owned`), `List.Unique` and appends of the 4500 targets per pass, and GPU preparation (two or three host text copies in `Embedded`/`Referenced`; `Attending.Used` now fast-exits when `@strict_gpu_` is absent, `86 -> 39 ms` on `Bootstrap.mlir`).
- `strict check .` of DeltaEngine warm (copy, 2026-11-26): 5.5 s, validation 1.5-1.7 s (was 5-6 s, InputOutputs.Settle), dependency cycles and project rules stamped, tests 3.3-17 s from the random light mutant round of each folder. Warm `strict check .` of Strict is 3.4 s (was 6.5 s): validation 2.5 s is the per-folder keys, a fresh process tokenizes the mentions of every file once (~1 s, the server's Memo keeps them but its check is not faster), traversal ~0.7 s and 15000 `File.Modified` stamps ~0.4 s.
- `strict test RealTimeStrategy native` (DeltaEngine, 73 files) after one touched file (16.6-17.3 s): struct lanes 3.2-3.9 s (lane state is texts like "3 Pair" keyed by register numbers: keep laned registers as numbers), `CallGraph.Closure` in `EmissionCache.Keys` (`Marked` copies the dictionary parameter, rebuilding the index), dependency linking (`Completed`, `Forwarded`, `Linkable` and round lists), and name validation reading every linked source again per file.

## Native builds (emission, lowering, linking)
- Web rebuilds after a one-line edit (LogoApp, 2026-11-26, machine 50 % busy): `web edit` 12-14 s (emcc link with Asyncify 5-8 s, wasm-opt -O1 4 s), release `web` 23-24 s (wasm-opt -O3 12 s, link with Closure 9.5 s).
- Build speed (under 10 s hard limit, 2-3 s target; LogoApp after a one-line edit 1.8-2.4 s on 2026-11-26, was 3.6-4 s): link 0.25-0.6 s, the test gate front end 0.31 s (the front end runs twice), lowering 0.3 s, emission 0.28 s.

## VM
- A VM call of a small Strict method costs about 6 us: `Machine.Bound` a quarter of it (one `Member` call per member slot, Number has 12 with its constants, and a `strict_list_push` per slot; reference counting under `Frame.Call`, `Bound` and `Rounding.Members` is 17 % of a Vector2 loop), the body about 150 ns per instruction (`Frame.Finish`, `strict_list_set`, `strict_rc_container`).
- VM calls dominate game logic (a member read through a method like `Vector2.X` is two calls): DeltaEngine's multi-tick simulation and content loading tests are Slow for it. `position + Vector2(1, 2) * 0.75` costs about 20 us (nsys 2026-11-26): `Rounding.Build` 21 % (prototype list copy 4 %, X and Y through `Number.Rounded`'s logarithms 3 %), `Machine.Bound` 19 % (`strict_list_push` 8 %: the serial path of a collecting loop pushes into its preallocated list). Hand-patched A/B of `Machine.Bound` in obj/Strict.mlir, 2 x 8-10 alternating runs: writing `items[k]` + `strict_rc_written` and `strict_list_filled` at the serial exit was 5-12 % faster by median (noisy), calling the outlined loop function for every count >= 2 was 20 % slower. Needs the write in `MlirLists.Append` and a filled block before the exit in `MlirLoops`/`MlirRangeLoop` (f64, i32, i64 counters) for parallel-eligible, non-filtering object lists; `MlirParallel` is at the method limit.

## Compiled CPU loops, lists and texts
- Serial loops still generic (`strict_hoisted_read`/`strict_list_read`, `strict_is_failure`, `strict_list_push`): gathers `table(value)`, calls that can fail or read lists (`MersenneTwister.Normals`), one thread of a parallel chunk, and raw `Tempered` calls (f64 body has two remainders, no whole-number version used).
- Texts: a substring of a plain text under 256 bytes is copied (Substring, a trimmed or cut text that is no span), Upper and Lower always copy.
- Matrix and vector speed: the VM calculates in f64, accessors (`point.X`) are not counted as 32 bit values, `Axis(n).Y` still allocates its Vector3, and a block is only reused when the local is alone.
- Whole-number call guards give up (f64 version runs) when a variable read after an `if` or loop that stores it is used in `+`, `-` or `*`: a maximum over the possible stores would keep the i64 path.
- ParticleStress at 1 million particles (4.8-5.5 ms per tick): 32 bit members (72 -> 44 bytes per particle) and writing in place over the list being read.
- Generated CPU loops still per element: reads of non-hoisted lists (`strict_list_read`), checks on instances, chunks with inner loops, serial loops, and f64 numbers and indices.
- Reading a packed list through `strict_list_items` (chunks without inner loops over Float8, byte, lane or quantized lists, quantized formats read by index, Float8 and quantized arguments of native library calls) replaces it with an f64 copy for good (`strict_list_unpacked`).

## Collector
- Realtime frames: mark incrementally over several frame ends when a full collection takes longer than a few ms.
- `GarbageCollectionExceedsLimit` and `GarbageCollectionAboveTarget` could name the largest allocating call sites of the window.

## GPU
