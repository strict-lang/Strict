# Build speed analysis

Where `strict build` spends its time today, what is already cached, and which changes would make a small incremental build fast. Measured 2026-11-24 on this PC with `STRICT_BUILD_PROFILE=1`, scratch copies outside the repo, `--no-recursive` (no `.nativecache` read or write) so the pipeline really ran. The machine was building for another session at the same time, so treat single-digit percentages as noise and the shape of the numbers as the result.

## The short version

| Case | Wall time | What dominates |
| --- | --- | --- |
| Warm VM (`run -c`, `test <file> unit`, `check <file>`) | 60–220 ms | nothing, already the fast path |
| `build` with a current `.nativecache` | 0.1–0.6 s | stamp check |
| Small program, cold (Sum, 1 file) | 2.7 s | lowering 1.29 s + link 0.75 s + clang 0.34 s |
| Small program, one-line body change | 1.45–1.58 s | link 0.90 s + lowering 0.24 s |
| Medium folder cold (Math, 17 files, 275 functions) | 31–37 s | emission 22–27 s (loop-heavy bodies 18–22 s) |
| Medium folder, all cached | 4.6–7.1 s | emission cache re-read + reachability/safety/GPU selection |
| Big program cold (Runtime tests, 2969 functions) | 66 s | frontend 14.8 + emission 11.4 + lowering 13.1 + link 15.6 |

A dev link without ThinLTO (option 7) cuts the small one-line change to about 0.6 s and the big link to about 1 s, at the cost of a slower executable; it is the one place where trading runtime speed for build speed pays.

Two different problems hide behind "builds are slow":

1. **Small edits still cost a full link.** Link is 0.90 s of the 1.45 s one-line-change build, and 15.6 s of the big one. This is not compiler work at all, it is ThinLTO + `llvm-rc`.
2. **Emission is linear in loop bodies, not functions.** 31 functions of `Vector2` emit in 0.17 s; 30 functions of `Mandelbrot` take 10.2 s; 50 functions of `Matrix` take 19.0 s. Roughly 0.36 s per loop-heavy function. A folder of numeric code is minutes of emission before a single part is lowered.

The per-method object design already exists and already works (see below); the wins left are (a) stop invalidating it for the wrong reasons, (b) stop re-parsing and re-analysing what did not change, (c) make the loop emission cheap, (d) share prebuilt runtime parts.

## Stages and what is cached

`NativeBuild.Compile` → `Rebuilt` → `IsUnchanged` → `Link`. Per stage, today:

| Stage | Cache | Cost when nothing changed | Cost when one body changes |
| --- | --- | --- | --- |
| Frontend (discovery, parse, chunks, inlining, struct lanes) | `obj/Name.strictbinary`, type inference `InferenceCache` in the server | skipped | skipped for a `build` (stamp hit); redone for a folder test |
| Emission (MLIR text per function) | `obj/Name.emitted`, keyed per function by source text + compiler stamp + every type layout it touches | re-read and re-parsed every request (96 MB `Strict.emitted`, 429 MB `Tests.test.emitted`) | one function re-emitted (8 ms), rest reused |
| Reachability / safety / GPU kernel selection / globals | none, redone every request | 1.3 + 0.6 + 0.5 + 0.2 s on Math; 1.3 + 2.1 s on the big program | redone |
| Lowering (`mlir-opt`, `mlir-translate` per part) | `obj/*.parts/*.o` reused when the part's `.owned` text is byte-identical | 0 s, whole stage skipped | one part re-lowered (0.24 s) |
| Host runtime assembly, `llvm-rc` | `.res` rebuilt each link | 0.05–0.06 s | 0.05–0.06 s |
| `clang` per part | part `.o` reuse as above | 0 s | 1 job, 0.045 s |
| Link (ThinLTO over all parts, LTO cache 64 MB) | ThinLTO cache | 0.16–0.37 s warm | 0.90 s Sum, 0.37 s big-warm |
| Whole pipeline | `.nativecache` inputs snapshot | 0.1–0.2 s | n/a |

Emission cache behaviour is correct: same source re-runs reuse (1/1 reused, 0.10–0.13 s, no link), a body change re-emits fresh and relinks (~1.16 s). Earlier "fresh build" anomalies in this session were another agent rebuilding `strict.exe`; `CodegenStamp.Current` returns `executable|mtime`, so every key in `obj/` became stale at once. A `strict.exe` rebuild therefore costs every package a full re-emit.

## Concrete options

Ordered by gain per effort. Each is independent.

### 1. Stop keying emission on the compiler executable's mtime (small, high gain)

`Compiler/CodegenStamp.strict` puts `executable|mtime` into every emission key, so rebuilding `strict` invalidates 4832 cached functions for `Strict.strict` and everything else. What actually changes codegen is the MLIR text and the runtime header (`Compiler/NativeRuntime*.mlir`), both of which can be hashed directly.

- Change: key on a content hash of the codegen sources and the runtime MLIR, not on the executable stamp.
- Gain: after a compiler rebuild, `build Strict.strict` goes from a full re-emit (minutes) to the cached path (the observed 32 s rebuild window suggests ~30 s instead of several minutes); every other package keeps its cache too. This is the difference between "rebuild the compiler and wait" and "rebuild and keep working".
- Effort: small, one file plus a hash source. Validate with `STRICT_EMISSION_CHECK=1` (re-emit and compare text) on one package before trusting it.
- Risk: a codegen change that does not touch the hashed inputs silently keeps stale MLIR. Mitigate by hashing the whole `Compiler/` + `Runtime/*.mlir` set that feeds emission, and by keeping the check mode runnable in CI.

### 2. Keep the parsed emission cache and the analysis results in the server (medium, high gain)

Today each request re-reads and re-parses `obj/Name.emitted` (429 MB for the big test binary), then redoes reachability, safety and GPU kernel selection even when every body was reused. That is the whole 4.6–7.1 s of an all-cached Math run, and 2.1 s of the big one, spent to learn that nothing changed.

- Change: keep the parsed function table and the reachability/safety/GPU selection keyed by the emission file's mtime+size in the resident server (`Runtime/Resident.strict` already holds type inference this way).
- Gain: all-cached folder build from 4.6–7.1 s to under 1 s; the big program's 3.4 s of analysis to near zero. Repeated `check` and folder `test` runs benefit as well.
- Effort: medium. The parse is already centralised, so this is a cache in front of it plus invalidation on file stamp.
- Risk: memory (a 429 MB file's parsed form is not small) and staleness. Bound it by keeping only the most recent package's table, and drop it on stamp change.

### 3. Compare keys instead of the MLIR text in `IsUnchanged` (small, medium gain)

`NativeBuild.IsUnchanged` decides whether to skip lowering/link by comparing the freshly written `.mlir` against the previous one. For the big program that is 27–38 MB of text per build, and it runs even when every part was reused from cache.

- Change: skip the text compare when every part already matched its `.owned` key, and compare part keys rather than concatenated MLIR otherwise.
- Gain: removes a 27–38 MB read/compare from the big incremental path; small on Sum, noticeable on `Strict.strict` and the test binaries.
- Effort: small, localised to the skip decision.

### 4. Shared prebuilt part cache for runtime and base-type parts (medium, high gain on cold builds)

Every program links the same runtime and base-type parts. A fresh build lowers and compiles them again. `Tools/PtxCache.strict` already does exactly this pattern for GPU kernels, so the shape exists in the codebase.

- Change: a shared on-disk part cache keyed by part text hash, outside `obj/`, for parts that come from the runtime and base types.
- Gain: most of the 1.29 s lowering + 0.34 s clang of a cold Sum build, and a large share of the 13.1 s lowering + 2.5 s clang of the big one, for every program after the first.
- Effort: medium. Needs a bounded, versioned store and eviction, and the same key discipline as the emission cache.
- Risk: a stale shared cache is a wrong binary. Version the store by the same codegen hash as option 1 and delete on mismatch.

### 5. Fix the loop emission blowup (medium, highest ceiling)

Each raw loop is emitted as several variants, and each variant regenerates the whole loop text: `MlirLoops.Begin` alone calls `MlirGpu.Launched`, `MlirKernel.Versioned`, `MlirParallel.Forked` and `MlirWholeLoop.Versioned`, and per copy `MlirDefaults.ParameterName` rebuilds the parameter list and does a linear `ParameterTexts.Index`. That is 0.36 s per loop-heavy function against 0.005 s for straight-line code, 22–27 s of emission for 275 Math functions and 8.3 s of bodies for the big program.

The variants are not optimizations with an off switch, they are different programs that all have to exist:

| Variant | Droppable? |
| --- | --- |
| CPU loop body | no, that is the program |
| GPU kernel version | no, `Gpu.Require` loops have no CPU path and error out |
| parallel fork / chunked version | no, a `parallel` loop has no serial meaning |
| fast-loop and lane specializations | only with a guard plus fallback, i.e. new code |

So the fix is to generate the text once and compose the variants from those strings, not to drop variants and not to add a second, faster-to-build emission mode. A debug emitter would double the check, mutation and test surface (against "exactly one way to solve each problem") and would make the dev binary behave differently from the released one.

- Change: name parameters once per function instead of per variant, and emit the loop body once as a template the variants reuse.
- Gain: Math emission plausibly from ~25 s to single-digit seconds; the same ratio on every numeric package. This is the only option that makes a *cold* build of numeric code fast.
- Effort: medium, but it touches the hottest and most intricate emitter, so it needs the emission check mode on a real package (Math) before and after.
- Risk: emitted text drift. Compare `STRICT_EMISSION_CHECK=1` output and Math's tests before accepting.

### 6. Small wins

- Reuse the `.res` resource object when the manifest did not change (saves `llvm-rc` 0.05–0.06 s per link, every link).
- Gate VM tests per chunk rather than per file in folder tests.
- Cache hygiene: `Networking/native/Socket.<random>.o` ×25, `NeuralNetworks/native/**/*.old*.o` and `quants.*.o` accumulate; they cost disk and confuse "what is current".
- Warm VM is already at the target (60–220 ms). Keep `run -c`, `test <file> unit`, `check <file>` and the VS Code path on the server; do not move work into the build path.

### 7. A dev build that skips ThinLTO (small, high gain on small edits)

The 0.90 s one-line-change link is ThinLTO, not a linker limit. Parts are compiled to bitcode (`clang -c -flto=thin -x ir`) and all optimization happens at link (`--lto-O3` on Linux, `-Wl,/opt:lldltojobs=all` on Windows), so one changed function invalidates its ThinLTO work and the parts that import it are re-optimized; a build where nothing changed links in 0.16 s. A dev link that compiles parts to objects and skips ThinLTO is about 0.045 s per part plus ~0.05 s of lld: Sum's one-line edit goes from 1.45 s to ~0.6 s, the big program's 15.6 s link to ~1 s.

- Change: one flag in `Toolchain` / `LinuxBuild` (`strict build --fast`, `STRICT_FAST_BUILD=1`), not a new emitter and not a second pipeline.
- Gain: the largest single number on a small edit, and it stacks with options 1–3 (which remove the rest of the 0.6 s).
- Effort: small, one flag through the existing toolchain objects. Semantics unchanged: `-O0`/no-LTO preserves behavior, only speed and size differ.
- Do not make it the default for folder builds: every dev run of a DeltaEngine game or a NeuralNetworks model would be 2–10x slower and people would turn it off. Default stays optimized, release builds keep ThinLTO, and benchmarks keep using `Release/<Platform>`.
- Risk: someone benchmarks a `--fast` build. Keep the flag out of `Release/` and say so in the build line it prints.

### Deferred

- **A slower, smaller debug executable from a reduced emitter.** Tempting because emission dominates a cold numeric build, but the variants are not optional (table under option 5) and the cost is regenerating text per variant, so option 5 gets the same build speed with the fast executable and one code path. Only the toolchain side of the idea survives, and that is option 7.
- **Per-method object files with per-method relocatable linking.** The parts already give incremental linking: warm ThinLTO link is 0.37 s for an 11.9 MB exe, fixed overhead 0.15–0.2 s. Going finer would trade that for a custom linker or many small objects, with a large effort and a small gain over options 1–5.
- **Skipping the link when only bodies change.** The warm LTO cache already makes this 0.16–0.37 s; not worth a second code path.

## Suggested order

1. Option 1 (compiler stamp) — smallest change, removes the worst cliff (every rebuild invalidates everything).
2. Option 7 (dev link without ThinLTO) — smallest change on the biggest number of a small edit.
3. Option 3 (key compare) — small, makes the big incremental path clean.
4. Option 2 (server-side parse + analysis cache) — turns the 4.6–7.1 s all-cached folder build into sub-second.
5. Option 5 (loop emission) — the ceiling for cold builds; do it with Math as the reference package.
6. Option 4 (shared prebuilt parts) — cold-build win that stacks on 5.
7. Option 6 (small wins) whenever a file is already open.

Options 1–3 and 7 are all "do not redo unchanged work" and together should bring a one-line change in a small program to well under 0.5 s and an all-cached folder build under a second. Options 4–5 are what make a *cold* build of numeric code stop being minutes.

## Risks to watch

- Every cache here is keyed on something that can silently go stale. Each option needs its own invalidation story and, where possible, the `STRICT_EMISSION_CHECK=1` comparison as the safety net.
- Speed rule applies to the fix as much as to the build: a cached path that takes 10 s where it took 1 s is a regression even if it skips work.
- Do not measure on a machine that is building for another session; the numbers above were taken under contention and are a lower bound on the real gain.
