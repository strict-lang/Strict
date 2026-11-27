# Build time (2026-11-27, `STRICT_BUILD_PROFILE=1`, `STRICT_GPU=0`)

One table per build kind. A step is listed when it is large enough to skip, cache or make incremental. Nested lines sit inside the phase above them.

## Self-build of `Compiler/Strict.strict`

Profiled run, emission cold, link warm: 89.5 s. An earlier run the same morning with a cold link and a compiler file newer than `strict.exe` was 118 s.

| Step | Time | Skip when |
| --- | --- | --- |
| Function bodies, 5030 functions, nothing cached | 48.4 s | The emission cache hits. A codegen-stamp miss emits all of them. Bodies are about 8× the 2026-11-26 seed (~5 s) inside the loop emitters, not in the cache. |
| MLIR emission, including the bodies | 58.0 s | Same cache. Reachability 3.9 s, safety 1.7 s and GPU kernels 2.1 s are inside this and are skipped by a served compile (`STRICT_SERVER=0` on this run). |
| Verified, the test gate | 14.2 s | Dependency linking 6.8 s and saving homes 4.5 s rerun every self-build. Homes can wait until a linked source changes. |
| Frontend inlining | 2.9 s | |
| Host lowering (`mlir-opt` and `mlir-translate` were under 1 s here) | 0.9 s | |
| Linking, warm ThinLTO cache | 0.6 s | The cold link the same morning was 21.3 s. The cache already skips it. |
| New executable checks | 0.05 s | 23.3 s when a compiler file is newer than `strict.exe` (that file's check, plus `Runtime/Hosted.strict` and `NeuralNetworks/GpuMemory.strict`). Already skipped when nothing is newer. |
| Not in a phase line | ~12 s | The 126 MB `.mlir` and the emitted table. |

## `Examples/Sum.strict`

| Build | Total | Where it went | Skip when |
| --- | --- | --- | --- |
| Fresh program | 2.25 s | Lowering 0.93 s (`mlir-opt` 0.26, `mlir-translate` 0.18), host clang 12 jobs 0.63 s, link 0.15 s | The 12 clang jobs are the runtime parts. A repeat compiles the one changed part. |
| One-line edit (`Sum + 0`) | 0.54 s | Lowering 0.14 s, clang 1 job 0.03 s, link 0.14 s, 1 of 2 functions re-emitted | Unchanged parts already stay. The link is the floor. |
| Web | 12.1 s | `emcc` 8.1 s, `wasm-opt` 2.2 s, clang 0.9 s | `wasm-opt` is the optional dev-build cut. |
| Fresh again, `Examples/obj` deleted, `Tools/PartCache` kept | 0.72 s | Lowering 0.12 s (`mlir-opt` 0.07, `mlir-translate` 0.05), host clang 3 jobs 0.05 s, link 0.14 s | The other objects are the runtime and base-type parts. Ten of them were copied from the shared cache. The cold run just before this was 1.65 s (clang 12 jobs, 0.59 s). |

## Native test, `strict test Math native`

Served compiles, same flags. The published 2026-11-26 one-line edit was 7.0 s.

| Build | Total | Reachability | Safety | Bodies | GPU kernels |
| --- | --- | --- | --- | --- | --- |
| First served compile, unchanged Math | 5.73 s | 1.96 s | 0.78 s | 0.24 s | 0.26 s |
| Second served compile, same source | 2.21 s | 0.017 s | 0.025 s | 0.09 s | 0.12 s |
| One-line edit that adds a call (`4 + 1`) | 5.17 s | 1.51 s | 0.039 s | 0.28 s | 0.14 s |
| One-line edit with the same calls (`6, 8` / `10`) | 4.15 s | 0.032 s | 0.043 s | 0.36 s | 0.15 s |

The resident server keeps the parsed table, the closure, the safety flags and the GPU table. An edit that changes calls still walks reachability (1.5 s); that walk is the result, not a stale edge list.
