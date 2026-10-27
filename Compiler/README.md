# Strict.Compiler
Builds native executables: `NativeBuild` behind `strict build` turns the instructions of all used types into MLIR (`MlirModule`, one `MlirFunction` per method with `MlirStatements`, `MlirCalls`, `MlirLoops` and friends, `MlirKernel` and `MlirFastLoop` add a copy without per element checks to simple loops over lists, LLVM vectorizes it), adds the runtime written directly in MLIR (`NativeRuntime.mlir` for texts, lists, files, processes and the garbage collector, `NativeRuntimeWindows.mlir` or `NativeRuntimeLinux.mlir` for the platform) and runs mlir-opt, mlir-translate and clang (`Toolchain`, `ToolRunner`). mlir-opt first drops every private function nothing calls (`--symbol-dce`, the runtime helpers a program does not use), for the host and the GPU module alike: 420 test and example executables 26% smaller (FizzBuzz 387 to 224 KB, clang 2.1 to 1.0 s), the PTX of a small GPU program 132 to 6 KB (NearestNeighbors), `strict.exe` 3.67 to 3.55 MB, same speed. The GPU module of a successful build is kept as `Name.kernels.mlir` next to `Name.ptx` (`PtxCache`): when emitted GPU input, code-generation flags and LLVM tools match, the PTX is reused instead of running mlir-opt, mlir-translate and clang for sm_75 and sm_80 again (MossBlock.test 13.4 to 6.8 s). Builds first require successful VM unit tests, reusing unchanged passing results; Slow tests remain explicit. Programs do not generate an extra test executable. A type without `Run` builds its test executable without running it. `strict test <file> native` explicitly builds and runs native tests; `strict test <file> lines` also instruments per-line timings. Program and library emission omit test bytecode; unit checks run separately before emission. Unit caches contain the selected test graph, and program loading compiles its own Run graph when needed. Release packaging retains the build executable and publishes a checked temporary copy, so repeated game builds can reuse it.

A type of two 32 bit numbers (Vector2 `MlirPairs`, Size `MlirCouples`) is one 64-bit slot. A register holding a Vector2 that is read as a list (a `Numbers` parameter or return, a List method other than X, Y and Length, a loop, list arithmetic) becomes a list right after it is written, a list given or returned as a Vector2 becomes a pair (`MlirPacking`, the first such read decides), an index into a Vector2 reads the pair (`strict_two_at`). One-number types with `value.Bits is 32` (Degrees) are rounded to 32 bit when constructed, like on the Strict VM.

Values are rewritten in place where nobody can see the old one: the dying value of a local, a list mapped to the same type or a returned struct becomes the destination of the call or loop replacing it (`MlirOwned`, `MlirDestined`, `MlirListDestined`, `MlirStructs`). A list of a flat type is one allocation with its elements inline (`strict_flat_list_new`).

`NativeCompiler.strict` is the native compiler `strict` hands its builds to (`Process.Started`, the output appears right away), so the front end `strict` stays small (2.4 MB instead of 5.2 MB) and builds of other programs do not rebuild it. `Bootstrap.mlir` is the seed, `NativeCompiler` compiled by itself: `Bootstrap.cmd` (Windows) and `bootstrap.sh` (Linux and macOS) build `NativeCompiler` from it, which then builds `strict`.

Completed steps taking at least 3 seconds print their elapsed time in seconds. Set `STRICT_BUILD_PROFILE=1` to print every measured stage, including sub-millisecond steps, cache hits/misses, linking passes, frontend/emission substeps, GPU target batches, external tools, packaging and build-stamp writes. Other values retain the three-second threshold. This flag changes diagnostics only: it keeps cache hits, optimization options and generated programs. Current compilers read it through a native getenv helper; a cached source fallback keeps the flag usable when building with an older compiler. `python Compiler/check_profile.py` checks VM/native flag behavior and that toggling it preserves unchanged-build skipping.

```powershell
$env:STRICT_BUILD_PROFILE = "1"
$env:STRICT_THREADS = "8"
.\strict.exe build Strict.strict
Remove-Item Env:STRICT_BUILD_PROFILE
```

Timings are elapsed wall seconds, rounded to four decimal places in detailed mode. Parent stages include their children: `Frontend` includes dependency linking and validation; `MLIR emission` includes reachability, safety, recursion, function bodies, helpers and globals; `GPU preparation` includes GPU tools; `Host lowering and translation` includes MLIR splitting/preparation and its tool batches. Batch timings describe all concurrent jobs together. Do not add parent and child times. `Total compilation` excludes dependency inventory/snapshot and stamp writing; `Build total` includes these within NativeCompiler, while a CLI cache hit returns after its own timed check. A cached build intentionally has no frontend/backend rows. `check_build.py` retains the normal fast-step suppression and slow-step reporting checks.

The previous bootstrap seed still called `NativeGate`, so rebuilding from it ran tests and used the older compiler pipeline despite current source removing that gate. The seed now comes from byte-identical stages 2 and 3 (SHA-256 `605746933f973a03af535a3a0f051185dcb6115a0754fe95f2bc16e8ce99cef6`). A separately compiled seed built and ran a program containing a failing inline test without creating or running a test executable. `check_build.py` rejects seeds containing the old gate.

Native model:
- Every value is one 64-bit slot: numbers are doubles, texts, lists and instances pointer bits, Errors carry a NaN tag so `x is Error` works for any type.
- Lists of numbers that C code hands out stay packed (32 bit floats, 16 bit halves, 8, 6 or 4 bit quantized, `Float8` for E4M3) and loops read them directly in 32 bit floats.
- A type with one member is that member, small number types (`Vector3`) are one block of floats, lists append and write in place while they are the newest version.
- A collecting `for` whose body only calculates runs on a pool of CPU threads, LLVM vectorizes the inner loops (SIMD).
- GPU: such loops become kernels (PTX built by clang, loaded through `nvcuda.dll`/`libcuda.so.1`, no CUDA toolkit to run; AMD through HIP from `.hsaco`; Metal on macOS, experimental). Matrix products of tokens (`Batch.Sums`, `Batch.Grouped`, `Batch.Gated` with the SiLU of its gate rows times its up rows in the epilogue, `MlirGated`) use the int8 tensor core kernels of `NativeRuntimeTokens.mlir`, `Attention.Merged` the 16 bit FlashAttention-2 style kernel of `NativeRuntimeAttention.mlir` and `Products` the 16 bit tensor core products of `NativeRuntimeProducts.mlir`, launches are captured into CUDA graphs and results stay on the GPU until the CPU reads them. `STRICT_GPU=0` keeps everything on the CPU, `STRICT_GPU=all` sends every possible loop to the GPU, `STRICT_GRAPHS`, `STRICT_DP4A`, `STRICT_GPU_WARM` and `STRICT_GPU_SHARE` switch single features.
- Memory: a conservative mark-sweep collector with a young generation plus reference counts that free large objects early, so a loop doing the same steps reuses the same memory; `STRICT_YOUNG`, `STRICT_RC=0` and `STRICT_RC_CHECK=1` tune and check it.

Native packages: a type whose methods have no body (`Examples/NativeAdd/Adder.strict`) is implemented in C in the `native/` folder next to it, `Adder.Add(1, 2)` calls `double Adder_Add(double self, double left, double right)`. A type with one member holds a C handle, its bodyless `from` is the constructor. Every argument and result is one double, `strict.h` reads and returns texts, number lists and packed rows, `strict_fail("Name")` returns an Error. `strict build` compiles the C files once and links them with the flags of `native/Windows.flags`, `Linux.flags` or `MacOS.flags`, the Strict VM loads them as a library.

Libraries: `strict build X.strict library` exports every public method with Number, Text, Boolean, Numbers or number record values as C functions (0 success, 1 for an Error, `strict_last_error()`), plus a generated C# wrapper.

iOS: `strict build X.strict ios` stages an arm64/iOS 15 static archive exporting `strict_main` and content in `Release/iOS`. Pure Strict archives can be built on Windows; native packages require full Xcode and `native/IOS.flags` on macOS. App linking, UIKit integration, signing and execution require the MacMini. See [iOS compiler and MacMini handoff](IOS.md) for verification, legacy references and remaining work.

Android: `strict build X.strict android` runs the tests on the host VM, then builds an arm64-v8a shared library for API 26 (`ANDROID_NDK_ROOT`, or an SDK with `ndk/<version>` in `ANDROID_HOME`, `ANDROID_SDK_ROOT` or `%LOCALAPPDATA%/Android/Sdk`): host LLVM compiles the IR to an AArch64 object, NDK clang compiles native packages with their `Android.flags` and links with 16 KB ELF alignment, sharing the POSIX runtime with Linux. It stages `Release/Android/jniLibs/arm64-v8a/libmain.so` and `Release/Android/assets/Content` as Gradle source sets; the APK/AAB, signing, Activity/JNI glue, window, input, audio and textures belong to DeltaEngine. Compute loops run on the CPU there, the tests of a type without `Run` need an Android runner.

GPU memory pressure: when a collection and the free pool blocks are not enough, cached copies of host lists are evicted by age once the queued work is done; argument buffers stay protected while a call dispatches nested work, and lists that only exist on the GPU are never evicted (`python Compiler/check_gpu_cache.py` checks these rules). Weights that do not fit are spilled and streamed back ahead of their use.
Output: `X.android.so`, staged as `Release/Android/jniLibs/arm64-v8a/libmain.so`, with existing packaged content mirrored to `Release/Android/assets/Content`. These folders can be included as Gradle source sets. This command does not create an APK/AAB, sign it, supply Activity/JNI lifecycle glue, or convert textures. Android window/input/audio integration and ASTC/ETC2 encoding and loading belong in DeltaEngine. Generated compute loops use their host CPU fallback; Android does not build or load CUDA/HIP/Metal kernels. A type without `Run` builds a test library; its native tests require an Android runner and are not executed on the host.

GPU memory pressure: cached host-backed dedicated copies are evicted by age after queued work completes when collection and free pool blocks are insufficient. Prepared argument buffers stay protected across nested dispatch until the owning call returns. GPU-only resident activations are not cold-copy candidates. `python Compiler/check_gpu_cache.py` checks row-format compatibility, nested input protection, restored scopes, and pooled-copy exclusion. On the RTX 3080 this lets the ordinary split/ThinLTO MOSS Spectrum runner finish its previously failing VAE transition with the exact baseline PNG; decoder transition overhead remains in StrictComfy/Handoff.md.

NativeCache checks dependency stamps in `strict` before launching NativeCompiler, and in NativeCompiler for direct calls. Successful builds and native tests record sources, native C/header/flag inputs, source-folder membership, content, compiler/tool binaries and required outputs. Input snapshots must still match after compilation/tests before results are saved. Unchanged builds perform metadata checks only; unchanged native tests execute nothing. Cross-platform, library and instrumented builds currently bypass this stamp.

Cache hits show the source modification time and executable build time in local time, to the minute: `Up to date: MossImage.strict (last changed 2026-10-26 02:36) < MossImage.exe (last build 2026-10-26 02:41)`. Native test cache hits retain the `Tests up to date:` prefix and identify the `.test` executable. `check_build.py` verifies timestamps against file metadata through the CLI, NativeCompiler and VM while checking that no MLIR or executable is regenerated.

PTX cache keys include emitted input, target flags, exported symbols and LLVM executables. Rebuilding the compiler alone no longer discards identical GPU code. Shared runtime PTX lives under `Tools/PtxCache`, published from a private staging directory so parallel builds cannot read partial results. Program sidecars remain self-contained next to the executable. Whole-file `Text.Combine(File(path).ReadLines, "\n")` reads can lower to one normalized text read (`MlirReadText`); LF, CRLF, trailing CR, BOM, Unicode and empty-file behavior are checked by `python Compiler/check_build.py`.

Measured on Windows, 2026-10-26. Timings include process startup. Cache misses below used an already prepared GPU runtime cache; they are not the cost of preparing a new LLVM/runtime combination.

| Source | Native cache miss | Entire build skipped |
| --- | ---: | ---: |
| Boolean | 0.8-1.2 s | 62 ms |
| Examples/Sum | 0.76 s | 55-57 ms |
| NeuralNetworks/Xor | 3.08 s | 76-78 ms |
| StrictComfy/MossGenerate | 6.28 s | 70-83 ms |
| StrictComfy/MossImage | 5.50 s | 72-80 ms |
| DeltaEngine/Games/Breakout | 17.72 s including native C rebuilds | 114-120 ms |

A separate MossGenerate copy took 11.0 s for its first program-specific GPU build, 6.03 s after changing a host log message, then 74 ms with no changes. Initial shared attention-runtime preparation still made the first build take 22.2 s before whole-file read fusion; subsequent programs reuse that runtime. After the concurrent upstream update changed GPU input, the first MossGenerate build took 21.17 s and MossImage 9.67 s; their next calls skipped in 70 ms. Those cold-cache and 2-3 s source-rebuild targets remain open.

MossImage phase profile: dependency snapshot 0.56 s, frontend 1.68 s, MLIR emission 1.23 s, GPU preparation 0.40 s, host LLVM/linking 1.88 s. Boolean's earlier 0.8 s was native test-code generation plus external LLVM lowering, translation, 16 clang workers and ThinLTO linking, without executing its tests. Xor also generates GPU kernels; compilation does not train the network. Source rebuilds still exceed the requested 10/100 ms small-file targets and 2-3 s complex-file target. Next work belongs in incremental frontend/IR reuse and emission/backend profiling, not rerunning tests or rebuilding identical shared kernels.

Focused VM/native tests and the standalone regression check cover skipping builds/tests, changed source/dependencies/options/content, missing executables and whole-file read semantics. One earlier profiling process exited with Windows heap-corruption status 0xC0000374; repeated builds after a fresh self-build and upstream memory fixes passed, so its cause is unconfirmed.

Latest visible phase measurements on this Windows workstation, using the exact Moss build commands from StrictComfy:

| Phase | NativeCompiler self-build | MossGenerate | MossImage |
| --- | ---: | ---: | ---: |
| Dependency snapshot | 0.33 s | 0.41 s | 0.41 s |
| Frontend | 15.54 s | 1.53 s | 1.43 s |
| MLIR emission | 5.29 s | 1.39 s | 1.26 s |
| GPU preparation | 0.20 s | 0.45 s, cached | 4.85 s, program kernels rebuilt |
| Host MLIR lowering | 4.31 s | 0.48 s | 0.45 s |
| Host LLVM translation | 2.87 s | 0.34 s | 0.31 s |
| LLVM splitting | 1.74 s | 0.22 s | 0.20 s |
| Parallel clang workers | 0.37 s | 0.16 s | 0.17 s |
| Final clang/ThinLTO link | 3.23 s | 0.81 s | 0.79 s |
| Wall time | 34.25 s | 6.00 s | 10.08 s |
| Unchanged repeat | — | 76 ms | 71 ms |

No tests ran during these builds. MossImage GPU preparation included 1.01 s lowering, 0.65 s translation and 2.69 s compiling two GPU targets; shared runtime PTX was already cached. The 10 s hard limit and 2-3 s target remain unmet for self-builds and some GPU cache misses. Frontend and emission dominate compiler self-builds; host linking alone does not explain them.

Validation of the seed/timing updates: Compiler check reports zero issues; NativeCompiler and Toolchain VM/native tests, Strict VM tests, ToolRunner VM tests and `python Compiler/check_build.py` pass. ToolRunner native tests fail before tool execution with `unsupported: text of List(IsWindow)` in `$Tests`; an unmodified HEAD copy fails identically. Explicit compiler native tests also exposed expensive dependency snapshots (18-32 s), which are separate from the 0.3-0.4 s build snapshots above.

Examples and DeltaEngine games, 2026-10-26, using ordinary native builds after the compiler update. Existing native objects, GPU runtime caches and packaged content were retained. Every build succeeded without executing tests. Each unchanged measurement immediately repeated the same command.

| Source | Initial build, wall time | Unchanged repeat | Reported steps, at least 3 s |
| --- | ---: | ---: | --- |
| Examples/Sum | 0.83 s | 57 ms | None |
| Examples/FizzBuzz | 0.89 s | 61 ms | None |
| Examples/ParallelSum | 1.42 s | 55 ms | None |
| Examples/NativeAdd/NativeAdd | 1.08 s | 45 ms | None |
| DeltaEngine/Games/LogoApp | 10.66 s | 134 ms | Frontend 4.31 s |
| DeltaEngine/Games/Breakout | 5.89 s | 121 ms | None |
| DeltaEngine/Games/WebWars | 17.70 s | 130 ms | Frontend 5.22 s, MLIR emission 4.33 s |
| DeltaEngine/Games/Blocks | 5.54 s | 123 ms | None |

ParallelSum has no `Run`, so its build created `ParallelSum.test.exe` without running it. Game commands ran from the DeltaEngine root, for example `strict build Games/WebWars/WebWars.strict`. Individual steps below 3 seconds still contribute to total build time. LogoApp and WebWars exceed the 10-second ceiling; frontend and MLIR emission are the reported game bottlenecks. Per-file stdout and exact wall times/executable sizes are retained in ignored `Compiler/output/*-build.log` and `Compiler/output/build-profile-20261026.json`.

Host modules over 100,000 lines (the compiler, `strict`) are cut into 16 parts by `MlirSplit`: every part keeps its own functions, declares the others and the globals (defined in part 0), and runs mlir-opt (single-threaded, its parallel verifier crashed under 16 jobs) and mlir-translate in parallel, the parts then go to clang and ThinLTO like llvm-split parts did. strict.exe 40 to 24.5 s, the compiler 75-85 to 56 s; `fast` builds keep one module.

Collecting loops run on the CPU pool when `MlirSafety` finds nothing unsafe in everything they call (Random, Library, readers and writers, foreign C, GPU-able loops, writes to parameters or members; file reads, processes and logging are allowed) and the estimated work (`MlirCost`: instructions per iteration, called methods included, inner loops 15 times, file or process calls 20,000) times the count reaches 50,000. The compiler then has 102 parallel loops (linker rounds, emission, check per file). While the pool runs each thread allocates from its own lane of size class cursors (`strict_gc_lanes`, the collector lock only for new pages and large objects) and nothing is counted: storing a reference pins the object (flag 32, freed by the collector later), objects of a worker stay changeable in place. A lane keeps its spans into the next pool job (`strict_gc_carried` remembers where they stood, `strict_gc_mine` accepts what the lane allocated from there since) instead of starting every job on fresh pages; threads past the 511th share lane 0, which owns nothing. Under the lock a worker only takes a page and counts it; setting the page up, clearing it (not at all when it comes from growing the heap, committed memory is zero, on the web `strict_commit` clears it) and pushing it onto its class list (compare and swap) happen outside. The lock waits by reading before it swaps, the lock word and the pool's claim and done counters have cache lines of their own. Self-build of the compiler, before and after these four changes, 96 threads (the default on 128 logical processors): 175k lock acquisitions held for 15.6 G cycles in total with 330 M spins, then 137k held for 1.3-1.5 G; dependency linking 5.8 to 3.4 s, function bodies 2.6 to 0.72 s, full build 26.9 s (8 threads: function bodies 2.1 s, 1 thread 13.9 s). IR is byte-identical at 1, 8 and 96 threads.

A pool job wakes only as many sleeping workers as it has chunks besides the caller's (`strict_pool_wake(helpers)`, one wake each, all at once when the job can use every sleeper). Waking all 95 workers for a job of 4 items made them queue for the pool lock again before the next job could wake anyone: Bytecode/TestCodegen's VM tests run 45,556 jobs of 4 items on average, at 96 threads the caller spent 39 of its 41 G cycles in jobs waking, the test took 21-25 s with 334 s of CPU (8 s at 4 threads), now 8.4-8.6 s at 96 threads. `check_parallel.py` times a job of 4 items while 63 workers sleep: 197-207 us before, 12 us after (16 items 231 to 45 us, 48 items 172 to 83 us, 96 and 400 items unchanged at 170-230 us). Process.RunTools serializes CreateProcessA (no child may inherit the output files of another one) with an SRW lock instead of the spin lock, so the waiting threads sleep: the parent of `strict test Expressions` used 17 s of CPU in a 7 s run, now 2-3 s. Folder tests at the default 96 threads: Bytecode 26-27 s to 13-14 s, Expressions 6.4-7.1 s to 5.6-6.8 s; children limited to 4 threads would still be 10-25% faster. The compiler self-build is unchanged (10-12 s with unchanged IR, both before and after).

## Build modes and dependency audit

`strict build <file>` remains the optimized release build: `-O2` host code and the existing optimized native/GPU paths. `small` selects `-Os`. New `fast` mode uses `-O0` host code and skips LLVM splitting and ThinLTO; optimized native C objects and GPU kernels remain reusable. Fast builds retain Release staging and content, as requested. They currently support only the current native desktop target and reject combinations with other build options. Switching modes invalidates the build stamp; unchanged fast builds skip like release builds. Executable size and runtime speed are expected to be worse in fast mode.

Native folders are now selected only when a linked type contains a foreign method, using the existing `MlirForeign.IsForeign` predicate. A Strict type sharing a folder with native providers no longer pulls in all those C objects. Xor's release executable fell from 884,736 to 458,752 bytes; its build fell from 3.26 to 2.58 s, then 1.95 s in fast mode. A CPU execution check still converged below 0.0001 error: release training about 22 ms, fast training about 62 ms. This change does not alter numerical storage precision or GPU optimization flags.

Representative audit, 2026-10-26, Windows workstation. Ordinary builds were followed by unchanged calls; native objects, packaged assets and shared GPU caches were retained. Cold GPU work is identified separately. These are selected cases, not a full-package test sweep. No builds executed tests.

| Case | Release build | Fast build | Finding |
| --- | ---: | ---: | --- |
| Examples/Swarm | 1.39 s after dependency fix | 0.98 s | Native tests compiled, not executed |
| Examples/ErrorHandling | 1.97 s | — | Native tests compiled, not executed |
| Examples/LargeCollections / MemoryPressure | 0.86 / 0.84 s | — | Small native programs |
| NeuralNetworks/Xor | 2.58 s after dependency fix | 1.95 s | Unneeded native C removed |
| NeuralNetworks/Iris | 3.32 s before dependency fix | — | Small training program |
| NeuralNetworks/Gpt2Benchmark | 17.78 s, cold GPU artifacts | — | GPU preparation 14.53 s, clang 12.59 s |
| NeuralNetworks/Qwen3Benchmark | 6.67 s after GPU preparation | — | First attempt crashed after 21.02 s GPU preparation |
| ImageProcessing/Png | 3.48 s | — | Native tests compiled, not executed |
| Compression/Deflate | 2.45 s | — | Native tests compiled, not executed |
| Json/Json | 1.80 s | — | Native tests compiled, not executed |
| MachineLearning/KMeans | 3.00 s | — | Native tests compiled, not executed |
| Math/Matrix | 8.85 s | — | MLIR emission alone 7.00 s |
| StrictComfy/MossGenerate | 7.30 s, warm GPU cache | 6.21 s | Host backend savings are modest |
| StrictComfy/MossImage | 6.60 s, warm GPU cache | 5.62 s | Host backend savings are modest |
| StrictComfy/H3Video | 5.33 s, forced rebuild with warm GPU cache | 5.65 s | No demonstrated overall gain |
| DeltaEngine/Games/LogoApp | 6.25 s with native objects cached | 6.54 s | Initial 11.10 s included 4.33 s native C compilation |
| DeltaEngine/Games/WebWars | 23.95 s | 22.94 s | Frontend 8.48 s and emission 7.83 s dominate release |
| DeltaEngine/Games/Breakout | 6.74 s | 6.15 s | Modest backend savings |

Unchanged builds skipped in 53-163 ms. Times vary with machine load; small release/fast differences are not proven improvements. The host-only experiment on identical IR was clearer: unoptimized clang/linking took 0.20-0.31 s versus 0.80-1.75 s at `-O2`. Fast mode helps host compilation but cannot remove frontend/emission work or first-time GPU preparation. The 2-3 s target and 10 s ceiling remain open for larger cases. Raw records include executable sizes and compiler-change detection in ignored `Compiler/output/build-audit.json`; per-file logs end in `-audit.log`.

Remaining build-pipeline work: native C packages compile all providers and extra C sources from their flags once any provider is needed; native cache dependency lists include broad package inventories, so unrelated edits can invalidate builds. GPU builds currently prepare both sm_80 and sm_75 when tensor kernels require them, with large shared runtime PTX. Keep those caches; dropping required GPU code would change program behavior. Frontend, emission and loop/test parallelization belong to the separate compiler optimization session.

Explicit native compiler-package tests exposed another pipeline cost: dependency snapshots alone took 45-48 s per file, before frontend compilation, while the new native-dependency regression itself ran below a millisecond. Static test-data discovery over the broad source inventory is a separate target from normal build snapshots. NativeBuild, NativeFolder, NativeCompiler and Toolchain native tests passed, as did their focused VM checks, Strict VM tests, the build regression script and Compiler/Examples checks. Some existing VM tests still exceed the test budget and need the explicit slow-suite work already listed in improvements.md.

Full verification stays explicit: `strict check <folder>`, `strict test <file>`, `strict test <file> native`, then an ordinary release build. Existing CI provides the broader checks. A combined full-check command should use the file/package parallel runner being developed by the other session, rather than add a sequential testing gate to builds. No unmeasured `-O3` host mode was added; the default remains the validated release pipeline.

## Base-package audit

Inventory covered the 41 root types and 16 immediate packages, 546 `.strict` files in total, with source review of package interfaces, dependency discovery, data readers, collections, parsers and native boundaries. This was a static inventory plus targeted reproductions, not a full test sweep or a line-by-line review of every implementation. Existing low-level compiler/runtime and parallelization work was left with the other session.

The largest measured waste is implicit test-data discovery. A native probe of NativeCompiler dependencies found 350 sources in 0.25 s, then `NativeCache.Data` returned 105,280 paths in 38.8-41.3 s. There were 50,434 unique paths: 49,486 under Tools, with Emscripten/LLVM headers, C/C++ sources, JavaScript, Python and libraries dominating; 994 paths contained `.git/`. Snapshotting those unique paths cost another 5.4-9.3 s. Tool and VCS trees came from quoted strings being treated as directory fixtures by `BinaryFile.Data`/`Found`; executable tool inputs are also tracked separately. This can invalidate test results after unrelated SDK/VCS changes. Raw paths are in ignored `Compiler/output/audit-data.txt`, with the runnable probe `AuditInputs.strict`. Fixed by shared DataFiles discovery: decode string literals with SyntaxNode.Unquoted, resolve each unique literal once, prune .git and ignored directory trees using repository root .gitignore rules, and retain explicit existing file literals even when ignored. The rule file is tracked as an input. The final native probe inventories 354 sources in 0.26 s and 15 fixture/rule inputs in about 0.36 s, then snapshots them in under 1 ms. Before string decoding was fixed, escaped LLVM/path fragments were mistaken for UNC shares: three cold Windows lookups each took about 2.7 s. The focused native regression reproduced 2.81 s and now completes below one second, including Git discovery.

| Area | Observation | Action |
| --- | --- | --- |
| Path | ChangeExtension treated a parent-directory dot as the filename extension | Fixed by reusing RemoveExtension; regression passes VM/native |
| Binary | Float32 decoded infinity and NaN encodings as finite values | Fixed exponent-255 handling; regression passes VM/native; DeltaMesh native build passes |
| Networking | Fixed port and incomplete failure cleanup | OS-assigned ports via Listen(0)/Port; failure cleanup; 16 overlapping process checks pass with the former port occupied |
| Neural native providers | Overlapping calls overwrite returned matrices and race model selection | Thread-owned buffers/caches, thread-exit cleanup, explicit chunk buffers and serialized pool dispatch; concurrent loads/expert outputs plus existing VM/native accuracy tests pass |
| Native socket provider | Winsock startup used a shared plain flag | InitOnceExecuteOnce; 16 caller threads acquire distinct ports |
| MachineLearning | MarkovText ignored failed downloads | Reuses Evaluation.Downloaded; local HTTP 404 regression passes VM/native without publishing a corpus |
| Json / Xml | Repeated get/Elements/Children calls rescan and slice raw text | Intentional lazy interfaces; profile repeated large-document access before adding indexes |
| Binary | Read expands bytes to Numbers; text lengths support only one or two 7-bit bytes | Profile large asset allocation and validate required length range before changing representation |
| Base types | Number.Random/Logarithm and collection bodies have special runtime/native implementations | Do not classify their fallback bodies as ordinary dead or incorrect code |
| NativePlugins | Local ImageLoader/ImageSaver directories contain no Strict source files | Legacy directory inventory, not an active package implementation |

The concurrency fixes above have focused native stress checks in Networking/check_sockets.py and NeuralNetworks/check_native_threads.py. These cover the touched providers and their worker pool, not every external native API. Pool job dispatch is serialized while chunks remain parallel; model metadata and scratch capacity are retained per calling thread until thread exit. Networking documentation was corrected to describe the Errors actually returned by Socket.Connect. Existing improvements entries about File.Length, inferred constant defaults, trait calls and slow test classification remain open; they were not silently removed.

The stale claim that Bootstrap predates eleven attention fields was removed after inspecting its MlirTensors.Kind constants: the seed already contains all twelve current fields, including routing. This does not replace a future fixpoint/runtime compatibility check after new compiler changes. The base-library trigonometry wish list also overlaps existing Degrees, Ratio and Vector2 operations; settle the intended API before adding more Number methods. Root-folder validation (the non-recursive Checked entry point), Binary and Networking validation report zero issues.

Rechecked after merge 7d8aac70: the dependency-discovery implementations above are unchanged, and Path/BinaryData regressions still pass. MlirGroups now looks through a wrapper containing one summing call; its Wrapping.Adding and Wraps tests pass, so that completed improvement entry was removed. GPU measurements from earlier build audits predate the latest mixed-format kernel changes and should not be used as current performance claims.

Native C object reuse now requires matching compilation options in its .o.nativecache sidecar as well as current source/header stamps. Compiler/check_native_cache.py reproduced default -> small retaining the default object (the native __OPTIMIZE_SIZE__ probe returned 2 instead of 1); default -> small -> default now returns 2, 1, 2. Missing option metadata conservatively rebuilds the object. This also keeps fast mode's intentionally optimized native C objects compatible with default builds.

Current verification: Bytecode, Runtime, Compiler, Networking, MachineLearning and NeuralNetworks checks report zero issues. Compiler/Toolchain now scopes Linux installation constants to their use, builds the same arguments with Text.Combine, and removes a redundant Inputs temporary; the complete Linux command was compared with the prior version. Two merged MlirSplit bugs mistook linkage text and thread_local inside string initializers for global attributes. Shared and Global now inspect only the prefix before the symbol; both regressions failed first and pass on the VM and natively. The normal optimized compiler self-build succeeds again (30.59 s: frontend 14.21 s, emission 8.93 s); strict.exe rebuild succeeds (14.34 s). NativeCompiler, Strict and targeted changed-file VM/native tests pass. No new fixpoint was claimed; the existing bootstrap seed remains unchanged.

Post-fix application checks used ordinary optimized builds and retained Release staging. MossGenerate built in 14.20 s with 6.12 s GPU preparation, MossImage in 9.39 s with 4.78 s GPU preparation, and LogoApp in 14.43 s with a 3.58 s clang job. Unchanged calls skipped in 72-113 ms, and no builds ran tests. These preparation-inclusive runs still exceed the 10 s ceiling for MossGenerate and LogoApp; they do not establish the 2-3 s target. Source/frontend/emission parallelization remains with the compiler optimization session.

With GPU/native preparation cached, deleting only each application's `.nativecache` and host `.mlir` forced complete frontend/emission/host builds: MossGenerate 5.27 s, MossImage 4.79 s, LogoApp 4.74 s including Release/Windows staging. The following unchanged calls took 72, 77 and 112 ms respectively. No test execution occurred. These builds meet the 10 s ceiling but still exceed the 2-3 s target; preparation-inclusive builds and the compiler's own frontend/emission remain the larger cases to optimize.

## Thread-pool merge audit (7c336d6e)

Measured on this Windows machine: 32 logical processors, 64 GB RAM; the runtime default is 24 threads. The executables initially predated merged sources. Bootstrapping NativeCompiler at STRICT_THREADS=8 took 34.86 s. A frozen copy of that first-generation compiler was used for thread-count comparisons. Fresh builds removed the native build stamp and the correct host MLIR (Boolean.test.mlir for Boolean); GPU/native caches and Release staging were retained. Raw JSONL, per-file stdout, binaries and the diagnostic sampler are in ignored Compiler/output/ThreadAudit/. Failed output binaries are not counted as successful builds even when an older executable remains beside the source.

| Fresh build | 4 threads | 8 threads | Default (24) |
| --- | ---: | ---: | ---: |
| Boolean | 1.13 s | 1.09 s | 1.11 s |
| Examples/Sum | 0.70 s | 0.68 s | 0.70 s |
| NeuralNetworks/Xor | 2.50 s | 2.48 s | 2.49 s |
| NeuralNetworks/Iris | 2.65 s | 2.59 s | 2.67 s |
| NativeCompiler self-build | 38.08 s | 35.49 s | 34.16 s |

Small-program IR and executables were byte-identical across thread counts. All three self-builds also produced identical IR (SHA-256 e4e5b5eb8ae4df848e495292351f3980efda9e31677eb93ca4cdfe84f7c4c11f) and executable (4f99f45ce29310fa11090e92389c9c10c9b3dec595917861ddcac14eb9d61819), 4,510,720 bytes. The 4-thread self-build overlapped a short failing Strict build, so its wall time is not an isolated comparison. The serial 8/default results show no large scaling gain; self-build frontend remained about 15.9 s and emission 10.3 s.

Correctness blocks broader performance claims. The first generation contains 30 strict_parallel call sites; the self-built second generation contains 108. The latter intermittently crashes compiling Boolean: AccessViolation (0xc0000005), heap corruption (0xc0000374), or an unsupported call of Error.Strict/List in Boolean_$Tests_0. It also sometimes produces the expected binary, so a single successful run is insufficient. Preserved generation2-NativeCompiler.exe and Compiler/NativeCompiler.audit2.exe reproduce it with `build Boolean.strict <Strict root>` under STRICT_THREADS=4 or 8. No runtime or parallelization source was modified by this audit.

Larger cases with the first-generation compiler also need correctness work: Strict with 4 threads reports unsupported Strict/Execute.Run in Strict_Run_1; at 8 threads it exceeded 120 s after frontend (7.22 s). MossGenerate reports unsupported Strict/Safetensors.Shape in MossGenerate_Configured_2 at both 1 and 8 threads. MossImage reports unsupported Strict/Safetensors.Values in MossImage_ComparedRun_4 at 4 threads. These Comfy probes ran from StrictComfy, with the frozen compiler also copied into Compiler/ to preserve TypeSource.Root's executable-location assumption. LogoApp exceeded a 45 s probe budget before reporting frontend; WebWars exceeded 20 s after frontend (6.46 s). Timed-out process trees were killed. These are failures/stalls, not valid build timings. Existing LinkRound/Linker VM tests and Compiler/Bytecode checks passed despite these integration failures.

Windows kernel CPU tracing could not start (WPR profiling policy, 0xc5585011). A process-only sampler of a mapped diagnostic compiler, built from second-generation IR through the existing MLIR split/ThinLTO pipeline, observed 8 threads before an AccessViolation at 3.36 s. In that short, crashing run, CPU-weighted instruction samples attributed about 35% to strict_pool_serve's idle spin and 42% outside the executable. This is not a healthy-build lock profile and does not prove that collector locks dominate. The map and sample JSON are retained for reproduction.

Source review narrows the remaining synchronization: small allocations use per-thread lanes; lane overflow, large allocations and lane refill still use the collector lock. Pinning performs an atomic flag update only for an object not already pinned. While the pool is busy, count updates switch to pinning and early freeing/in-place behavior changes. Scheduling also spins: workers spin up to 200,000 iterations before sleeping, and the caller spins waiting for completion. Profile those paths separately after the bootstrap and missing-call regressions are resolved; deleting locks without restoring ownership guarantees would not validate correctness. Original pre-audit executables were restored; experimental generations remain isolated in ignored output files.


## Pool recovery after c06d0649

Two runtime bugs reproduced before their fixes. A delayed worker could pass the old job check while the caller published the next job's body/count, then claim and execute that new body under the old token. Publishing now invalidates the old token with an acquire/release exchange before changing shared job fields. Separately, pool stores pin references instead of counting them, but strict_rc_fate ignored the pin flag and reclaimed those objects afterward. Reclamation now leaves pinned objects with the collector. No extra allocator lock or conservative loop filter was added. `python Compiler/check_parallel.py` forces both failures against the actual runtime: premature body executions changed from 1 to 0 and reclamation of a pool-pinned object from 1 to 0.

A fresh parallel strict.exe previously crashed in Text.Lines' second test; crash tracing caught an overwritten MethodEntry with the number 2 in its instructions field. After the lifetime fix, 60 fresh Text, Runtime/Frame and Runtime/Operations VM runs at 36 threads passed across two rebuilt versions. Compiler self-builds at 1, 8 and 36 threads produced identical IR and executables. Bootstrap generations also matched. Installed NativeCompiler.exe and strict.exe are clean builds from these sources.

FileCompiler.Checked collected name diagnostics serially. It now collects per-file issue lists independently, then logs and counts them in source order. A two-file invalid project returned identical messages and ordering before/after. The compiler frontend probe measured name validation at 3.68 s before and 0.18 s after at eight threads. Compiler executable size rose from 4,641,280 to 4,643,328 bytes; self-build wall time fell from 26.18 to 22.38 s. The same final code took 47.53 s with one thread and produced the same executable hash, demonstrating about 2.1x total build speedup at eight threads. Earlier 36-thread builds took 31.26 s before this validation change; profiled final runs took about 23.8 s. These results support moderate concurrency here, rather than selecting a new universal default from one machine.

Source-stage review: SourceFiles performs inventory/dependency closure before ResolveCheck builds shapes/inference; Shapes already has a parallel collecting loop. LinkRound independently emits newly needed methods and retains earlier methods across dependency rounds. The C# reference registers type stubs/dependencies before parsing member/method signatures and defers method bodies. Current Strict still repeats validation and metadata work; builds discover sources once and hand that list from the build stamp to the frontend. On this machine compiler discovery took about 0.29 s, initial inference 0.18 s, main codegen 0.006 s, dependency linking 8.0-8.5 s, syntax validation 0.4 s, parallel name validation 0.18-0.28 s and inlining 0.06-0.14 s. File loading alone does not explain the remaining frontend time.

A per-build fused-source snapshot experiment reduced 4,074 reads of 343 canonical files to 2,012 reads, eliminating the 747 reads of List.strict. Linking remained about 8.4 s, so the metadata field, cache plumbing and experimental check were removed. The read trace and frontend probe remain in ignored Compiler/output/PoolRecovery and Compiler/output/FrontendAudit.strict for investigation. Native File.Exists currently calls fopen/fclose before checking the directory bit; replacing that with platform metadata is a focused remaining lookup optimization.

A successful mapped compiler build was sampled without kernel tracing. At eight threads about 3.5% of sampled CPU-weighted instruction locations were in strict_gc_small; a 36-thread run reported 22.8%, with more sampled CPU overall. The function includes the refill lock. Pinning accounted for about 1-2%. System locations included NtWaitForAlertByThreadId, NtCreateFile and NtQueryAttributesFile. This suspension sampler weights current instruction pointers with preceding thread CPU deltas; wait contexts and inlining limit attribution. It is evidence for investigating refill contention and repeated filesystem lookups, not a precise lock-time measurement. Raw maps, JSON and logs are under ignored Compiler/output/ThreadAudit.

Representative ordinary release builds below forced frontend/emission/host work by removing native/bytecode stamps and host MLIR. GPU/native caches and Release staging were retained. No build executed tests. Games ran from the DeltaEngine project root.

| Build | Wall time | Remaining cost |
| --- | ---: | --- |
| Boolean | 1.71 s | Native test executable generated, tests not executed |
| Examples/Sum | 0.86 s | Still above the 100 ms source-build target |
| Examples/ParallelSum | 1.12 s | Native test executable generated |
| NeuralNetworks/Xor | 2.39 s | GPU cache populated |
| NeuralNetworks/Iris | 2.54 s | GPU cache populated |
| Strict | 10.22 s | Frontend 3.87 s; total compilation 9.72 s |
| NativeCompiler | 22.38 s | Frontend 9.82 s, emission 3.74 s |
| StrictComfy/MossGenerate | 5.13 s cached GPU; 27.54 s first preparation | First GPU preparation 22.91 s |
| StrictComfy/MossImage | 4.55 s cached GPU; 8.87 s first preparation | First GPU preparation 4.69 s |
| DeltaEngine/Games/LogoApp | 4.75 s | Release/Windows staging retained |
| DeltaEngine/Games/WebWars | 14.32 s | Frontend 4.16 s, emission 4.08 s |
| NeuralNetworks/Qwen3Benchmark | 32.41 s first preparation | GPU preparation 23.35 s; this run succeeded |

Moss binaries were byte-identical before/after populating GPU caches. Immediate unchanged Moss and game builds skipped in 80-128 ms with source/build timestamps. The prior missing Execute.Run/Safetensors.Shape/Safetensors.Values errors and Strict/LogoApp/WebWars stalls did not recur in these application builds. Cold GPU preparation, WebWars and compiler self-builds still exceed the 10 s ceiling, and the small-file 10/100 ms and complex-build 2-3 s targets remain open.

Targeted VM and native tests passed for FileCompiler, LinkRound, ResolveCheck, MlirParallel, MlirSafety, NativeCompiler and Strict; Frame and Operations also passed native tests. Compiler, Bytecode, Expressions and Runtime validation reported zero issues. These are targeted recovery checks and representative builds, not the full daily repository/model/game execution sweep. Existing long VM tests remain slow (FileCompiler.Parsed alone took about 34 s); classification and enforcement of the test budget remain in improvements.md.

Final recovery checks also passed nine fresh VM runs at 1, 4 and default threads. MossGenerate, MossImage, LogoApp, WebWars and Qwen3Benchmark rebuilt at 36 threads with byte-identical IR/executables to their eight-thread builds; cached-GPU Qwen3Benchmark took 9.07 s. Compiler/check_build.py passed build/test skipping, cache invalidation, Release retention, GPU-cache reuse and whole-file read semantics. A final strict.exe rebuild took 10.75 s and retained its validated hash.


## Detailed build profile after cca586ae

Paired cold-frontend compiler self-builds used the same source and produced identical MLIR and executable hashes at one/eight threads. Native/bytecode stamps and host MLIR were removed; GPU/native caches were retained. Detailed logging was enabled for both runs. Executable size is 4,662,272 bytes (18,944 bytes above the preceding compiler). Stage 2/3 output also matched. These measurements distinguish parallel speedup from costs that remain sequential or dominated by fixed work.

| Stage | One thread | Eight threads | Observation |
| --- | ---: | ---: | --- |
| Entire compiler build, wall | 46.57 s | 21.74 s | 2.14x overall |
| Source discovery | 0.284 s | 0.271 s | Little scaling; 343-file inventory |
| Shape loading/fusion | 0.042 s | 0.022 s | Small cost already |
| Type inference, including shapes | 0.208 s | 0.189 s | Scopes/refinement dominate the remainder |
| Dependency linking | 13.036 s | 8.202 s | Only 1.59x; repeated late passes remain |
| Syntax validation | 0.395 s | 0.422 s | Essentially unchanged |
| Name validation | 1.320 s | 0.182 s | 7.26x |
| Reachability | 0.436 s | 0.344 s | Little scaling |
| Safety and recursion edges | 0.725 s | 0.611 s | Little scaling |
| Recursive-function analysis | 0.341 s | 0.524 s | No demonstrated gain |
| Function-body emission | 5.859 s | 0.862 s | 6.79x |
| Globals/text assembly | 1.175 s | 1.227 s | Essentially unchanged |
| Total MLIR emission | 8.819 s | 3.838 s | Faster bodies leave analysis/assembly dominant |
| Host lowering/translation | 14.513 s | 3.392 s | Existing independent tool jobs benefit strongly |
| Host clang batch | 3.302 s | 1.015 s | Independent jobs scale |
| Final linker | 2.804 s | 2.578 s | Little scaling |

Each link pass adds one level of calls; linking ends when a pass asks no type for new methods and the main type gained none (`LinkRound.IsSettled`), the compiler settles on pass 21. A pass that does not settle within `Linker.Passes` (50) fails with `LinkingDidNotConverge` and the pending signatures instead of returning a truncated program (the old 20-pass guard had cut HelloLogger's 15 linked types to 5 in its own tests). A type with one member forwards a call to its member's type only once it was asked for that method in an earlier pass and still lacks it: forwarding every method not linked yet sent `BuildProfile.Linked` to `Logger.Linked` or `TypeShape.MemberScope` to `List(Text)` for one pass, then recompiled `Logger`, `Text` or `List(Text)` from scratch when the forward vanished (139 such relinks per self-build, types dropping out and coming back). Signatures list method names sorted and separated by `;` (specialized names like `F(A, B)` stay whole), so another call order is no change; the main type is completed again with the forwarded calls of its pass. Types linked in the previous pass are not checked on disk again (`Linkable` had read `List.strict` once per list type and pass, 747 times per self-build), base types are collected with `Unique` and each entry's used types are deduplicated before list elements are looked up (two file probes each). Compiler self-build at eight threads: dependency linking 10.75 to 3.77 s, frontend 13.5 to 6.3 s; Strict links in 1.55 s. The compiler's IR is byte-identical to the old linker's, Strict's differs only in the order of two generated equality helpers.

Representative eight-thread profiles, same default optimized release pipeline:

| Program | Frontend | MLIR emission | GPU preparation | Final link | Build total |
| --- | ---: | ---: | ---: | ---: | ---: |
| Examples/Sum | 0.079 s | 0.003 s | 0 s | 0.254 s | 0.814 s |
| NeuralNetworks/Xor | 0.660 s | 0.098 s | 0.068 s | 0.554 s | 2.342 s |
| Strict | 4.123 s | 1.692 s | 0.251 s | 1.498 s | 10.380 s |
| MossGenerate, GPU/native caches populated | 1.126 s | 1.204 s | 0.599 s | 0.723 s | 5.210 s |
| MossImage, GPU/native caches populated | 1.040 s | 1.097 s | 0.423 s | 0.631 s | 4.681 s |
| LogoApp | 1.880 s | 0.184 s | 0.001 s | 0.479 s | 4.973 s |
| WebWars | 4.600 s | 4.133 s | 0.519 s | 1.297 s | 14.287 s |

Sum spends most time in fixed backend/process work, including 16 clang jobs for only two emitted functions. Xor inventories 458 files for 26 linked types. Moss inventories 492 files for 90-97 linked types. LogoApp/WebWars inventory 770 files in about 1.1 s. The dependency inventory and the frontend used to discover them separately, now `BuildProfile.Compiled` discovers once and passes the list down: discovery plus inventory in the compiler self-build went from 1.2-1.4 s to 0.6-0.8 s, build total from 11.9-12.4 s to 11.0-11.3 s (unchanged IR, backend skipped). Narrow that inventory before spending effort on already-small shape parsing (about 0.02 s). Release staging cost 0.061 s for LogoApp and 0.399 s for WebWars in these runs.

WebWars emission is concentrated in generated atlas constructors: seven SceneAtlas_Group/LevelAtlas_Group helpers each emit 306-583 KB of MLIR, about 3.77 MB combined. An isolated parallel per-function probe measured 0.22-0.88 s each, while ordinary game functions were mostly below 25 ms. Their summed wall durations overlap across threads; the measured whole function-body batch was 2.955 s. SceneAtlas.strict is 59,465 source bytes/487 lines and LevelAtlas.strict 39,747 bytes/314 lines. Investigate large constant/record-list lowering and generated asset representation, preserving data exactly, rather than treating every function as an equal unit of work.

Moss has a different emission profile: GPU-kernel emission takes 0.78-0.85 s while host function bodies take 0.10-0.12 s. On the first run after merged GPU/native updates, MossGenerate spent 24.125 s in GPU preparation: program PTX 3.712 s, shared runtime PTX 17.156 s, plus lowering/translation/cache work. Five native C compilations added about 4.45 s sequentially. First build total was 33.345 s; a forced source rebuild with these caches populated took 5.210 s and retained identical binary bytes. MossImage first GPU preparation was 4.988 s, then 0.423 s cached. Further targets are emitted GPU variants/runtime size, independent native C job batching and cache reuse, not the already-fast host-body emission.

Raw logs and parsed stage data are in ignored Compiler/output/ThreadAudit and Compiler/output/ProfileAudit/stages*.json; LinkAudit.strict records call-count/signature changes and EmissionAudit.strict records per-function emission costs. Profiling does not disable parallel loops or add logging inside collecting worker bodies. Native tests passed for BuildProfile, NativeCompiler, MlirNatives, Linker, FileCompiler, ResolveCheck, Hosted, Execute, NativeBuild, NativeFolder, Toolchain, MlirModule and DeviceBuild. VM checks passed for these except NativeFolder, which exceeded its 120-second budget and was killed in Counted; existing NativeBuild and MlirModule VM checks took about 96/88 s. Full daily repository/model execution was not repeated. Slow VM integration checks still need explicit classification.


## Test policy

`Slow(T)` is transparent for code generation: MethodEntry.returnType exposes T while declaredType preserves the wrapper and IsSlow exposes the marker, including after bytecode serialization, resolution and inlining. HeaderEntries records the source marker for callers. MlirCost uses it as a cost hint without bypassing loop purity or GPU safety.

NativeCompiler checks unit tests through its own VM before building, so bootstrap does not depend on an existing strict executable. Native test executables use the same Started/Finished budget hooks as hosted VM tests. A monotonic clock and versioned watchdog enforce 20 ms; completion and the watchdog share one failure reporter and stop the process without running concurrent CRT teardown. Explicit Slow tests do not arm this deadline. Web test executables report TestBudgetWatchdogUnavailable rather than silently running unenforced units; ordinary web builds check their units on the host VM first.

Unit-only bytecode/native stamps are distinct from explicit functional execution. Reachable test entry generation excludes unused constant getters, avoiding model/asset initialization for excluded methods. Named constructor defaults now qualify constants in their declaring type: NativeBuild's Fast must be NativeBuild.Fast, not an uninitialized Fast local in NativeCompiler. The existing named-argument fixture reproduced this before the fix.

Verification: VM/native deadline and infinite-loop regressions, validator/build rejection, explicit method selection, transparent Slow values, excluded fixtures, manual test-before-Run order, build/cache/staging and detailed-profile checks pass. Native HeaderTokens, MethodEntry, NamedArguments and MlirCost tests pass. Compiler self-builds have identical MLIR and executable hashes; CLI rebuild passes its unit gate. The regenerated bootstrap lowers for Windows/Linux/macOS, and the Windows seed compiler independently builds and runs a program. All 152 plugin unit tests pass; lint has no errors (four existing Android environment-name warnings). Static validation passes changed packages. Existing unmarked VM tests in TypeCodegen, ValueCodegen, ErrorCodegen, Specialization, NativeBuild, NativeFolder and BinaryFile still hit the new budget; package unit checks are not all green. Full daily suites and non-Windows execution were not repeated.

Shared fixture budgets now use lazy VM constants rather than timing eager preparation. Linker treats hosted package methods as native and restores ordinary main-type methods reached through callbacks, alongside specialized methods. Native runtime assembly (`NativeBuild.Mlir`) and folder compilation (`NativeFolder.Counted`) are marked Slow; their assertions remain available explicitly. Final compiler self-builds match in MLIR and executable bytes, and both compiler and CLI were rebuilt. Deadline/lazy-cache, frame initialization (17.40 ms for 10,000 frames), build/staging/cache and profile scripts pass; changed-package static validation reports zero issues. Native Constants, Machine, Frame, Accessor, MlirNatives, MlirPurity, Linker and Specialization tests pass, as do 152 plugin tests. Package-wide VM unit checks still expose genuine budget failures; no blanket Slow exemption was added. The regenerated seed lowers for Windows/Linux/macOS and independently builds and runs a Windows program. Full daily suites were not repeated.
