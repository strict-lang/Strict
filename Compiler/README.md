# Strict.Compiler
Builds native executables: `NativeBuild` behind `strict build` turns the instructions of all used types into MLIR (`MlirModule`, one `MlirFunction` per method with `MlirStatements`, `MlirCalls`, `MlirLoops` and friends, `MlirKernel` and `MlirFastLoop` add a copy without per element checks to simple loops over lists, LLVM vectorizes it), adds the runtime written directly in MLIR (`NativeRuntime.mlir` for texts, lists, files, processes and the garbage collector, `NativeRuntimeWindows.mlir` or `NativeRuntimeLinux.mlir` for the platform) and runs mlir-opt, mlir-translate and clang (`Toolchain`, `ToolRunner`). mlir-opt first drops every private function nothing calls (`--symbol-dce`, the runtime helpers a program does not use), for the host and the GPU module alike: 420 test and example executables 26% smaller (FizzBuzz 387 to 224 KB, clang 2.1 to 1.0 s), the PTX of a small GPU program 132 to 6 KB (NearestNeighbors), `strict.exe` 3.67 to 3.55 MB, same speed. The GPU module of a successful build is kept as `Name.kernels.mlir` next to `Name.ptx` (`PtxCache`): when emitted GPU input, code-generation flags and LLVM tools match, the PTX is reused instead of running mlir-opt, mlir-translate and clang for sm_75 and sm_80 again (MossBlock.test 13.4 to 6.8 s). Builds never run tests or generate an extra test executable for a program. A type without `Run` builds its test executable without running it. `strict test <file> native` explicitly builds and runs native tests; `strict test <file> lines` also instruments per-line timings. Fresh program and library builds omit test bytecode and its dependencies; existing VM bytecode caches remain usable. Release packaging retains the build executable and publishes a checked temporary copy, so repeated game builds can reuse it.

A type of two 32 bit numbers (Vector2 `MlirPairs`, Size `MlirCouples`) is one 64-bit slot. A register holding a Vector2 that is read as a list (a `Numbers` parameter or return, a List method other than X, Y and Length, a loop, list arithmetic) becomes a list right after it is written, a list given or returned as a Vector2 becomes a pair (`MlirPacking`, the first such read decides), an index into a Vector2 reads the pair (`strict_two_at`). One-number types with `value.Bits is 32` (Degrees) are rounded to 32 bit when constructed, like on the Strict VM.

Values are rewritten in place where nobody can see the old one: the dying value of a local, a list mapped to the same type or a returned struct becomes the destination of the call or loop replacing it (`MlirOwned`, `MlirDestined`, `MlirListDestined`, `MlirStructs`). A list of a flat type is one allocation with its elements inline (`strict_flat_list_new`).

`NativeCompiler.strict` is the native compiler `strict` hands its builds to (`Process.Started`, the output appears right away), so the front end `strict` stays small (2.4 MB instead of 5.2 MB) and builds of other programs do not rebuild it. `Bootstrap.mlir` is the seed, `NativeCompiler` compiled by itself: `Bootstrap.cmd` (Windows) and `bootstrap.sh` (Linux and macOS) build `NativeCompiler` from it, which then builds `strict`.

Completed steps taking at least 3 seconds print their elapsed time in seconds: dependency snapshot, frontend, MLIR emission, GPU preparation and external tools. The final host linker is labeled `Linking`. Faster steps stay silent; there are no tool-start messages. Total compilation is also reported only at 3 seconds or more and excludes dependency snapshots and process startup. GPU preparation includes its nested LLVM tools; do not add those twice. Unchanged builds print only `Up to date` and run no compiler or tests. `check_build.py` checks both fast-step suppression and slow-step reporting in the VM and native code without waiting three seconds.

The previous bootstrap seed still called `NativeGate`, so rebuilding from it ran tests and used the older compiler pipeline despite current source removing that gate. The seed now comes from byte-identical stages 2 and 3 (SHA-256 `605746933f973a03af535a3a0f051185dcb6115a0754fe95f2bc16e8ce99cef6`). A separately compiled seed built and ran a program containing a failing inline test without creating or running a test executable. `check_build.py` rejects seeds containing the old gate.

Native model:
- Every value is one 64-bit slot: numbers are doubles, texts, lists and instances pointer bits, Errors carry a NaN tag so `x is Error` works for any type.
- Lists of numbers that C code hands out stay packed (32 bit floats, 16 bit halves, 8, 6 or 4 bit quantized, `Float8` for E4M3) and loops read them directly in 32 bit floats.
- A type with one member is that member, small number types (`Vector3`) are one block of floats, lists append and write in place while they are the newest version.
- A collecting `for` whose body only calculates runs on a pool of CPU threads, LLVM vectorizes the inner loops (SIMD).
- GPU: such loops become kernels (PTX built by clang, loaded through `nvcuda.dll`/`libcuda.so.1`, no CUDA toolkit to run; AMD through HIP from `.hsaco`; Metal on macOS, experimental). Matrix products of tokens (`Batch.Sums`, `Batch.Grouped`) use the int8 tensor core kernels of `NativeRuntimeTokens.mlir`, `Attention.Merged` the 16 bit FlashAttention-2 style kernel of `NativeRuntimeAttention.mlir` and `Products` the 16 bit tensor core products of `NativeRuntimeProducts.mlir`, launches are captured into CUDA graphs and results stay on the GPU until the CPU reads them. `STRICT_GPU=0` keeps everything on the CPU, `STRICT_GPU=all` sends every possible loop to the GPU, `STRICT_GRAPHS`, `STRICT_DP4A`, `STRICT_GPU_WARM` and `STRICT_GPU_SHARE` switch single features.
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
