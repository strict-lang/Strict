# Strict.Compiler
Builds native executables. `NativeBuild` behind `strict build` turns the instructions of all used types into MLIR (`MlirModule`, one `MlirFunction` per method with `MlirStatements`, `MlirCalls`, `MlirLoops` and friends; `MlirKernel` and `MlirFastLoop` add a copy without per element checks to simple loops over lists, which LLVM vectorizes). It adds the runtime written directly in MLIR (`NativeRuntime.mlir` for texts, lists, files, processes and the garbage collector, `NativeRuntimeWindows.mlir` or `NativeRuntimeLinux.mlir` for the platform) and runs mlir-opt, mlir-translate and clang (`Toolchain`, `ToolRunner`). mlir-opt first drops every private function nothing calls (`--symbol-dce`), for the host and the GPU module alike.

`NativeCompiler.strict` is the native compiler `strict` hands its builds to (`Process.Started`, output appears right away), so the front end `strict` stays small and builds of other programs do not rebuild it. `Bootstrap.mlir` is the seed, `NativeCompiler` compiled by itself (stages 2 and 3 identical): `Bootstrap.cmd` (Windows) and `bootstrap.sh` (Linux and macOS) build `NativeCompiler` from it, which then builds `strict`. A build holds `<source>.lock` (`File.Locked`, the operating system releases it when the process ends, also after a crash or kill), another build of the same target waits for it up to 300 s.

## Building
- Builds first require passing VM unit tests, reusing unchanged passing results; Slow tests only run when asked for. Program and library emission omit test bytecode.
- A type without `Run` builds its test executable without running it. `strict test <file> native` builds and runs native tests, `strict test <file> lines` also records per-line timings.
- `strict build <file>` is the optimized release build (`-O2`), `small` uses `-Os`, `fast` uses `-O0` and skips LLVM splitting and ThinLTO (current desktop target only, no other options). Native C objects and GPU kernels are shared between modes.
- `NativeCache` keeps a `.nativecache` stamp of sources, native C/header/flag inputs, folder membership, content, compiler and tool binaries and outputs. `strict` checks it before launching NativeCompiler; an unchanged build or native test does nothing and prints `Up to date: X.strict (last changed ...) < X.exe (last build ...)` or `Tests up to date:`. Options are part of the key. Cross-platform, library and instrumented builds bypass it.
- Native C objects are reused only when their `.o.nativecache` sidecar matches sources, headers and compile options.
- The GPU module of a successful build is kept as `Name.kernels.mlir` next to `Name.ptx` (`PtxCache`): when emitted input, target flags, exported symbols and LLVM tools match, the PTX is reused. Shared runtime PTX lives in `Tools/PtxCache`, published from a private staging directory so parallel builds never read partial results.
- Host modules over 100,000 lines are cut into 16 parts by `MlirSplit`: each part keeps its own functions, declares the others and the globals (defined in part 0), runs mlir-opt and mlir-translate in parallel, then goes through clang and ThinLTO. `fast` builds keep one module.
- Release packaging keeps the build executable and publishes a checked copy, so repeated game builds reuse it.
- `python Compiler/check_build.py` checks build and test skipping, cache invalidation, Release staging, GPU cache reuse and that builds of one target take turns on its lock.

Steps taking at least 3 seconds print their time. `STRICT_BUILD_PROFILE=1` prints every stage (cache hits, linking passes, emission substeps, GPU batches, external tools, packaging) without changing the output. Parent stages include their children, batches report all concurrent jobs together.
```powershell
$env:STRICT_BUILD_PROFILE = "1"
.\strict.exe build Strict.strict
Remove-Item Env:STRICT_BUILD_PROFILE
```

## Linking
Each link pass adds one level of calls; linking ends when a pass asks no type for new methods and the main type gained none (`LinkRound.IsSettled`). A pass that does not settle within `Linker.Passes` fails with `LinkingDidNotConverge` and the pending signatures. A type with one member forwards a call to its member's type only once it was asked for that method in an earlier pass and still lacks it. Signatures list method names sorted and separated by `;`, so another call order is no change. Native folders are only compiled in when a linked type has a foreign method (`MlirForeign.IsForeign`).

## Native model
- Every value is one 64-bit slot: numbers are doubles, texts, lists and instances pointer bits, Errors carry a NaN tag so `x is Error` works for any type.
- A type of two 32 bit numbers (Vector2 `MlirPairs`, Size `MlirCouples`) is one slot. A Vector2 read as a list (a `Numbers` parameter or return, a List method other than X, Y and Length, a loop, list arithmetic) becomes a list right after it is written, a list given or returned as a Vector2 becomes a pair (`MlirPacking`), an index into a Vector2 reads the pair (`strict_two_at`). One-number types with `value.Bits is 32` (Degrees) are rounded to 32 bit when constructed, like on the Strict VM.
- Lists of numbers that C code hands out stay packed (32 bit floats, 16 bit halves, 8, 6 or 4 bit quantized, `Float8` for E4M3) and loops read them directly as 32 bit floats. A list of a flat type is one allocation with its elements inline (`strict_flat_list_new`).
- A type with one member is that member, small number types (`Vector3`) are one block of floats.
- Values are rewritten in place where nobody can see the old one: the dying value of a local, a list mapped to the same type or a returned struct becomes the destination of the call or loop replacing it (`MlirOwned`, `MlirDestined`, `MlirListDestined`, `MlirStructs`). Lists append and write in place while they are the newest version.
- Whole-file `Text.Combine(File(path).ReadLines, "\n")` lowers to one normalized text read (`MlirReadText`).
- Memory: a conservative mark-sweep collector with a young generation plus reference counts that free large objects early, so a loop doing the same steps reuses the same memory. `STRICT_YOUNG`, `STRICT_RC=0` and `STRICT_RC_CHECK=1` tune and check it.

## Parallel loops
A collecting `for` runs on a pool of CPU threads when `MlirSafety` finds nothing unsafe in everything it calls (Random, Library, readers and writers, foreign C, file writes, process execution, logging and writes to parameters or members; known read-only file operations are allowed) and the estimated work (`MlirCost`: instructions per iteration including called methods, inner loops 15 times, external calls 20,000) times the count reaches 50,000. Nested GPU loops use the CPU on pool threads. `STRICT_THREADS` sets the thread count.

While the pool runs each thread allocates from its own lane (`strict_gc_lanes`, the collector lock only for new pages and large objects) and nothing is counted: storing a reference pins the object, the collector frees it later. A lane keeps its spans into the next pool job; threads past the 511th share lane 0. A job wakes only as many sleeping workers as it has chunks (`strict_pool_wake`), the caller sleeps until the last chunk finishes (`strict_pool_await`). Tools started together by `Process.RunTools` share the threads: unless `STRICT_THREADS` is set each child gets the threads divided by how many of its batch run at once, at least 4; the share belongs to the batch (the thread starting its tools holds it), so overlapping batches keep their own. `python Compiler/check_parallel.py` checks job publishing and pinned object reclamation.

Native constants use an atomic state, unset, its owner or ready: one caller claims it with its owner token (the address of its thread local `strict_constant_wait`), initializes and pins the value, then publishes it with release ordering; readers acquire ready before reading, without touching thread local memory. Access by the owner returns `RecursiveConstant`, so does waiting in a cycle: a waiter notes the state it waits for in its `strict_constant_wait` and follows owners and what they wait for (`strict_constant_blocked`), a chain back to itself never finishes. `python Compiler/check_parallel.py` checks both.

## GPU
Collecting loops become kernels: PTX built by clang and loaded through `nvcuda.dll`/`libcuda.so.1` (no CUDA toolkit needed to run), AMD through HIP from `.hsaco`, Metal on macOS (experimental). Matrix products of tokens (`Batch.Sums`, `Batch.Grouped`, `Batch.Gated` with SiLU of the gate rows times the up rows in the epilogue, `MlirGated`) use the int8 tensor core kernels of `NativeRuntimeTokens.mlir`, `Attention.Merged` the 16 bit FlashAttention-2 style kernel of `NativeRuntimeAttention.mlir` and `Products` the 16 bit tensor core products of `NativeRuntimeProducts.mlir`. Launches are captured into CUDA graphs and results stay on the GPU until the CPU reads them.

`STRICT_GPU=0` keeps everything on the CPU, `STRICT_GPU=all` sends every possible loop to the GPU, `STRICT_GRAPHS`, `STRICT_DP4A`, `STRICT_GPU_WARM` and `STRICT_GPU_SHARE` switch single features.

When a collection and the free pool blocks are not enough, cached copies of host lists are evicted by age once queued work is done. Argument buffers stay protected while a call dispatches nested work, lists that only exist on the GPU are never evicted, weights that do not fit are spilled and streamed back ahead of their use. `python Compiler/check_gpu_cache.py` checks these rules.

## Native packages and libraries
A type whose methods have no body (`Examples/NativeAdd/Adder.strict`) is implemented in C in the `native/` folder next to it: `Adder.Add(1, 2)` calls `double Adder_Add(double self, double left, double right)`. A type with one member holds a C handle, its bodyless `from` is the constructor. Every argument and result is one double, `strict.h` reads and returns texts, number lists and packed rows, `strict_fail("Name")` returns an Error. `strict build` compiles the C files once and links them with the flags of `native/Windows.flags`, `Linux.flags` or `MacOS.flags`, the Strict VM loads them as a library.

`strict build X.strict library` exports every public method with Number, Text, Boolean, Numbers or number record values as C functions (0 success, 1 for an Error, `strict_last_error()`), plus a generated C# wrapper.

## Mobile
iOS: `strict build X.strict ios` stages an arm64/iOS 15 static archive exporting `strict_main` plus content in `Release/iOS`. Pure Strict archives build on Windows, native packages need Xcode and `native/IOS.flags` on macOS. App linking, UIKit, signing and running need the MacMini, see [IOS.md](IOS.md).

Android: `strict build X.strict android` runs the tests on the host VM, then builds an arm64-v8a shared library for API 26 (`ANDROID_NDK_ROOT`, or an SDK with `ndk/<version>` in `ANDROID_HOME`, `ANDROID_SDK_ROOT` or `%LOCALAPPDATA%/Android/Sdk`). Host LLVM compiles the IR, NDK clang compiles native packages with their `Android.flags` and links with 16 KB ELF alignment, sharing the POSIX runtime with Linux. Output goes to `Release/Android/jniLibs/arm64-v8a/libmain.so` and `Release/Android/assets/Content`, usable as Gradle source sets. The APK/AAB, signing, Activity/JNI glue, window, input, audio and textures belong to DeltaEngine. Compute loops run on the CPU, the tests of a type without `Run` need an Android runner.

## Tests
`Slow(T)` is transparent for code generation: `MethodEntry.returnType` exposes T, `declaredType` keeps the wrapper and `IsSlow` the marker, also after serialization, resolution and inlining. NativeCompiler checks unit tests through its own VM before building, so bootstrap does not need an existing `strict`. Native test executables use the same budget hooks as VM tests: a monotonic clock and watchdog enforce 100 ms per test, Slow tests have no deadline. Web test executables report `TestBudgetWatchdogUnavailable`, ordinary web builds check their units on the host VM first. Test entries only include reachable methods and constant getters, so unused model fixtures never initialize.
