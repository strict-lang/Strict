# Performance
Open speed work, one short line each, measurements go into commit messages. GPU details: [improvements-gpu.md](improvements-gpu.md), models: [improvements-models.md](improvements-models.md).

- The VM is about 100 times slower than native on compiler code (boxed numbers, dispatch, allocation): the reason internal package tests count as Slow.
- Testing one file on the VM spends 0.7-4.3 s before its first test (Compiler/MlirLibrary 4.3 s, MlirCalls 2.7 s), all of it again every run: the link rounds themselves are cheap, compiling the used methods takes 10-30 ms per type, Changes(lines).Members and .Methods are recomputed per method body and call site (TypeShape.CallScope, MethodBodies, ChangedInstance) and source discovery tokenizes every file (SourceFiles.PackagesIn).
- VM calls dominate game logic (a member read through a method like `Vector2.X` is two calls): DeltaEngine's multi tick simulation and content loading tests are Slow for it.
- Build speed: under 10 s hard limit, 2-3 s target (LogoApp, Blocks and Breakout 3.6-4 s): a build with tests runs the frontend twice (test gate, then native), lowering and the thin LTO link are the largest phases left.
- Cold GPU builds spend most time on the shared runtime PTX for both sm_75 and sm_80.
- `MlirCost` sums expensive alternative call paths and overestimates tiny loops: a loop called again is probed on the caller once its last fork took under 1 ms, a loop run once still forks.
- Collector lane refill: a lane still takes the collector lock for every waiting page it pops and every fresh page (`strict_gc_refill_busy`), one shared lock for 64 lanes.
- Trees as pools with 32 bit backward offsets ([Goose](https://github.com/aardappel/goose)): count SyntaxNode allocations of `strict build Strict.strict` first.
- GPU preparation takes 0.5-0.6 s of every build with kernels also when nothing changed (`DeviceBuild.Built`: `PtxCache` reads the runtime PTX and hashes the kernel text with `HashCode` per character), `MlirSplit` still reads the just written module back.
- All native providers compile once any of them is needed.
- Reachability, safety and globals/text assembly of MLIR emission run serially.
- A native library call copies every Numbers argument into new doubles (`strict_list_flat`, 1.2 GB for 148.6M numbers, unpacking packed lists first).
- ParticleStress at 1 million particles (8.0 ms per tick, Unity's particle system 9.5): the parallel loop takes 6.6 ms, reading and writing 80 bytes per particle alone needs 4.6 at this machine's memory bandwidth (32 bit slots would halve it), the rest is the work per particle (41 ns on one P core: a fifth the pow of `grow ^ seconds`, 7% the two fmod of Torque2D.Turned, the rest checks after every call and f32 pair conversions).
- A `for` that drops items collects objects in parallel, but numbers and flat elements stay on one thread (any bit pattern is a number, so no hole marker): keep a bitmap of the kept slots instead.
- Emitting one huge function is quadratic: `MlirKinds.KindOf` walks back through the method for every register and variable kind (Writer, Stored, VariableKind/FieldKind chains) without a table per function, DeltaEngine's `RtsHost` test function (126k MLIR lines, 212 loops, 2049 variables) still spends 12.9 s in function bodies, `strict test RealTimeStrategy native` about 134 s before.
