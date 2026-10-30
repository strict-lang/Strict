# Performance
Open speed work, one short line each, measurements go into commit messages. GPU details: [improvements-gpu.md](improvements-gpu.md), models: [improvements-models.md](improvements-models.md).

- The VM is about 100 times slower than native on compiler code (boxed numbers, dispatch, allocation): the reason internal package tests count as Slow.
- Testing one file on the VM (VS Code, and every file of a folder run in its own process) spends 2-4 s before its first test, all of it again every run: source discovery 0.4 s, shapes and inference of all 313 files 0.5 s, 17 dependency link rounds 1.4 s (0.6 compiling the used methods of 199 types, the rest every round over all entries again: Targets, Forwarded, BaseTypes), resolving them for the VM 0.65 s (Compiler/MlirLibrary.strict).
- VM calls dominate game logic (a member read through a method like `Vector2.X` is two calls): DeltaEngine's multi tick simulation and content loading tests are Slow for it.
- Release packaging runs a game's ContentBuild.strict on the VM (SoulCraft 7.7 s, natively 1 s plus a cached 0.6 s build): build it natively and run the executable (`Release` has 16 methods).
- Build speed: under 10 s hard limit, 2-3 s target (LogoApp, Blocks and Breakout 3.6-4 s): a build with tests runs the frontend twice (test gate, then native), lowering and the thin LTO link are the largest phases left.
- Cold GPU builds spend most time on the shared runtime PTX for both sm_75 and sm_80.
- `MlirCost` sums expensive alternative call paths and overestimates tiny loops.
- Collector lane refill (`strict_gc_small` taking the collector lock) contends at high thread counts.
- Trees as pools with 32 bit backward offsets ([Goose](https://github.com/aardappel/goose)): count SyntaxNode allocations of `strict build Strict.strict` first.
- After emission the whole module text is copied about 8 times: `NativeBuild.Device` searches `"\n" + mlir`, `NativeBuild.Mlir` chains `+`, `MlirSplit` reads the written file back.
- All native providers compile once any of them is needed.
- Reachability, safety and globals/text assembly of MLIR emission run serially.
- A native library call copies every Numbers argument into new doubles (`strict_list_flat`, 1.2 GB for 148.6M numbers, unpacking packed lists first).
- ParticleStress at 1 million particles (8.3 ms per tick, Unity's particle system 9.5): the parallel loop takes 6.6 ms, reading and writing 80 bytes per particle alone needs 4.6 at this machine's memory bandwidth (32 bit slots would halve it), the rest is the work per particle (41 ns on one P core: a fifth the pow of `grow ^ seconds`, 7% the two fmod of Torque2D.Turned, the rest checks after every call and f32 pair conversions).
- A `for` that drops items collects objects in parallel, but numbers and flat elements stay on one thread (any bit pattern is a number, so no hole marker): keep a bitmap of the kept slots instead.
