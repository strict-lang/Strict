# Performance
Open speed work, one short line each, measurements go into commit messages. GPU details: [improvements-gpu.md](improvements-gpu.md), models: [improvements-models.md](improvements-models.md).

- The VM is about 100 times slower than native on compiler code (boxed numbers, dispatch, allocation): the reason internal package tests count as Slow.
- VM calls dominate game logic (a member read through a method like `Vector2.X` is two calls): DeltaEngine's multi tick simulation and content loading tests are Slow for it.
- Build speed: under 10 s hard limit, 2-3 s target (LogoApp, Blocks and Breakout 3.6-4 s): a build with tests runs the frontend twice (test gate, then native), lowering and the thin LTO link are the largest phases left.
- Cold GPU builds spend most time on the shared runtime PTX for both sm_75 and sm_80.
- WebWars' generated SceneAtlas/LevelAtlas constructors dominate its emission.
- `MlirCost` sums expensive alternative call paths and overestimates tiny loops.
- Collector lane refill (`strict_gc_small` taking the collector lock) contends at high thread counts.
- Trees as pools with 32 bit backward offsets ([Goose](https://github.com/aardappel/goose)): count SyntaxNode allocations of `strict build Strict.strict` first.
- After emission the whole module text is copied about 8 times: `NativeBuild.Device` searches `"\n" + mlir`, `NativeBuild.Mlir` chains `+`, `MlirSplit` reads the written file back.
- All native providers compile once any of them is needed.
- Reachability, safety and globals/text assembly of MLIR emission run serially.
- A native library call copies every Numbers argument into new doubles (`strict_list_flat`, 1.2 GB for 148.6M numbers, unpacking packed lists first).
- A `for` whose body drops items (an `if` without `else` as last line, like DeltaEngine `EffectSystem.Living`) runs on one thread: DeltaEngine Games/ParticleStress advances its 1 million particles single threaded.
