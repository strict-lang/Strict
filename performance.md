# Performance
Open speed work, one short line each, measurements go into commit messages. GPU details: [improvements-gpu.md](improvements-gpu.md), models: [improvements-models.md](improvements-models.md).

- The VM is about 100 times slower than native on compiler code (boxed numbers, dispatch, allocation): the reason internal package tests count as Slow.
- Build speed: under 10 s hard limit, 2-3 s target (LogoApp, Blocks and Breakout 3.6-4 s): a build with tests runs the frontend twice (test gate, then native), lowering and the thin LTO link are the largest phases left.
- Reachability, safety, recursion analysis and globals/text assembly of MLIR emission run serially.
- Native C files of packages compile one after another and all providers compile once any is needed.
- Cold GPU builds spend most time on the shared runtime PTX for both sm_75 and sm_80.
- WebWars' generated SceneAtlas/LevelAtlas constructors dominate its emission.
- Native `File.Exists` opens and closes each file before checking its directory bit.
- `MlirCost` sums expensive alternative call paths and overestimates tiny loops.
- Collector lane refill (`strict_gc_small` taking the collector lock) contends at high thread counts.
- `BinaryData.Read` keeps one Number per byte: a packed byte list for large assets.
- Trees as pools with 32 bit backward offsets ([Goose](https://github.com/aardappel/goose)): count SyntaxNode allocations of `strict build Strict.strict` first.
