# Improvements
What to fix next, one short line each, bugs first. Remove a line once it is done, add what is found at the end. No history or measurements (those go into commit messages). Speed work: [performance.md](performance.md).

## Bugs
- The GPU budget ignores other programs on Windows: WDDM's `cuMemGetInfo` reported 22.7 of 24 GB free while another process held 12.5 GB, so Strict grew to its full peak and Windows paged the other process's memory to RAM, slow tokens until done (this was the Ternary Bonsai "10 GB stall", on the first run under every fresh holder); next to ComfyUI the figure moved by 0.9 GB between starts. DXGI's QueryVideoMemoryInfo budget and its change notification give the real limit.
- A pure loop over GPU heavy calls forks onto CPU threads (MlirCost's threshold is 2 for a costly body): `Qwen3Images.Encoded` of 2 images ran its vision towers in workers and did not finish in 4.5 min, sequential takes 16 s more than one image.
- GPT-2 got slower since its 2026-10-22 numbers on the 3080: `Gpt2Benchmark` decodes 1.74 instead of 1.10 ms per token on the GPU and 10.6 instead of 9.3 ms with STRICT_GPU=0 (BenchBlur on the GPU 7.8 instead of 6.6 ms, 5.5 ms of CPU before its 0.7 ms kernel); the Oct 20 Gpt2Text.exe still runs, older commits need their own bootstrap to bisect (the current compiler emits calls their runtime lacks, LLVM 22 runtimes do not build with 23)
- When the native library of a package fails to build, the VM returns clang's error text as the result of the native call instead of an Error (HttpChecks compared it with the expected HMAC digest).
- Reading a field of an element of a big `Json` costs time and memory in proportion to the whole document: Rebalancing's warmup read `pair.Base` of `exchangeInfo` elements (239 KB) 128,000 times and peaked at 3 GB (OutOfMemory on an 8 GB server), copying each element into its own `Json(value to Text)` first made it cheap.
- WebWars and ArenaWars do not build natively: `Native compilation failed, unsupported: text of List(cell) in WebWarsTerrain_Navigation_0` (`ArenaLevel_Navigation_2` in ArenaWars), both build their navigation with DeltaEngine's `Graphs/NavigationMap.FromGrid` (DeltaEngine 3a4ef02 with Strict 0da462d8, SoulCraft builds).

## Next
- GPU speed and memory: [improvements-gpu.md](improvements-gpu.md); Ternary Bonsai and Qwen3.8-Flash-Next: [improvements-models.md](improvements-models.md).
- Game engine (iOS, Android, one game loop, Spine, platform services): [improvements-engine.md](improvements-engine.md).
- Nightly CI: build and run every program's Run once a day (games and long running apps for a few frames); work too slow for a Slow test (rebuilding all content, big models) belongs in a Run, not in tests.
- Networking: `Http.Send`, `Http.Started`/`Received` and `Hmac.Sha256` are in (Windows, Linux and the web checked, macOS untested), DeltaEngine's Google sign-in and sheet downloads use them: switch Rebalancing's `NativeIO.Http`/`Hmac` (with the user, it trades live); client certificates when a program needs them, TLS for multiplayer sockets later.

## Waiting
- Later: Qwen3-Coder-Next 80B-A3B and Qwen3-Coder-30B with all experts, image blur on the GPU, Apple Silicon, AMD GPUs, batched training ideas.

