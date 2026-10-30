# Improvements
What to fix next, one short line each, bugs first. Remove a line once it is done, add what is found at the end. No history or measurements (those go into commit messages). Speed work: [performance.md](performance.md).

## Bugs
- A pure loop over GPU heavy calls forks onto CPU threads (MlirCost's threshold is 2 for a costly body): `Qwen3Images.Encoded` of 2 images ran its vision towers in workers and did not finish in 4.5 min, sequential takes 16 s more than one image.
- WebWars and ArenaWars do not build natively: `Native compilation failed, unsupported: text of List(cell) in WebWarsTerrain_Navigation_0` (`ArenaLevel_Navigation_2` in ArenaWars), both build their navigation with DeltaEngine's `Graphs/NavigationMap.FromGrid` (DeltaEngine 3a4ef02 with Strict 0da462d8, SoulCraft builds).

## Next
- GPU speed and memory: [improvements-gpu.md](improvements-gpu.md); Ternary Bonsai and Qwen3.8-Flash-Next: [improvements-models.md](improvements-models.md).
- Game engine (iOS, Android, one game loop, Spine, platform services): [improvements-engine.md](improvements-engine.md).
- Nightly CI: build and run every program's Run once a day (games and long running apps for a few frames); work too slow for a Slow test (rebuilding all content, big models) belongs in a Run, not in tests.
- Networking: `Http.Send`, `Http.Started`/`Received` and `Hmac.Sha256` are in (Windows, Linux and the web checked, macOS untested), DeltaEngine's Google sign-in and sheet downloads use them: switch Rebalancing's `NativeIO.Http`/`Hmac` (with the user, it trades live); client certificates when a program needs them, TLS for multiplayer sockets later.

## Waiting
- Later: Qwen3-Coder-Next 80B-A3B and Qwen3-Coder-30B with all experts, image blur on the GPU, Apple Silicon, AMD GPUs, batched training ideas.

