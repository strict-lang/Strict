# Improvements
Bugs to fix, most harmful first, one short line each. Remove a line once it is done, add what is found. No history or measurements (those go into commit messages). Speed work: [performance.md](performance.md). Anything large or difficult, features and far fetched things: [ideas.md](ideas.md).

- Ben-Home-PC: two greedy runs of Qwen3-Coder-30B-A3B on the same prompt give different outputs (Ben, not investigated: atomics or reduction order in the expert path?).
- Ben-Home-PC: once in 8 Qwen3-Coder 30B runs, right after loading, a full collection followed another (19 in a row, about 40 ms each, nothing freed) until GarbageCollectionExceedsLimit: probably the own block path of strict_gpu_obtain, whose pause after a collection is skipped when it freed one reusable block.
- The resident server kept stale sources or folder listings after DeltaEngine edits (files restored with Copy-Item keep their old write time): `strict test Games/TorinsTowers/TowersGame.strict unit` failed with UnresolvedLink on a different existing method each run (Match.Queue, RtsClient.Viewer, LiveCatalog.Validated) until `strict stop`.
- Mutation run of DeltaEngine Graphs/NavigationMap.strict used up 16 GB (OutOfMemory) on a mutant of a doubling loop instead of killing it at a memory budget.
- `outer.Key.Z` inside a `for` body gives MethodNotFound: Character.Z (DeltaEngine Games/RacingGame/RaceSettings.Edited worked around it with a type constant).

- GPU speed and memory: [improvements-gpu.md](improvements-gpu.md); Ternary Bonsai and Qwen3.8-Flash-Next: [improvements-models.md](improvements-models.md).
- Game engine (iOS, Android, one game loop, Spine, platform services): [improvements-engine.md](improvements-engine.md).
