# Improvements
Bugs to fix, most harmful first, one short line each. Remove a line once it is done, add what is found. No history or measurements (those go into commit messages). Speed work: [performance.md](performance.md). Anything large or difficult, features and far fetched things: [ideas.md](ideas.md).

- The resident server kept stale sources or folder listings after DeltaEngine edits (files restored with Copy-Item keep their old write time): `strict test Games/TorinsTowers/TowersGame.strict unit` failed with UnresolvedLink on a different existing method each run (Match.Queue, RtsClient.Viewer, LiveCatalog.Validated) until `strict stop`.
- Mutation run of DeltaEngine Graphs/NavigationMap.strict used up 16 GB (OutOfMemory) on a mutant of a doubling loop instead of killing it at a memory budget.
- `outer.Key.Z` inside a `for` body gives MethodNotFound: Character.Z (DeltaEngine Games/RacingGame/RaceSettings.Edited worked around it with a type constant).
- `"abc" to Number` gives 0 instead of an Error; the member limit (15 `has`) is not enforced (DeltaEngine GameData/SheetDownload has 20 members); MagicNumbers skips numbers in methods without tests and in leading lines that look like tests (plain `Label(...)` calls).
- `strict check NeuralNetworks` reports 8 issues: GpuMemory unused stub parameters, Safetensors.Tuned, Qwen3.Forwarded and QwenSafetensors.Blocks over 12 lines, a Safetensors line over 120 characters, QwenSafetensors with 19 methods.

- GPU speed and memory: [improvements-gpu.md](improvements-gpu.md); Ternary Bonsai and Qwen3.8-Flash-Next: [improvements-models.md](improvements-models.md).
- Game engine (iOS, Android, one game loop, Spine, platform services): [improvements-engine.md](improvements-engine.md).
