# Handoff: quality and speed pass (2026-11-12)

Delete this file when the work below is landed.

## Where things are
- Work happens in the worktree `.claude/worktrees/quality` (branch `quality`). `master` in the main checkout is still 50a0b375 with the old `strict.exe`.
- Branch `landing` marks the last fully verified state, `strict.landing.exe` in the quality worktree is its compiler. Land it from the main checkout (PowerShell): `git merge --ff-only landing; .\strict.exe stop; Copy-Item .claude\worktrees\quality\strict.landing.exe strict.exe -Force`.
- `quality` can be ahead of `landing` with picked but not yet verified commits: `git log --oneline landing..quality`.
- Lists: `bugs.md`, `performance.md`, `ideas.md` on `quality` are current.

## Verifying a state before moving `landing`
In the quality worktree, with `STRICT_GPU=0 STRICT_SERVER=0`:
1. Rebuild the compiler until two stages have the same hash: `rm -f obj/Strict.mlir obj/Strict.nativecache obj/Strict.strictbinary; touch Strict.strict; ./strict.exe build Strict.strict windows; sha256sum strict.exe`, then `./strict.exe build Examples/Sum.strict` and `./Examples/Sum.exe 2 3` prints 5.
2. `./strict.exe test <Folder> native` for Compiler Expressions Bytecode Runtime Validators Math Json Examples NeuralNetworks ImageProcessing MachineLearning Networking Compression Xml Binary (all pass).
3. `python Runtime/check_tests.py`, `python Compiler/check_build.py` and `python Compiler/check_parallel.py` (the last two with STRICT_GPU unset), `./strict.exe check .` (13 known issues: 11 StatusNumberMustNotBeReturned in Runtime, 2 MethodAlreadyExists in NeuralNetworks).
4. `python Compiler/check_memory.py` (its web case needs the complete `Tools/emsdk` of the main checkout; every other case passes in the worktree).
5. GPU kernels not lost: build `NeuralNetworks/Attention.strict`, `Gpt2Block.strict`, `Convolution.strict` with the GPU on and compare `grep -ho "\.entry [A-Za-z0-9_]*" NeuralNetworks/obj/<Name>*.ptx | sort` with the previous compiler (11, 164, 55 entries).
6. Sibling repos: `strict check` and one native folder test build of Ben, StrictComfy and DeltaEngine folders with the old and the new compiler (delete `obj/Tests.test.exe` first, a native folder test stays up to date after a compiler change). A build failure inside the compiler now prints its Error and stack trace.
7. Then `git branch -f landing quality` and copy `strict.exe` to `strict.landing.exe`.

## Decisions taken by the owner (2026-11-12)
- A prefix `not` belongs to its one operand, no brackets needed in `and`/`or` chains (README says so; the looser precedence branch is dropped). Wanted too: syntax highlighting in `../strict-vscode-plugin` with different colours per condition of a long `if` (`and`/`or` one colour, `not` another, reddish).
- No checks inside loops: flat list reads stay unchecked (the range check branch is dropped). Checks belong before the loop (first iterations checked or a precheck of first and last index) or in tests.
- Matrix and vector speed is the top performance goal and a benchmark: unrolling, integer index math, folding operations, no wrapper traces or allocations.
- A native folder test staying up to date after a compiler change is intended; clean full runs belong to CI and nightly runs.
- One update process for every program with a backend (see the bugs.md line about the failed replace of strict.exe): up to date means say so and stop; outdated means stop the running process, build, overwrite, resume; strict itself through `strict.new.exe` and an updater step.
- Older waiting fixes are decided one by one later: the three tightened rules (branch `tightened`), swapped constructor arguments (0c0d70e6, branch worktree-agent-a81bd2b2377be5ecb, Ben has 9 such lines), the lost-change rule (2642707b, branch worktree-agent-adc730c3a8f42dcf9, DeltaEngine Games/GhostWars/GhostAi.strict:29), the issue format (33b443d2, branch worktree-agent-adca418840c161373, the VS Code plugin pattern has to change with it).

## In progress when this was written
See the end of this file (updated last).
- Matrix speed next rules (integer index math in raw loops, inlining across parts, unrolling small constant loops, the `*` wrapper, list built in the flat block): an agent on branch worktree-agent-ab56b2cc88b1eb466.
- GPU device suspects (seven, bugs.md lines about MlirDeviceSums, MlirDeviceLoops, MlirDeviceRows, MlirDeviceStatements, MlirDots, MlirWholeKernel.Free): an agent on branch worktree-agent-a4f0dfbc738c6f8be.
- Update process for executables: an agent on branch worktree-agent-a5ccd2d6b92d227c0.
- Not started: the VS Code highlighting, the clear older bugs.md lines (dead code lists in Bytecode, Expressions, NeuralNetworks; `Networking/check_sockets.py`; a Text index past the end; `%` by zero in comparisons).
