# Handoff: bugs and speed pass (2026-11-13)

Delete this file when the work below is landed and nothing is waiting on it.

## Where things are
- Work happens in the worktree `.claude/worktrees/quality` (branch `quality`), on top of master ce9e1dc8.
- Branch `landing` marks the last fully verified state, `strict.landing.exe` in the quality worktree is its compiler. Land it from the main checkout (PowerShell): `git merge --ff-only landing; .\strict.exe stop; Copy-Item .claude\worktrees\quality\strict.landing.exe strict.exe -Force`.
- `quality` can be ahead of `landing`: `git log --oneline landing..quality`.
- A second Strict session works on Ben-Home-PC (Orca environment `Ben-Home-PC`, workspace `C:/code/GitHub/strict-lang/Strict`, conversation 226c4849, resumed in terminal "Strict BenchBlur"): it took the BenchBlur regression on its branch `benchblur` and waits for Ben's yes on its fix (below). Messages from it do not reach this machine through Orca; read its terminal with `orca terminal read --environment Ben-Home-PC --terminal <handle> --screen`.

## Verifying a state before moving `landing`
In the quality worktree, with `STRICT_GPU=0 STRICT_SERVER=0` (scratchpad `verify.sh` and `verify2.sh` did all of it):
1. Rebuild the compiler until two stages have the same hash: `rm -f obj/Strict.mlir obj/Strict.nativecache obj/Strict.strictbinary; touch Strict.strict; ./strict.exe build Strict.strict windows; sha256sum strict.exe`; stop when a stage log says FAILED. Start from an exe whose frontend is known good: a frontend codegen bug in the building exe miscompiles every later stage (2026-11-13: a Text loop wrapped into Text.from made the compiler run out of memory in ByteEncoder). Then `./strict.exe build Examples/Sum.strict` and `./Examples/Sum.exe 2 3` prints 5.
2. `./strict.exe test <Folder> native` for Compiler Expressions Bytecode Runtime Validators Math Json Examples NeuralNetworks ImageProcessing MachineLearning Networking Compression Xml Binary (all pass); NeuralNetworks again with the GPU on.
3. `python Runtime/check_tests.py`, `python Compiler/check_build.py` and `python Compiler/check_parallel.py` (the last two with STRICT_GPU unset), `./strict.exe check .` with `Probe/` moved aside (13 known issues: 11 StatusNumberMustNotBeReturned in Runtime, 2 MethodAlreadyExists in NeuralNetworks).
4. `python Compiler/check_memory.py`: its web case needs the main checkout's `Tools/emsdk`; in the worktree run a scratch copy without the WebPressure block (the cases after it pass).
5. GPU kernels not lost: build `NeuralNetworks/Attention.strict`, `Gpt2Block.strict`, `Convolution.strict` with the GPU on, `grep -ho "\.entry [A-Za-z0-9_]*" NeuralNetworks/obj/<Name>*.ptx | sort | wc -l` gives 11, 164 (82 unique, an `.older.ptx` too), 55.
6. Sibling repos: `strict check .` in copies of the `.strict`, `.md`, `.c`, `.h` and `.flags` files of StrictComfy, Ben and DeltaEngine (scratchpad `sib/`) with the old and the new exe: 0 validation issues, no new compile errors; failing tests there come from content the copies lack.
7. Then `git branch -f landing quality` and copy `strict.exe` to `strict.landing.exe`.

## Decisions taken by the owner (2026-11-12)
- A prefix `not` belongs to its one operand, no brackets needed in `and`/`or` chains. Syntax highlighting per condition is done in `../strict-vscode-client` (uncommitted there, not seen in a real editor).
- No checks inside loops: flat list reads stay unchecked. Checks belong before the loop or in tests.
- Matrix and vector speed is the top performance goal and a benchmark.
- A native folder test staying up to date after a compiler change is intended.
- One update process for every program with a backend.
- Waiting fixes decided one by one later: the three tightened rules (branch `tightened`: let reassignment, ReturnTypeMismatch, nested ResultMustBeUsed), swapped constructor arguments (worktree-agent-a81bd2b2377be5ecb), the lost-change rule (worktree-agent-adc730c3a8f42dcf9), the issue format (worktree-agent-adca418840c161373).

## Waiting for the owner (2026-11-13)
- BenchBlur (Ben-Home-PC session): 20 times slower since 73f19bf3 (`MlirWholeFunction.HasConstantDivisor`), its average divides by a variable count, so no whole versions and no GPU kernel. Proposed: CPU whole-number functions get the `%failed` flag the GPU versions have (branch-free zero test, the loop reruns its slow path once when set); it also closes the `%` by zero line of bugs.md. Needs a yes because it touches the "no checks inside loops" decision.
- bugs.md lines that need a semantic choice: a Range for a `Numbers` parameter (convert or reject), a call on the own type name (`Ticker.Next`), `%` by zero (above).

## State when this was written (2026-11-13, 04:00)
- `landing` is this commit, `strict.landing.exe` its compiler, verified by the steps above.
- Fixed from bugs.md today: Text index past the end, unspaced constants with " = " in a text, Boolean and invalid Run arguments (exit 2), a step on a non-Number (`StepMustBeOnNumber`), nested list text conversion, `ElementType` of a Couple key, subnormal literals, three WebGPU suspects, `check_sockets.py`, image saves creating folders, `strict build Strict.strict` as release, a Number for a composite parameter, a trailing comma constant crash, a for returning a one-list type, two duplicates (return type read, IsDigit, IsList pairs).
- Matrix speed (minimum of 30-50 alternating runs, timing program `Probe/MatrixSpeed.strict`, untracked): chained product 83 -> 49 ns, `Transformed` 158 -> 86, `Axis` 66 -> 35. Rules: a builder that becomes a flat block starts without a list, a raw collecting loop of a constant count into a flat type writes the flat block (`MlirBytes.Flattened`), an inlined accessor of a parameter no longer calls `strict_rc_dead(0, ...)`. Next ideas are in the Matrix line of performance.md.
- Next planned: a native `List(Number).Sorted` (copy plus `qsort`, `llvm.func` comparator like the Darwin runtime's directory sort; move its `qsort` declaration into the shared runtime), then the small-block free cost with a sampling profile (symbol build needed), then the remaining performance.md items.
- Leftovers: the stale `worktree-agent-*` branches and worktrees named above; `Probe/` in the worktree holds untracked probes.
