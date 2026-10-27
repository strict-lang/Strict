# Strict.Runtime
The Strict VM: `strict <file>` and `strict test` run the instructions of `Bytecode` here without building an executable. `Execute` loads the types, `Resolver` numbers the locals and registers of every method into slots, resolves call targets and jumps and decodes constants, `Machine` calls methods and binds members and arguments, `Frame` runs one call in its own list of slots (a call of a method that only returns a member, like `Vector2.X`, reads it through `Accessor` without a call), `Looping`, `Operations` and `Storage` carry out the instructions, `Natives` the methods of the base types the VM implements itself.

## Test budgets
Automatic test runs select unmarked methods and enforce 100 ms per test (the goal is under 10 ms), including leading local setup; source checking and compilation are outside the budget. `TestBudget` gives the VM and native test runner the same clock and watchdog hooks, a test that does not return is stopped. Methods of the internal packages (Compiler, Runtime, Bytecode, Expressions, `TypeSource.IsSlow`) count as Slow. Explicit file or method runs include Slow tests. Only automatic passing results are cached. `Execute.Run` runs the requested tests before loading the program's Run graph.

VM constants are cached per `Machine`: a getter runs on first use and its result is stored, recursive access reports `RecursiveConstant`. Unused Slow fixtures are never initialized.

`python Runtime/check_tests.py` checks deadlines, Slow selection and fixture exclusion, `python Runtime/check_frames.py` frame initialization speed.
