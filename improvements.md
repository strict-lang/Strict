# Improvements
In priority order. Remove what is done, add what is found.

1. **Dictionary generics**: generic types need monomorphization in the native compiler, one copy of each generic method per concrete type. `Dictionary.strict` and `Examples/BaseTypesTest` are not compiled natively (`Dictionary.Length` is unsupported).
2. **HashCode**: `from(generic)` needs the same monomorphization (`generic.Length` on a Generic is unsupported). The computation itself is a placeholder; Dictionary lookups need a real hash per type (numbers, texts, lists, instances by their members) to beat a list scan.
3. **Operator overloads by parameter type**: List has `-(other) List` and `-(generic) List` (same for `+`, `*`, `/`). The native compiler only maps `+` to runtime calls, so `list - list`, `*` and `/` crash, and so do `3 + (4)` and the Error for `(1) + ("Hi")`. `List.Reverse` crashes as well: inside List, `Length` resolves to the Length type instead of the method, the mapping `for` does not collect a list, and `outer` is never assigned. `List.strict` crashes natively because of all of these.
4. **Strict VM parity**: VS Code runs `NativeCompiler run test` for every file. That passes 241 files, while native `test` passes 296. All Expressions and Language files crash on the VM. ImageProcessing tests fail there (AdjustBrightness gets -0.25), and so do several Runtime, Compiler and Validators tests.
5. **Test results for VS Code**: print one line per test (passed or failed, with its line number) and a total. Then the client no longer guesses passes from missing failure lines, and a crash no longer marks passed tests as failed. Also add a folder mode: today every file starts its own process, which parses all files and infers all types again.
6. **Old packages**: `HighLevelRuntime`, `TestRunner` and `Optimizers` are superseded by Expressions, Bytecode (FileCompiler, TestCodegen, ConstantFolding, LoadReuse, Linker) and Runtime. Nothing references them, but VS Code compiles and tests them on every discovery.
   - `HighLevelRuntime` evaluates single text lines, and `3 + 4 is 7` gives an Error.
   - `TestRunner` compares the text on both sides of `is`.
   - `Optimizers` works on the old string-typed BytecodeInstruction, which FileCompiler never produces.

   Redo two of their ideas first, then delete them:
   - Drop test-only constants from production code: `Dictionary.get` builds its `abbreviations` Dictionary on every call.
   - Propagate method-local constants.

   `Language` (the old line-level type model, which still parses `mutable` parameters) needs the same check.
7. **Native gaps**:
   - `to Type` is missing (Variable, Language Member and Type).
   - ImageProcessing executables crash when run (Image and ColorValue members).
   - `First` of an empty list crashes instead of giving an Error.
8. **Validators**: a `let` with only constant values must be `constant`. The check misses list literals.
9. **Tooling**: LanguageServer in Strict (the C# one is on the `dotnet` branch), CI with the native toolchain.
10. **Strict VM speed**: 245 ms for 1M iterations, about 5x slower than native (45 ms). Fast enough for now. When needed, update frames and registers in place for values with only one reference.
