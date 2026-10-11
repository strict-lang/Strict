# Improvements
In priority order. Remove what is done, add what is found.

1. **Dictionary generics**: generic types need monomorphization in the native compiler, one copy of each generic method per concrete type. `Dictionary.strict` and `Examples/BaseTypesTest` are not compiled natively (`Dictionary.Length` is unsupported).
2. **HashCode**: `from(generic)` needs the same monomorphization (`generic.Length` on a Generic is unsupported). The computation itself is a placeholder; Dictionary lookups need a real hash per type (numbers, texts, lists, instances by their members) to beat a list scan.
3. **Operator overloads by parameter type**: List has `-(other) List` and `-(generic) List` (same for `+`, `*`, `/`). The native compiler only maps `+` to runtime calls, so `list - list`, `*` and `/` crash, and so do `3 + (4)` and the Error for `(1) + ("Hi")`. `List.Reverse` crashes as well: inside List, `Length` resolves to the Length type instead of the method, the mapping `for` does not collect a list, and `outer` is never assigned. `List.strict` crashes natively because of all of these.
4. **Strict VM parity**: VS Code runs `NativeCompiler run test` for every file. Of 216 files, 157 pass, 29 have failing tests and 30 crash, while native `test` passes 211.
   - Crash (exit 139): `Text`, `List`, `Path`, most of Bytecode (FileCompiler, Linker, TestLines, CallTarget, all Codegen files), Expressions (Binary, KnownTypes, PhraseTokenizer, ResolveCheck, ShuntingYard, SyntaxNode, SyntaxParser, TypeShape), Compiler (MlirAutomatic, MlirEquality, MlirKinds, Toolchain) and `Examples/Parsing`. `Number.strict` exits 127.
   - Failing tests: Math (Degrees, Radians, Ratio, Vector2), Runtime (Frame, Locals, Looping, Machine, Operations, Storage), Compiler Mlir (Calls, Constants, Instances, Lists, Module, Texts), Bytecode (BinaryNames, ByteEncoder, SourceFiles, TypeEntry), `Dictionary`, `HashCode`, `Character`, `Variable`, `Examples/BaseTypesTest` and `Examples/RemoveParentheses`.
   - `ImageProcessing/Color`: `Color(1, 0, 0) to ColorValue` gives `(0, 0, 0, 0)`. `Compiler/NativeCompiler`: `File.ReadLines` is an unsupported native call.
   - Logging an instance of a one-member type prints `(C:\Program Files\NASM\nasm.exe)` on the VM (as C# did) but the bare path natively (`Examples/ProcessProbe`, `Process.Find`). README does not say which is right.
5. **Errors instead of crashes**: every crash above prints nothing, not even a RuntimeError with a stack trace back to the `.strict` line.
   - A parse error (e.g. `number + + other` in a method body) crashes the same way (exit 139) instead of printing a ParsingFailed message with file and line.
   - Native `test` reports a missing test executable as `127 failed in Number.strict tool not found: Number.test.exe`. The exit code reads like a failure count and the line looks like a test summary.
6. **Output for VS Code**:
   - `run test`: print one line per test (passed or failed, with its line number and duration) and a total. Then the client no longer guesses passes from missing failure lines, and a crash no longer marks passed tests as failed.
   - Add a folder mode for `run test`: today every file starts its own process, which parses all files and infers all types again.
   - `check` prints automatic instance text `(UnusedMethodParameterMustBeRemoved, other, Never used)` instead of `ValidationIssue.to Text` (`UnusedMethodParameterMustBeRemoved: other`). Issues also need a line number: the client marks the first line containing the target for now. A folder outside root is printed with mixed separators (`C:/.../pkg\Unused.strict`).
7. **Native gaps**:
   - `to Type` is missing (`Variable.strict`).
   - ImageProcessing executables crash when run (Image and ColorValue members).
   - `First` of an empty list crashes instead of giving an Error.
8. **Validators**: a `let` with only constant values must be `constant`. The check misses list literals.
9. **Tooling**: LanguageServer in Strict (the C# one is on the `dotnet` branch), CI with the native toolchain.
10. **Strict VM speed**: 245 ms for 1M iterations, about 5x slower than native (45 ms). Fast enough for now. When needed, update frames and registers in place for values with only one reference.
