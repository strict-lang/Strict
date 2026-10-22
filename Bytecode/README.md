# Strict.Bytecode
Compiles Strict types into register based instructions, shared by the Strict VM (`Runtime`) and the native compiler (`Compiler`). `FileCompiler` compiles a file: `TypeCodegen` one type, `MethodCodegen`, `ValueCodegen` and friends the statements and expressions, `TestCodegen` the tests at the start of every method, `ConstantFolding` inlines constants. `Linker` adds every used type with only the methods that are called (`Automatic` writes `Members` and `to HashCode` for a type when they are used), `BinaryFile` reads and writes the `.strictbinary` cache next to each file.

Inline numeric assertions use absolute plus relative tolerance `1e-7 * (1 + max(abs(actual), abs(expected)))`. Step-constrained numbers compare exactly after their normal step rounding, including record fields. Boolean, text and production comparisons keep their existing behavior.

<!-- test comparisons expand records and lists through eight nested types; deeper or recursive types use existing exact equality. A recursive runtime comparison helper would remove this compiler expansion limit. -->
