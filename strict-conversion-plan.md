# Strict Self-Hosting Conversion Plan

Before changing any `.strict` file, build `Strict` and use the Strict executable to parse, compile,
run tests, and execute the changed Strict code.

**Top priority.** The goal is to progressively convert all C# implementation of Strict into `.strict`
files, using the current C# bootstrap (`Strict.exe` on .NET 10) to compile and run the new Strict
implementation. Long term: get rid of the C# bootstrap entirely and have everything in `.strict`.

This file tracks which layers have been converted, what `.strict` files exist, how many tests are
written in Strict, and what C# features are still missing from the Strict runtime.

---

## Architecture Overview (10 phases, bottom to top)

| # | C# Project | Purpose | C# Files | C# Lines | Test Methods |
|---|-----------|---------|----------|----------|--------------|
| 0 | *(Base types)* | Boolean, Number, Text, List, Dictionary, File, etc. | 0 (`.strict` already) | — | in each type |
| 1 | `Strict.Language` | Load `.strict` files, package/type parsing, member resolution | 32 | 4,453 | 335 (173+162) |
| 2 | `Strict.Expressions` | Lazy-parse method bodies into expression trees | 29 | 3,335 | 553 (326+227) |
| 3 | `Strict.Validators` | Static analysis, type checking, constant folding | 3 | 451 | 45 (39+6) |
| 4 | `Strict.TestRunner` | Run in-code tests (`is` assertions) via HighLevelRuntime | 1 | 37 | 20 |
| 5 | `Strict.HighLevelRuntime` | Interpret expressions (for test running & validation) | 11 | 1,508 | 88 (84+4) |
| 6 | `Strict.Bytecode` | Generate register-based bytecode + serialization | 37 | 2,428 | 55 (54+1) |
| 7 | `Strict.Optimizers` | Remove test code, dead stores, constant fold instructions | 9 | 529 | 59 |
| 8 | `Strict` (exe) | VirtualMachine execution, Runner orchestration | 6 | 1,611 | 107 (69+38) |
| 9 | `Strict.Compiler` + `Strict.Compiler.Assembly` | Native code gen (NASM x64, gcc/clang linking) | 5 | 918 | 49 (46+3) |
| — | `Strict.LanguageServer` | IDE support (LSP, hover, autocomplete) — lower priority | 17 | 672 | 2 |
| — | `Strict.Transpiler` (Roslyn) | C#→Strict transpiler (helps bootstrap) — tool | 7 | 375 | 50 (37+13) |

**Total C# to convert:** ~131 production files, ~15,750 lines of code, 907 test methods across 10 test projects.

---

## Phase 0 — Base Types Verification (prerequisite)

All base `.strict` types live at the repo root. They are already in Strict but need thorough
end-to-end testing via the `Examples/BaseTypesTest/` multi-file package.

| Base Type | `.strict` file | Tested in BaseTypesTest | Status |
|-----------|---------------|------------------------|--------|
| `Boolean` | `Boolean.strict` | ☐ | 0% |
| `Number` | `Number.strict` | ☐ | 0% |
| `Text` | `Text.strict` | ☐ | 0% |
| `List` | `List.strict` | ☐ | 0% |
| `Dictionary` | `Dictionary.strict` | ☐ | 0% |
| `File` | `File.strict` | ☐ | 0% |
| `Directory` | `Directory.strict` | ☐ | 0% |
| `Range` | `Range.strict` | ☐ | 0% |
| `Error` / `ErrorWithValue` | `Error.strict`, `ErrorWithValue.strict` | ☐ | 0% |
| `Character` | `Character(strict)` | ☐ | 0% |
| `Any` | `Any(strict)` | ☐ | 0% |
| `Enum` | `Enum(strict)` | ☐ | 0% |
| `Mutable` | `Mutable(strict)` | ☐ | 0% |
| `Iterator` | `Iterator(strict)` | ☐ | 0% |

**Current state:**
- `Examples/BaseTypesTest/` exists with 2 `.strict` files (`BaseTypesTest.strict`, `TextHelper.strict`)
- Tests: `RunBaseTypesTestPackageFromDirectory` in `Strict.Tests` passes
- Still to add: explicit tests for Boolean, Number, Text, List, Dictionary, File, Directory, Range, Error, Character

**Target:** 14+ test methods in `BaseTypesTest.strict`, one per base type, covering all key operations.

---

## Phase 1 — `Strict.Language` → Strict Package

**Goal:** Convert all 32 C# files (4,453 lines) + 19 test files (335 test methods) to `.strict`.

This is the lowest layer and the hardest — it bootstraps itself. The plan is to convert the
simplest, most self-contained classes first and work upward.

### Missing Strict Runtime Features (blockers)

Before Strict.Language can be written in Strict, the runtime needs these capabilities:

| Missing Feature | Used In | Priority |
|----------------|---------|----------|
| `Path.Combine(a, b)` | `Package.cs`, `Repositories.cs` | High |
| `Path.GetFileName(path)` | `Package.cs`, `Runner.cs` | High |
| `Path.GetFileNameWithoutExtension(path)` | `Repositories.cs` | High |
| `Path.GetDirectoryName(path)` | `Repositories.cs` | High |
| `Directory.Exists(path)` | `Repositories.cs` | High |
| `Directory.GetFiles(path, pattern)` | `Repositories.cs` | High |
| `File.ReadAllLines(path)` | `Repositories.cs`, `Runner.cs` | High |
| `string.Split(chars[])` | `SpanExtensions.cs`, `TypeLines.cs` | High |
| `string.StartsWith(prefix)` | `Body.cs`, `TypeLines.cs` | Medium |
| `string.EndsWith(suffix)` | `SpanExtensions.cs` | Medium |
| `string.Contains(substring)` | `TypeLines.cs`, `Context.cs` | Medium |
| `string.Trim()` | `TypeLines.cs` | Medium |
| `Char` / `char` comparisons | `SpanExtensions.cs` | Medium |
| `ReadOnlySpan<char>` / `Span` patterns | `SpanExtensions.cs`, `Body.cs` | Medium |
| Exception types / `throw` / `catch` | Throughout | High |
| `async` / `await` / `Task<T>` | `Repositories.cs` | Low (defer) |
| Reflection / Attributes | `LogAttribute.cs`, test infra | Low (defer) |
| HTTP / GitHub download | `GitHubStrictDownloader.cs` | Lowest (defer) |

### Naming Convention Notes

Just as C# has reserved words and uses workarounds (e.g. `Type.ValueLowercase` for "value",
`using Type = Strict.Language.Type` to avoid `System.Type` conflicts), Strict requires the same
approach: when a direct name conflicts, pick a clear alternative rather than treating it as a blocker.

#### Naming conflict rules in Strict

Strict enforces that a constant member named `X` (where `X` is an existing type) must have type `X`,
not an auto-numbered enum value. This is the same principle as C#'s naming restrictions.

**Solution:** Use a prefix or suffix to disambiguate, exactly like `Type.ValueLowercase`:
- TypeKind enum constants `Boolean`, `Number`, etc. → prefix with `Kind`: `KindBoolean`, `KindNumber`
- Keyword `Mutable` conflicts with the built-in `Mutable` type → rename to `MutableKeyword`
- Keyword names like `Has`, `Constant`, `Let` are already uppercase and don't conflict → use as-is

### Conversion Order for `Strict.Language`

| Priority | C# File | Description | Strict equivalent plan | Status |
|----------|---------|-------------|------------------------|--------|
| 1 | `Keyword.cs` | String constants for keywords | `Language/Keyword.strict` — 9 text constants (MutableKeyword for Mutable conflict) | ✅ 100% |
| 2 | `BinaryOperator.cs` | 16 operator string constants | `Language/BinaryOperator.strict` — 16 text constants, no conflicts | ✅ 100% |
| 3 | `UnaryOperator.cs` | 1 unary operator constant | `Language/UnaryOperator.strict` — `constant Not = "not"` | ✅ 100% |
| 4 | `TypeKind.cs` | Enum: None/Boolean/Number/etc. | `Language/TypeKind.strict` — 12 constants with Kind prefix | ✅ 100% |
| 5 | `Limit.cs` | Size limit constants | `Language/Limit.strict` — 11 numeric constants | ✅ 100% |
| 6 | `TypeLines.cs` | Raw lines of a type file | `Language/TypeLines.strict` — `has typeName Text`, `has lines Texts`, `to Text` | ✅ 100% |
| 7 | `NamedType.cs` | Name + Type pair (abstract base) | `Language/NamedType.strict` — concrete simplification with `elementName`, `typeName`, `isMutable`, `isConstant`, `IsPublic`, `to Text` | ✅ 70% |
| 8 | `NumberExtensions.cs` | Simple number helpers | Methods on Number — needs method body support | 🚧 Deferred |
| 9 | `StringExtensions.cs` | `MakeFirstLetterUppercase`, etc. | Methods on Text — complex C# spans/strings | 🚧 Deferred |
| 10 | `SpanExtensions.cs` | `IsWord`, `IsKeyword`, etc. | Performance-critical span methods | 🚧 Deferred |
| 11 | `Variable.cs` | Variable: name, type, isMutable | `Language/Variable.strict` — `variableName`, `typeName`, `isMutable`, `initialValue`, `to Text` | ✅ 75% |
| 12 | `Parameter.cs` | Method parameter | `Language/Parameter.strict` — `parameterName`, `typeName`, `isMutable`, `defaultValue`, `HasDefault`, `to Text` with mutable prefix | ✅ 75% |
| 13 | `Member.cs` | Type member definition | `Language/Member.strict` — `memberName`, `typeName`, `isMutable`, `isConstant`, `initialValue`, `IsPublic`, `Keyword`, `to Text` with keyword-aware formatting | ✅ 75% |
| 14 | `Expression.cs` | Abstract expression base | `Language/Expression.strict` — expanded simplified metadata (`returnTypeName`, `lineNumber`, `isMutable`) plus parser-safe `IsConstant`/`to` methods | ✅ 50% |
| 15 | `ConcreteExpression.cs` | Concrete expression with type | `Language/ConcreteExpression.strict` — de-duplicated via `has Expression` + `expressionText` with parser-safe methods | ✅ 50% |
| 16 | `ExpressionParser.cs` | Abstract parser interface | `Language/ExpressionParser.strict` — `Parse` classifies expression types, `IsAssignment`/`IsBinaryExpression`/`IsReassignment`/`ExtractReturnExpression`/`ExtractCondition`/`ExtractIterator` helpers | ✅ 60% |
| 17 | `TypeParser.cs` | Parse member/method headers | `Language/TypeParser.strict` — simplified parse signature trait | ✅ 25% |
| 18 | `Method.cs` (partial) | Method definition, no body parse | `Language/Method.strict` — `methodName`, `returnTypeName`, params/public/trait flags, `HasReturnType`, `HasParameters`, `IsRunMethod` | ✅ 60% |
| 19 | `Context.cs` | Base for Package/Type lookup | `Language/Context.strict` — parser-safe lookup surface (`FindType`, `TryGetType`, `GetType`) | ✅ 40% |
| 20 | `Package.cs` | Package = directory of types | `Language/Package.strict` — parser-safe package metadata + lookup/add methods | ✅ 45% |
| 21 | `Type.cs` | Type definition | `Language/Type.strict` — 12 methods: `IsMember`, `IsMethodHeader`, `MemberCount`, `MethodCount`, `MemberKind`, `ExtractAfterKeyword`, `MemberNames`, `MethodHeaders`, `BodyLines`, `MethodName`, `HasReturnType`, `ReturnTypeName`. Verified via Runner/VM pipeline. | ✅ 80% |
| 22 | `Body.cs` | Method body, lazy parse | `Language/Body.strict` — line classification: `IsMethodCallLine`, `IsReturnLine`, `IsIfLine`, `IsForLine`, `IsDeclarationLine`, `IsReassignment`, `IsBinaryExpression`, `IsNotExpression` | ✅ 55% |
| 23 | `Repositories.cs` | Load packages from GitHub/disk | Needs async/HTTP — defer | 🚧 Deferred |
| 24 | `GitHubStrictDownloader.cs` | HTTP download — defer | Needs HTTP client | 🚧 Deferred |

**Naming convention in Strict Language/ files:**
Strict enforces that a member named `x` (where `X` is an existing type) must have type `X`.
This means `has name Text` fails if a `Name` type exists — use a name that either:
- Starts the type's name: `has text Text`, `has number Number` (standard Strict convention)
- Uses a name with no matching type: `has typeName Text`, `has elementName Text`

**Summary of what's done vs what's next:**
- ✅ **5 pure-constant types done** (Phase 1a) — Limit, Keyword, TypeKind, UnaryOperator, BinaryOperator
- ✅ **21 Language types converted in `.strict` form** — TypeLines, NamedType, Parameter, Member, Variable, Expression, ConcreteExpression, ExpressionParser, TypeParser, TypeFinder, Method, Context, Package, Type, Body, Parser + 5 pure constants
- ✅ **29 Expression types in `.strict` form (Phase 2 complete)** — Value, TextExpression, NumberExpression, BooleanExpression, MethodCall, MemberCall, ParameterCall, VariableCall, Binary, Return, IfExpression, ForExpression, Declaration, ListExpression, NotExpression, MutableReassignment, DictionaryExpression, ListCall, Instance, To, TypeComparison, SelectorIf, TypePattern, ValueInstance, ValueListInstance, ValueTypeInstance, ValueDictionaryInstance, PhraseTokenizer, ShuntingYard
- ✅ **TypeFinder.strict** — Shared type registry with `Find`/`Get`/`Has`/`Count`/`FindPlural` methods. Replaces per-type `typeNames` approach; types reference a common TypeFinder instead of each carrying their own type list.
- ✅ **Type.strict has 12 methods** — `IsMember`, `IsMethodHeader`, `MemberCount`, `MethodCount`, `MemberKind`, `ExtractAfterKeyword`, `MemberNames`, `MethodHeaders`, `BodyLines`, `MethodName`, `HasReturnType`, `ReturnTypeName`
- ✅ **VM fixes** — characters.Length works via recursive EvaluateMemberCall + TryGetNativeLength. BinaryGenerator emits LoadVariableToRegister for member calls with instance. Register save/restore for for-loop bodies.
- ✅ **4 end-to-end examples** — ParseHelloLogger (type line classification), ParseExpressions (expression classification + Substring/characters.Length), ParseMethodHeaders (method header parsing + reassignment detection), Parser (minimum type structure parsing surface)
- ✅ **ExpressionParser.strict expanded** — Parse + IsAssignment/IsBinaryExpression/IsReassignment + extract helpers
- ✅ **Body.strict expanded** — IsMethodCallLine/IsReturnLine/IsIfLine/IsForLine/IsDeclarationLine/IsReassignment/IsBinaryExpression/IsNotExpression
- 🚧 **Known PhraseTokenizer limitation** — `IndexOf("(")` fails in VM/bytecode path because PhraseTokenizer interprets `(` as expression grouping. Works fine in C# HighLevelRuntime expression parsing. Workaround: use space-based parsing for method headers in VM examples.
- 🚧 **Deferred from Phase 1** — Number/String/Span extension parity plus Repositories and GitHub downloader (deferred by design)
- 🚧 **Operator precedence note** — `is` has lowest precedence (1), `and` is 6, so `A is false and B is false` parses as `A is (false and B is false)`. Use parenthesized `(not A) and (not B)` or helper methods instead.

**Target metrics for Phase 1:**
- `.strict` files to generate: ~23 (excluding deferred files)
- Test methods to write: ~335 (matching existing C# test count)
- Estimated Strict LOC: ~3,000–4,000

**Progress table:**

| Metric | Target | Actual | % |
|--------|--------|--------|---|
| `.strict` files created | 23 | 23 | 100% |
| Test methods written | 335 | 36 | 11% |
| C# files replaced | 32 | 0 | 0% |

---

## Phase 2 — `Strict.Expressions` → Strict Package

**Goal:** Convert 29 C# files (3,335 lines) + 25 test files (553 test methods) to `.strict`.

Depends on Phase 1 (needs Type, Method, Body from Strict.Language).

### Key Expressions to Convert

| C# File | Description | Complexity | Status |
|---------|-------------|------------|--------|
| `Value.cs` | Literal values (numbers, text, booleans) | Low | 0% |
| `ValueInstance.cs` | Runtime value wrapper | Medium | 0% |
| `ValueListInstance.cs` | List value at runtime | Medium | 0% |
| `ValueTypeInstance.cs` | Struct-like value instance | Medium | 0% |
| `ValueDictionaryInstance.cs` | Dictionary value at runtime | Medium | 0% |
| `VariableCall.cs` | Reference to a declared variable | Low | 0% |
| `ParameterCall.cs` | Reference to a method parameter | Low | 0% |
| `MemberCall.cs` | `instance.member` access | Medium | 0% |
| `MethodCall.cs` | `instance.Method(args)` call | Medium | 0% |
| `Binary.cs` | `left op right` binary operations | High | 0% |
| `Not.cs` | `not expression` unary | Low | 0% |
| `Boolean.cs` | Boolean literal expression | Low | 0% |
| `Number.cs` | Number literal expression | Low | 0% |
| `Text.cs` | Text literal expression | Low | 0% |
| `List.cs` | List literal / collection expression | Medium | 0% |
| `Dictionary.cs` | Dictionary literal expression | Medium | 0% |
| `ListCall.cs` | `list(index)` access | Medium | 0% |
| `Declaration.cs` | `constant/let/mutable name = value` | Medium | 0% |
| `MutableReassignment.cs` | `name = newValue` | Medium | 0% |
| `If.cs` | `if condition then/else` | Medium | 0% |
| `SelectorIf.cs` | `value is X then Y else Z` | Medium | 0% |
| `For.cs` | `for collection/range` loop | High | 0% |
| `Return.cs` | Explicit `return value` | Low | 0% |
| `To.cs` | `value to Type` conversion | Medium | 0% |
| `TypeComparison.cs` | `value is Type` check | Low | 0% |
| `Instance.cs` | `value` self-reference | Low | 0% |
| `PhraseTokenizer.cs` | Tokenize expression text | High | 0% |
| `ShuntingYard.cs` | Operator precedence parsing | High | 0% |
| `MethodExpressionParser.cs` | Full expression parser (500+ LOC) | Very High | 0% |

**Progress table:**

| Metric | Target | Actual | % |
|--------|--------|--------|---|
| `.strict` files created | 29 | 29 | 100% |
| Test methods written | 553 | 0 | 0% |
| C# files replaced | 29 | 0 | 0% |

---

## Phase 3 — `Strict.Validators` → Strict Package

**Goal:** Convert 3 C# files (451 lines) + 3 test files (45 test methods) to `.strict`.

Depends on Phases 1 & 2.

| C# File | Description | Status |
|---------|-------------|--------|
| `Visitor.cs` | Abstract visitor base | 0% |
| `TypeValidator.cs` | Check unused expressions, type errors | 0% |
| `ConstantCollapser.cs` | Collapse constant expressions | 0% |

**Progress table:**

| Metric | Target | Actual | % |
|--------|--------|--------|---|
| `.strict` files created | 3 | 0 | 0% |
| Test methods written | 45 | 0 | 0% |
| C# files replaced | 3 | 0 | 0% |

---

## Phase 4 — `Strict.TestRunner` → Strict Package

**Goal:** Convert 1 C# file (37 lines) + 2 test files (20 test methods) to `.strict`.

Depends on Phases 1, 2, 3 (needs HighLevelRuntime internally).

| C# File | Description | Status |
|---------|-------------|--------|
| `TestInterpreter.cs` | Run `is` assertions in method bodies | 0% |

**Progress table:**

| Metric | Target | Actual | % |
|--------|--------|--------|---|
| `.strict` files created | 1 | 0 | 0% |
| Test methods written | 20 | 0 | 0% |
| C# files replaced | 1 | 0 | 0% |

---

## Phase 5 — `Strict.HighLevelRuntime` → Strict Package

**Goal:** Convert 11 C# files (1,508 lines) + 7 test files (88 test methods) to `.strict`.

This is the tree-walking interpreter used for test execution and validation.

| C# File | Description | Complexity | Status |
|---------|-------------|------------|--------|
| `Statistics.cs` | Counters for test run metrics | Low | 0% |
| `TestBehavior.cs` | Enum: RunTests / SkipTests | Low | 0% |
| `ExecutionFailed.cs` | Exception wrapper types | Low | 0% |
| `ExecutionContext.cs` | Variable scope / call frame | Medium | 0% |
| `ToEvaluator.cs` | Evaluate `to Type` conversions | Medium | 0% |
| `SelectorIfEvaluator.cs` | Evaluate `value is X then Y` | Medium | 0% |
| `IfEvaluator.cs` | Evaluate `if condition` branches | Medium | 0% |
| `ForEvaluator.cs` | Evaluate `for collection` loops | High | 0% |
| `MethodCallEvaluator.cs` | Dispatch method calls | High | 0% |
| `BodyEvaluator.cs` | Evaluate all expressions in a body | High | 0% |
| `Interpreter.cs` | Top-level interpreter entry point | High | 0% |

**Progress table:**

| Metric | Target | Actual | % |
|--------|--------|--------|---|
| `.strict` files created | 11 | 0 | 0% |
| Test methods written | 88 | 0 | 0% |
| C# files replaced | 11 | 0 | 0% |

---

## Phase 6 — `Strict.Bytecode` → Strict Package

**Goal:** Convert 37 C# files (2,428 lines) + 5 test files (55 test methods) to `.strict`.

Includes the instruction set, bytecode generator, and serializer.

### Sub-layers

#### Instructions (24 files)

| C# File | Description | Status |
|---------|-------------|--------|
| `Instruction.cs` | Abstract base instruction | 0% |
| `RegisterInstruction.cs` | Instruction with a register | 0% |
| `InstanceInstruction.cs` | Instruction with instance | 0% |
| `SetInstruction.cs` | Load literal into register | 0% |
| `LoadConstantInstruction.cs` | Load named constant | 0% |
| `LoadVariableToRegister.cs` | Load variable into register | 0% |
| `StoreVariableInstruction.cs` | Store value as variable | 0% |
| `StoreFromRegisterInstruction.cs` | Store register into variable | 0% |
| `BinaryInstruction.cs` | Binary operation (add, mul, etc.) | 0% |
| `Invoke.cs` | Method invocation | 0% |
| `PrintInstruction.cs` | Output to console | 0% |
| `ReturnInstruction.cs` | Return from method | 0% |
| `Jump.cs` / `JumpIf.cs` / `JumpIfTrue.cs` / `JumpIfFalse.cs` | Conditional/unconditional jumps | 0% |
| `JumpIfNotZero.cs` / `JumpToId.cs` | More jump variants | 0% |
| `LoopBeginInstruction.cs` / `IterationEnd.cs` | Loop control | 0% |
| `ListCallInstruction.cs` | List index access | 0% |
| `WriteToListInstruction.cs` / `WriteToTableInstruction.cs` | Mutation | 0% |
| `RemoveInstruction.cs` | Remove from list | 0% |

#### Generator & Serialization (13 files)

| C# File | Description | Status |
|---------|-------------|--------|
| `Register.cs` | Register enum (R0–R7) | 0% |
| `Registry.cs` | Register allocator | 0% |
| `InstructionType.cs` | Instruction type enum | 0% |
| `InvokedMethod.cs` / `InstanceInvokedMethod.cs` | Method call wrappers | 0% |
| `BytecodeGenerator.cs` | Expression → instructions (700+ LOC) | 0% |
| `BytecodeDecompiler.cs` | Bytecode → partial .strict source | 0% |
| `Serialization/ExpressionKind.cs` | Enum for expression serialization | 0% |
| `Serialization/ValueKind.cs` | Enum for value serialization | 0% |
| `Serialization/NameTable.cs` | String table for bytecode | 0% |
| `Serialization/TypeBytecodeData.cs` | Type + method bytecode bundle | 0% |
| `Serialization/BytecodeSerializer.cs` | Write `.strictbinary` ZIP files | 0% |
| `Serialization/BytecodeDeserializer.cs` | Read `.strictbinary` ZIP files | 0% |

**Progress table:**

| Metric | Target | Actual | % |
|--------|--------|--------|---|
| `.strict` files created | 37 | 0 | 0% |
| Test methods written | 55 | 0 | 0% |
| C# files replaced | 37 | 0 | 0% |

---

## Phase 7 — `Strict.Optimizers` → Strict Package

**Goal:** Convert 9 C# files (529 lines) + 9 test files (59 test methods) to `.strict`.

| C# File | Description | Status |
|---------|-------------|--------|
| `InstructionOptimizer.cs` | Abstract base + optimizer chain | 0% |
| `TestCodeRemover.cs` | Remove test-only instructions | 0% |
| `ConstantFoldingOptimizer.cs` | Fold constant binary ops | 0% |
| `StrengthReducer.cs` | Replace expensive ops with cheaper | 0% |
| `DeadStoreEliminator.cs` | Remove never-loaded stores | 0% |
| `RedundantLoadEliminator.cs` | Remove duplicate loads | 0% |
| `JumpThreadingOptimizer.cs` | Simplify redundant jumps | 0% |
| `UnreachableCodeEliminator.cs` | Remove code after unconditional jumps | 0% |
| `AllInstructionOptimizers.cs` | Compose all optimizers in order | 0% |

**Progress table:**

| Metric | Target | Actual | % |
|--------|--------|--------|---|
| `.strict` files created | 9 | 0 | 0% |
| Test methods written | 59 | 0 | 0% |
| C# files replaced | 9 | 0 | 0% |

---

## Phase 8 — `Strict` (VirtualMachine + Runner) → Strict Package

**Goal:** Convert 6 C# files (1,611 lines) + 8 test files (107 test methods) to `.strict`.

This is the execution engine — the capstone of the self-hosting effort.

| C# File | Description | Status |
|---------|-------------|--------|
| `RegisterFile.cs` | Fixed-size register array | 0% |
| `CallFrame.cs` | Variable scope per method call | 0% |
| `Memory.cs` | Registers + frame per VM | 0% |
| `VirtualMachine.cs` | Execute bytecode instructions (750+ LOC) | 0% |
| `Runner.cs` | Orchestrate parse→validate→compile→run (570+ LOC) | 0% |
| `Program.cs` | CLI entry point — keep in C# or convert last | 0% |

**Progress table:**

| Metric | Target | Actual | % |
|--------|--------|--------|---|
| `.strict` files created | 6 | 0 | 0% |
| Test methods written | 107 | 0 | 0% |
| C# files replaced | 6 | 0 | 0% |

---

## Phase 9 — `Strict.Compiler` + `Strict.Compiler.Assembly` → Strict Package

**Goal:** Convert 5 C# files (918 lines) + 1 test file (49 test methods) to `.strict`.

| C# File | Description | Status |
|---------|-------------|--------|
| `Strict.Compiler/Platform.cs` | Enum: Windows/Linux/MacOS | 0% |
| `Strict.Compiler/ToolNotFoundException.cs` | Exception for missing NASM/gcc | 0% |
| `Strict.Compiler/InstructionsCompiler.cs` | Abstract compiler interface | 0% |
| `Strict.Compiler.Assembly/InstructionsToAssembly.cs` | Bytecode → NASM x64 assembly (900+ LOC) | 0% |
| `Strict.Compiler.Assembly/NativeExecutableLinker.cs` | Invoke NASM + gcc/clang | 0% |

**Progress table:**

| Metric | Target | Actual | % |
|--------|--------|--------|---|
| `.strict` files created | 5 | 0 | 0% |
| Test methods written | 49 | 0 | 0% |
| C# files replaced | 5 | 0 | 0% |

---

## Overall Progress Dashboard

| Phase | Project | C# Files | Target `.strict` Files | Actual `.strict` Files | Tests Written | C# % Done |
|-------|---------|----------|------------------------|------------------------|---------------|-----------|
| 0 | Base Types (verification) | 0 | 0 (already `.strict`) | 2 (BaseTypesTest) | 1 | 0% |
| 1 | `Strict.Language` | 32 | 22 | 20 (Limit, Keyword, TypeKind, UnaryOperator, BinaryOperator, TypeLines, NamedType, Parameter, Member, Variable, Expression, ConcreteExpression, ExpressionParser, TypeParser, TypeFinder, Method, Context, Package, Type, Body) | 28 | 27% |
| 2 | `Strict.Expressions` | 29 | 29 | 29 (all expression types + Value/Tokenizer/ShuntingYard) | 0 | 10% |
| 3 | `Strict.Validators` | 3 | 3 | 0 | 0 | 0% |
| 4 | `Strict.TestRunner` | 1 | 1 | 0 | 0 | 0% |
| 5 | `Strict.HighLevelRuntime` | 11 | 11 | 0 | 0 | 0% |
| 6 | `Strict.Bytecode` | 37 | 37 | 0 | 0 | 0% |
| 7 | `Strict.Optimizers` | 9 | 9 | 0 | 0 | 0% |
| 8 | `Strict` (VM + Runner) | 6 | 6 | 0 | 0 | 0% |
| 9 | `Strict.Compiler(.Assembly)` | 5 | 5 | 0 | 0 | 0% |
| **Total** | | **133** | **123** | **51** (2 BaseTypesTest + 20 Language + 29 Expressions) | **28** | **12%** |

---

## Missing Runtime Features Tracker

These C# / .NET features need to be added to the Strict runtime before each phase can proceed.

| Feature | Needed For Phase | Priority | Status |
|---------|-----------------|----------|--------|
| `Path.Combine` | 1 (Language) | 🔴 Critical | ✅ Added (`Path.+`) |
| `Path.GetFileName` | 1 (Language) | 🔴 Critical | ✅ Added (`Path.FileName`) |
| `Path.GetFileNameWithoutExtension` | 1 (Language) | 🔴 Critical | ✅ Added (`Path.RemoveExtension`) |
| `Path.GetDirectoryName` | 1 (Language) | 🔴 Critical | ✅ Added (`Path.PathOnly`) |
| `Path.ChangeExtension` | 1 (Language) | 🟠 High | ✅ Added (`Path.ChangeExtension`) |
| `Directory.Exists` | 1 (Language) | 🔴 Critical | ✅ Added |
| `Directory.GetFiles(path, pattern)` | 1 (Language) | 🔴 Critical | ✅ Added (`Directory.Files`) |
| `Directory.CreateDirectory` | 1 (Language) | 🟠 High | ✅ Added (`Directory.Create`) |
| `File.ReadAllLines` | 1 (Language) | 🔴 Critical | ❌ Missing |
| `File.WriteAllText` | 1 (Language) | 🟠 High | ✅ Covered by `File.Write` |
| `File.Exists` | 1 (Language) | 🟠 High | ✅ Added |
| `Text.Split(separator)` | 1 (Language) | 🔴 Critical | ✅ Added |
| `Text.Trim()` / `TrimStart()` / `TrimEnd()` | 1 (Language) | 🟠 High | ✅ Added (`Trim`) |
| `Text.IndexOf(substring)` | 1 (Language) | 🟠 High | ✅ Added |
| `Text.LastIndexOf(substring)` | 1 (Language) | 🟠 High | ✅ Added |
| `Text.Substring(start, length)` | 1 (Language) | 🟠 High | ✅ Added |
| `Text.Replace(old, new)` | 2 (Expressions) | 🟡 Medium | ✅ Added |
| `Text.ToUpper()` / `ToLower()` | 2 (Expressions) | 🟡 Medium | ✅ Added (`Upper`/`Lower`, delegated to `Character`) |
| `Char` / `char` comparisons & casing support | 1 (Language) | 🟠 High | ✅ Added (`Character.Upper`/`Lower` + Text iteration over Character) |
| Exception handling (`throw`/`catch`) | 1+ | 🔴 Critical | ➖ Not needed (`Error` type) |
| `async`/`await` / `Task<T>` | 1 (Repositories) | 🟡 Defer | ⏸ Deferred |
| HTTP client / web download | 1 (GitHub download) | 🟢 Defer | ⏸ Deferred |
| Reflection / Attributes | Test infra | 🟢 Defer | ⏸ Deferred |
| `ZipArchive` / ZIP handling | 6 (Bytecode serial.) | 🟡 Medium | ⏸ Deferred |
| Binary I/O (`BinaryReader`/`BinaryWriter`) | 6 (Bytecode serial.) | 🟡 Medium | ⏸ Deferred |
| Process execution (`Process.Start`) | 9 (Compiler) | 🟡 Medium | ⏸ Deferred |

---

## Rules for Conversion

1. **TDD always**: Write the failing `.strict` test first, then implement.
2. **All tests from the C# `.Tests` project must be ported** to equivalent Strict inline tests (`is` assertions in methods).
3. **Strict limits apply**: No method longer than ~50 lines, no type longer than ~400 lines. Split aggressively.
4. **No duplication**: If logic exists in a base type or lower-layer type, call it — don't copy it.
5. **Only what is called is included** in the final bytecode (tree-shaking by default).
6. **Start with the simplest files** (constants, enums, small data types) before tackling parsers/VMs.
7. **Deferred items** (async, HTTP, reflection) will remain in C# thin wrappers until the runtime supports them.
8. **Update this file** after each new `.strict` file is created or each C# file is replaced.

---

## How to Run the Current Baseline

```bash
# Run all current C# tests
dotnet test Strict.Tests/Strict.Tests.csproj

# Run the multi-file package BaseTypesTest example
dotnet run --project Strict/Strict.csproj -- Examples/BaseTypesTest

# Run a single .strict file
dotnet run --project Strict/Strict.csproj -- Examples/SimpleCalculator.strict
