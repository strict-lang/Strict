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
| 1 | `Keyword.cs` | String constants for keywords | `Language/Keyword.strict` | ✅ 100% |
| 2 | `BinaryOperator.cs` | 16 operator string constants | `Language/BinaryOperator.strict` | ✅ 100% |
| 3 | `UnaryOperator.cs` | 1 unary operator constant | `Language/UnaryOperator.strict` | ✅ 100% |
| 4 | `TypeKind.cs` | Enum: None/Boolean/Number/etc. | `Language/TypeKind.strict` | ✅ 100% |
| 5 | `Limit.cs` | Size limit constants | `Language/Limit.strict` | ✅ 100% |
| 6 | `TypeLines.cs` | Raw lines of a type file | `Language/TypeLines.strict` | ✅ 100% |
| 7 | `NamedType.cs` | Name + Type pair | `Language/NamedType.strict` | ✅ 70% |
| 8 | `NumberExtensions.cs` | Simple number helpers | Methods on Number | 🚧 Deferred |
| 9 | `StringExtensions.cs` | String helpers | Methods on Text | 🚧 Deferred |
| 10 | `SpanExtensions.cs` | Span helpers | Performance-critical | 🚧 Deferred |
| 11 | `Variable.cs` | Variable | `Language/Variable.strict` + root `Variable.strict` | ✅ 75% |
| 12 | `Parameter.cs` | Method parameter | `Language/Parameter.strict` | ✅ 75% |
| 13 | `Member.cs` | Type member definition | `Language/Member.strict` — `Parse`, kind/name/type extract | ✅ 80% |
| 14 | `Expression.cs` | Expression base | `Language/Expression.strict` | ✅ 50% |
| 15 | `ConcreteExpression.cs` | Concrete expression | `Language/ConcreteExpression.strict` | ✅ 50% |
| 16 | `ExpressionParser.cs` | Parser interface | `Language/ExpressionParser.strict` — assignment/compare/reassign classifiers | ✅ 55% |
| 17 | `TypeParser.cs` | Parse member/method headers | Split across `Type.strict` + `MethodParser.strict` | ✅ 50% |
| 18 | `Method.cs` (partial) | Method definition | Root `Method.strict` data + `Language/MethodParser.strict` | ✅ 70% |
| 19 | `Context.cs` | Package/Type lookup base | `Language/Context.strict` | ✅ 40% |
| 20 | `Package.cs` | Package = directory of types | `Language/Package.strict` | ✅ 45% |
| 21 | `Type.cs` | Type definition | `Language/Type.strict` — Members/Methods/line classifiers; HLR tests green | ✅ 80% |
| 22 | `Body.cs` | Method body | `Language/Body.strict` — ExpressionKind classification | ✅ 60% |
| 23 | `Repositories.cs` | Load packages | Needs async/HTTP | 🚧 Deferred |
| 24 | `GitHubStrictDownloader.cs` | HTTP download | Needs HTTP client | 🚧 Deferred |
| — | *(driver)* | File → Type dump | `Language/Parser.strict` — VM file read works | ✅ 40% |

**Naming convention in Strict Language/ files:**
Strict enforces that a member named `x` (where `X` is an existing type) must have type `X`.
This means `has name Text` fails if a `Name` type exists — use a name that either:
- Starts the type's name: `has text Text`, `has number Number` (standard Strict convention)
- Uses a name with no matching type: `has typeName Text`, `has elementName Text`

**Summary of what's done vs what's next:**
- ✅ **5 pure-constant types done** (Phase 1a) — Limit, Keyword, TypeKind, UnaryOperator, BinaryOperator
- ✅ **Language package `.strict` files** — TypeLines, NamedType, Parameter, Member, Variable, Expression, ConcreteExpression, ExpressionParser, TypeParser, TypeFinder, MethodParser, Context, Package, Type, Body, Parser + constants. Root `Method.strict` is data-only (`Name`/`Type`/`Parameters`); parsing lives in `MethodParser.strict`.
- ✅ **Object-model cleanup** — Language types use `Name`/`Type` (not legacy `elementName`/`typeName`/`expressionText`). Guarded by `StrictLanguageConversionTests` (11 tests).
- ✅ **Type.strict** — real member/method line parse under **HighLevelRuntime** (inline tests green). `Members`/`Methods` + `MethodParser.Parse` for headers/params/body span.
- ✅ **MethodParser.strict** — `Parse` / `ParseBody` / parameter extraction; avoids `IndexOf("(")` via `OpenParen`/`CloseParen` constants + character scan.
- ✅ **Parser.Run** — reads a real file via `File(path).ReadLines` under the **VM** (Path CLI args work after VM `File.from` Path fix). Logs path + ok when non-empty.
- ✅ **29 Expression types in `.strict` form (scaffold)** — files exist; many still stringly; package load of `Expressions/` alone is fragile (e.g. `Value` → `Expression` cross-package).
- 🚧 **VM gaps (do not block HLR TDD)** — `Type` method invokes can stack-overflow under VM; `BinaryGenerator` still stringifies some member chains (`file.TextReader`); prefer HLR tests for Language library logic.
- 🚧 **Known PhraseTokenizer limitation** — bare `IndexOf("(")` fails (paren as grouping). Workaround: constants + character loops (see MethodParser).
- 🚧 **Deferred from Phase 1** — Number/String/Span extension parity plus Repositories and GitHub downloader (deferred by design)
- 🚧 **Operator precedence note** — `is` has lowest precedence (1), `and` is 6, so `A is false and B is false` parses as `A is (false and B is false)`. Use parenthesized `(not A) and (not B)` or helper methods instead.

**Baseline health (updated):**
| File | Status |
|------|--------|
| Type, Body, Parameter, Member, MethodParser, Expression, ExpressionParser | PASS-lib (parse + HLR tests) |
| Parser | **VM dumps Members + Methods** for real files (`HelloLogger` → member `logger`, method `Run`) |
| Boolean.and/or/xor | Non-recursive `.strict` bodies + native VM handlers (fixed Type stack-overflow root cause) |
| BinaryGenerator | `is` → Equal; struct `instance.field` → FieldLoad; for-if list aggregate only on then; filter tests from bytecode |
| Expressions package | Scaffold; next focus (Phase 2) |
| C# bootstrap | Still production pipeline; Language parsers now usable under VM |

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
| `Value.cs` | Literal values | Low | ✅ Value.strict + literals |
| `ValueInstance.cs` | Runtime value wrapper | Medium | ✅ ValueInstance.strict |
| `ValueListInstance.cs` | List value at runtime | Medium | ✅ ValueListInstance.strict |
| `ValueTypeInstance.cs` | Struct-like value instance | Medium | ✅ ValueTypeInstance.strict |
| `ValueDictionaryInstance.cs` | Dictionary value at runtime | Medium | ✅ ValueDictionaryInstance.strict |
| `VariableCall.cs` | Variable reference | Low | ✅ VariableCall.strict |
| `ParameterCall.cs` | Parameter reference | Low | ✅ ParameterCall.strict |
| `MemberCall.cs` | `instance.member` | Medium | ✅ MemberCall.strict + Parse |
| `MethodCall.cs` | Method invocation | Medium | ✅ MethodCall.strict |
| `Binary.cs` | Binary ops | High | ✅ Binary.strict + IsArithmetic/IsLogical |
| `Not.cs` | Unary not | Low | ✅ NotExpression.strict |
| `Boolean.cs` | Boolean literal | Low | ✅ BooleanExpression.strict |
| `Number.cs` | Number literal | Low | ✅ NumberExpression.strict |
| `Text.cs` | Text literal | Low | ✅ TextExpression.strict |
| `List.cs` | List literal | Medium | ✅ ListExpression.strict |
| `Dictionary.cs` | Dictionary literal | Medium | ✅ DictionaryExpression.strict |
| `ListCall.cs` | Index access | Medium | ✅ ListCall.strict |
| `Declaration.cs` | let/mutable/constant | Medium | ✅ Declaration.strict + Parse |
| `MutableReassignment.cs` | Reassignment | Medium | ✅ MutableReassignment.strict |
| `If.cs` | if/else | Medium | ✅ IfExpression.strict |
| `SelectorIf.cs` | Selector if | Medium | ✅ SelectorIf.strict |
| `For.cs` | for loop | High | ✅ ForExpression.strict |
| `Return.cs` | return | Low | ✅ Return.strict |
| `To.cs` | to Type | Medium | ✅ To.strict + Parse |
| `TypeComparison.cs` | is Type | Low | ✅ TypeComparison.strict |
| `Instance.cs` | from construction | Low | ✅ Instance.strict |
| `PhraseTokenizer.cs` | Tokenize | High | ✅ PhraseTokenizer.strict |
| `ShuntingYard.cs` | Precedence | High | ✅ ShuntingYard.strict Postfix |
| `MethodExpressionParser.cs` | Full parser | Very High | ✅ ExpressionParser.strict (classifier; C# bootstrap remains) |

**Progress table:**

| Metric | Target | Actual | % |
|--------|--------|--------|---|
| `.strict` files created | 29 | **32** (+Expression, NumberChars, ParseDemo) | 100%+ |
| Test methods written | 553 | **~140 inline asserts** + 6 conversion C# tests | ~25% |
| C# files replaced | 29 | 0 (bootstrap still C#; Strict package parallel) | 0% |

**Phase 2 status (completed as parallel Strict package):**
- Package `Strict/Expressions` **loads** (local `Expression.strict` base; no Language/Expression dependency).
- All AST types PASS-lib under HighLevelRuntime with inline tests.
- `ExpressionParser` classifies lines; `ShuntingYard` Postfix; `PhraseTokenizer` tokens.
- `StrictExpressionsConversionTests` 6/6 green.
- Full C# `MethodExpressionParser` remains bootstrap; Strict package is the self-host surface.
---

## Phase 3 — `Strict.Validators` → Strict Package

**Goal:** Convert 3 C# files (451 lines) + 3 test files (45 test methods) to `.strict`.

Depends on Phases 1 & 2.

| C# File | Description | Status |
|---------|-------------|--------|
| `Visitor.cs` | Abstract visitor base | ✅ Visitor.strict (type/member/method/body line surface) |
| `TypeValidator.cs` | Unused members/vars, mutable, hide checks | ✅ TypeValidator.strict + DeclarationRules.strict |
| `ConstantCollapser.cs` | Collapse constant expressions | ✅ ConstantCollapser.strict (binary + to fold helpers) |

**Phase 3 status (parallel Strict package):**
- Package `Strict/Validators` loads with ValidationIssue, Visitor, TypeValidator, DeclarationRules, ConstantCollapser, ValidateDemo.
- Line-level analysis mirrors C# rules (unused member/variable, mutable never reassigned, parameter hides member, constant fold).
- `StrictValidatorsConversionTests` 4/4 green.
- C# TypeValidator/ConstantCollapser remain bootstrap for production Runner pipeline.

**Progress table:**

| Metric | Target | Actual | % |
|--------|--------|--------|---|
| `.strict` files created | 3 | **6** (+DeclarationRules, ValidationIssue, ValidateDemo) | 100%+ |
| Test methods written | 45 | inline asserts + 4 conversion C# tests | ~20% |
| C# files replaced | 3 | 0 (bootstrap still C#; Strict package parallel) | 0% |

---

## Phase 4 — `Strict.TestRunner` → Strict Package

**Goal:** Convert 1 C# file (37 lines) + 2 test files (20 test methods) to `.strict`.

Depends on Phases 1, 2, 3 (needs HighLevelRuntime internally).

| C# File | Description | Status |
|---------|-------------|--------|
| `TestInterpreter.cs` | Run `is` assertions in method bodies | ✅ Line-level TestInterpreter.strict over MethodUnderTest/TypeUnderTest |

**Phase 4 status (parallel Strict package):**
- Package `Strict/TestRunner` loads: TestStatistics, TestResult, Assertion, MethodUnderTest, TypeUnderTest, TestInterpreter, TestDemo.
- Line-level models evaluate simple text `is` / `is not` assertions (not full HLR expression eval yet).
- `TestDemo` runs under VM: 2 methods, 5 assertions, pass/fail results logged.
- `StrictTestRunnerConversionTests` 4/4 green.
- C# `TestInterpreter` remains bootstrap for production Runner (uses HLR).
- VM fixes landed while unblocking: assignment store uses `PreviousRegister`; `IsFileInstance` null-safe.

**Progress table:**

| Metric | Target | Actual | % |
|--------|--------|--------|---|
| `.strict` files created | 1 | **7** (interpreter + models + demo) | 100%+ |
| Test methods written | 20 | inline asserts + 4 conversion C# tests | ~25% |
| C# files replaced | 1 | 0 (bootstrap still C#; Strict package parallel) | 0% |

---

## Phase 5 — `Strict.HighLevelRuntime` → Strict Package

**Goal:** Convert 11 C# files (1,508 lines) + 7 test files (88 test methods) to `.strict`.

This is the tree-walking interpreter used for test execution and validation.

| C# File | Description | Complexity | Status |
|---------|-------------|------------|--------|
| `Statistics.cs` | Counters for test run metrics | Low | ✅ RuntimeStatistics.strict |
| `TestBehavior.cs` | Enum: OnFirstRun / TestRunner / Disabled | Low | ✅ TestBehavior.strict |
| `ExecutionFailed.cs` | Exception wrapper types | Low | deferred (Error RuntimeValue) |
| `ExecutionContext.cs` | Variable scope / call frame | Medium | ✅ line-level ExecutionContext.strict |
| `ToEvaluator.cs` | Evaluate `to Type` conversions | Medium | ✅ ToEvaluator.strict |
| `SelectorIfEvaluator.cs` | Evaluate `value is X then Y` | Medium | ✅ SelectorIfEvaluator.strict |
| `IfEvaluator.cs` | Evaluate `if condition` branches | Medium | ✅ IfEvaluator.strict |
| `ForEvaluator.cs` | Evaluate `for collection` loops | High | ✅ ForEvaluator.strict (sum/map slice) |
| `MethodCallEvaluator.cs` | Dispatch method calls | High | ✅ MethodCallEvaluator.strict (+,-,*,/,is,>) |
| `BodyEvaluator.cs` | Evaluate all expressions in a body | High | ✅ BodyEvaluator + Interpreter.EvaluateBody |
| `Interpreter.cs` | Top-level interpreter entry point | High | ✅ Interpreter.strict + ExpressionEvaluator |

**Phase 5 status (parallel Strict package + VM hardening):**
- Package `Strict/HighLevelRuntime` loads with line-level RuntimeValue + evaluators + Interpreter.
- **VM fixes:** 64 virtual registers (no silent wrap); `is not` if-conditions; comparison ops write Boolean results.
- **Working under VM:** expression eval, `let` binding + lookup, `return`, If/To/For helpers.
- Demos/tests: `RuntimeDemo`, `RuntimeValueTests`, `EvaluatorTests`, `IfToTests`, `ContextTests`, `InterpreterTests`, `BodyTests` all green.
- `StrictHighLevelRuntimeConversionTests` + `RegistryTests` + comparison codegen tests.
- C# Interpreter remains bootstrap for production test runner / validation.

**Progress table:**

| Metric | Target | Actual | % |
|--------|--------|--------|---|
| `.strict` files created | 11 | **21** (+evaluators helpers + 7 demo/test types) | 100%+ |
| Test methods written | 88 | 7 VM demos + 7 C# conversion tests + bytecode registry tests | ~30% |
| C# files replaced | 11 | 0 (bootstrap still C#; Strict package parallel) | 0% |

---

## Phase 6 — `Strict.Bytecode` → Strict Package

**Goal:** Convert 37 C# files (2,428 lines) + 5 test files (55 test methods) to `.strict`.

Includes the instruction set, bytecode generator, and serializer.

### Sub-layers

#### Instructions (24 files) — unified line-level model

| C# File | Description | Status |
|---------|-------------|--------|
| `Instruction.cs` | Abstract base instruction | ✅ `BytecodeInstruction.strict` (4-field data model) |
| `RegisterInstruction.cs` | Instruction with a register | ✅ via `register` field |
| `InstanceInstruction.cs` | Instruction with instance | ✅ via factories |
| `SetInstruction.cs` | Load literal into register | ✅ `InstructionBuilder.SetNumber` |
| `LoadConstantInstruction.cs` | Load named constant | ✅ `InstructionBuilder.LoadConstant` |
| `LoadVariableToRegister.cs` | Load variable into register | ✅ `InstructionBuilder.LoadVariable` |
| `StoreVariableInstruction.cs` | Store value as variable | ✅ `InstructionBuilder.StoreConstant` |
| `StoreFromRegisterInstruction.cs` | Store register into variable | ✅ `InstructionBuilder.StoreRegister` |
| `BinaryInstruction.cs` | Binary operation (add, mul, etc.) | ✅ `InstructionBuilder.BinaryOp` |
| `Invoke.cs` | Method invocation | ✅ `InstructionBuilder.InvokeOp` + `InvokeInfo` |
| `PrintInstruction.cs` | Output to console | ✅ `InstructionBuilder.PrintOp` |
| `ReturnInstruction.cs` | Return from method | ✅ `InstructionBuilder.ReturnOp` |
| `Jump.cs` / `JumpIfTrue` / `JumpIfFalse` | Conditional/unconditional jumps | ✅ `InstructionBuilder.JumpOp` |
| `JumpIfNotZero.cs` / `JumpToId.cs` | More jump variants | ✅ codes in `InstructionType` + `IsJump` |
| `LoopBeginInstruction.cs` / `IterationEnd.cs` | Loop control | ✅ `LoopBeginOp` / `LoopEndOp` |
| `ListCallInstruction.cs` | List index access | ✅ `ListCallOp` (`IndexCall` const; List* prefix banned) |
| `WriteToListInstruction.cs` / `WriteToTableInstruction.cs` | Mutation | ✅ codes on `InstructionType` |
| `RemoveInstruction.cs` | Remove from list | ✅ codes on `InstructionType` |

#### Generator & Serialization (13 files)

| C# File | Description | Status |
|---------|-------------|--------|
| `Register.cs` | Register slots + count | ✅ `Register.strict` (`Count=64`, `NameOf`, `IsValid`) |
| `Registry.cs` | Register allocator | ✅ `Registry.strict` (no wrap; returns -1 when exhausted) |
| `InstructionType.cs` | Instruction type enum | ✅ `InstructionType.strict` + `InstructionNames.strict` |
| `InvokedMethod.cs` / `InvokeMethodInfo` | Method call wrappers | ✅ `InvokeInfo.strict` |
| `BinaryGenerator.cs` | Expression → instructions | ✅ `LineGenerator` + `ExpressionCodegen` (line-level) |
| `Decompiler.cs` | Bytecode → partial .strict source | ✅ `Decompiler.strict` |
| `Serialization/ExpressionKind.cs` | Enum for expression serialization | ✅ `ExpressionKind.strict` |
| `Serialization/ValueKind.cs` | Enum for value serialization | ✅ `ValueKind.strict` |
| `Serialization/NameTable.cs` | String table for bytecode | ✅ `NameTable.strict` (+ BuiltIn names) |
| `Serialization/BinaryType` / `BinaryMethod` / `BinaryMember` | Type + method bytecode bundle | ✅ `BinaryTypeData` / `BinaryMethod` / `BinaryMember` |
| `BinaryExecutable.cs` | Methods-per-type + entry | ✅ `BinaryExecutable.strict` |
| `Serialization/BytecodeSerializer.cs` | Write `.strictbinary` ZIP | 🚧 Deferred (ZIP/binary I/O still C#) |
| `Serialization/BytecodeDeserializer.cs` | Read `.strictbinary` ZIP | 🚧 Deferred (ZIP/binary I/O still C#) |

**Phase 6 status (parallel Strict package):**
- Package `Strict/Bytecode` loads with instruction model, registry, line generator, decompiler, name table, and binary metadata types.
- **Instruction model:** 4-field `BytecodeInstruction` (`typeName`, `register`, `amount`, `label`) + `InstructionBuilder` factories + `InstructionText` formatting (Strict member/param limits).
- **Line-level codegen:** `LineGenerator` / `ExpressionCodegen` emit load/store/binary/return/jump from simple expression lines (same style as HighLevelRuntime evaluators).
- **Working under VM:** Registry, instruction factories, name table, generator, decompiler, values, executable assembly.
- Demos/tests: `BytecodeDemo`, `RegistryTests`, `InstructionTests`, `NameTableTests`, `GeneratorTests`, `DecompilerTests`, `ValueTests`, `ExecutableTests` all green.
- `StrictBytecodeConversionTests` 8/8 green.
- C# `BinaryGenerator` / ZIP serializer remain bootstrap for production Runner pipeline.
- ZIP serialize/deserialize remain deferred (see Missing Runtime Features).

**Progress table:**

| Metric | Target | Actual | % |
|--------|--------|--------|---|
| `.strict` files created | 37 | **30** (core + demos; ZIP ser/deser deferred) | ~80% |
| Test methods written | 55 | 8 VM demos + 8 C# conversion tests + inline asserts | ~30% |
| C# files replaced | 37 | 0 (bootstrap still C#; Strict package parallel) | 0% |

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
| 2 | `Strict.Expressions` | 29 | 29 | **32** (AST + Parser + NumberChars + demo) | ~140 + 6 C# | **~40%** |
| 3 | `Strict.Validators` | 3 | 3 | **6** | ~4 C# + inline | **~40%** |
| 4 | `Strict.TestRunner` | 1 | 1 | **7** | ~4 C# + inline | **~40%** |
| 5 | `Strict.HighLevelRuntime` | 11 | 11 | **21** | ~7 C# + demos | **~35%** |
| 6 | `Strict.Bytecode` | 37 | 37 | **30** | 8 demos + 8 C# | **~40%** |
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
| `File.ReadAllLines` | 1 (Language) | 🔴 Critical | ✅ Via `File(...).ReadLines` (`TextReader` trait; VM accepts Path or Text) |
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
