# Strict Self-Hosting Conversion Plan

Before changing any `.strict` file, build `Strict` and use the Strict executable to parse, compile,
run tests, and execute the changed Strict code.

**Top priority.** The goal is to progressively convert all C# implementation of Strict into `.strict`
files, using the current C# bootstrap (`Strict.exe` on .NET 10) to compile and run the new Strict
implementation. Long term: get rid of the C# bootstrap entirely and have everything in `.strict`.

This file tracks which layers have been converted, what `.strict` files exist, how many tests are
written in Strict, and what C# features are still missing from the Strict runtime.

---

## Roadmap — remaining work (2026-10-09)

Phases: A stabilize, B clean C#, C performance, D self-hosting, E usability, F hardening.
Order: A → B → C1–C4 → D1–D3 → C5–C7 → D4–D6 → E → D7 → F.

### Phase A — Stabilize what exists (≈3 sessions)
A1 Open bugs found so far, each with one regression test:
1. Interpreter returns None for `cond then x else obj.Method(Type(...))` (seen in
   LineGenerator.GenerateBody); minimal repro in InterpreterTests, fix HLR conditional evaluation.
2. Printer emits `(a is b) and c is in (...)` which the parser rejects; `a is b and c is d` silently
   parses as `a is (b and c) is d`. Fix printer/parser to agree; add validator error
   `AmbiguousComparisonPrecedence` for unbracketed mixes.
3. `Examples/Parsing/*`, `Examples/CompactTypeTest/*` fail standalone (MethodRequiresTest,
   ValueTypeNotMatchingWithAssignmentType, UseConstantHere); fix and include subfolders in
   `StrictProgramPaths`.
4. Source vs binary mismatch `holder.WithName("y").Name` ("Y" vs "y": member `Name` vs type Name).
5. Interpreter `for MethodCall / list.Add(value)` copy returned 0 (seen in TypeValidator.Validate).
6. `Range` loops only define `index`; `value` silently becomes the instance → validator error
   `ValueIsNotDefinedInRangeLoop`.
7. Single-element list double brackets vs validator (fixed for calls; verify for constructors).
8. Flaky `NativeImageLoadProcessSavePipeline` first run after build (shared output paths) → unique
   output dirs per test.
9. Cache: include OS/runtime build in binary cache validity (Platform.Current is folded as constant).
10. VM `list + element` now copies: verify no regressions in Mutable list semantics (List.Add uses
    `value = value + element`), add test for `mutable list` growing in a loop on VM + interpreter.
11. Member constraints (`has text with Length > 1 ...`) are never checked at construction (only used
    to pre-fill list lengths): `Name("y")` is accepted. Enforce in interpreter and VM (generate the
    check in BinaryGenerator). A one-off `Y` vs `y` output difference was seen only with such an
    invalid Name; recheck determinism after constraints are enforced.

A1.11 status: interpreter checks member constraints on construction (also runs custom `from`
bodies that assign members, which it skipped before; Color's test relied on Byte clamping of an
invalid ColorValue(255, 0, 0)). The VM does not check constraints yet: generating the checks in
BinaryGenerator costs time in hot loops (ColorValue per pixel), decide together with Phase C
(compile time check for constant arguments + optional VM checks).

Open (found while verifying): 5 Slow Strict.Transpiler tests (ExecuteOperation,
GenerateFileReadProgram, LinkedListAnalyzer, ReduceButGrow, RemoveDuplicateWords) already failed
before 2026-10-09 (expected C# output out of date with the converted Examples) → fix in Phase B.

Phase B result (2026-10-09): B1 done (no production C# file above 504 lines except
ValueInstance.cs with its commented-out packed RGBA experiment, all splits audited line by line),
B2 test skips removed except generic List/Dictionary, parser-limitation ignore removed, B4
compiler exception/platform duplicates unified, 5 stale transpiler tests fixed, 3 TODOs resolved.
Moved to Phase D: AST based replacements for Body value piping and validator double-bracket scan.

Phase B notes (2026-10-09): skipped base tests now run (Number, Expressions package, to Text,
digits, Text.Split, Parser, ShuntingYard). Still skipped: generic List/Dictionary methods (need a
concrete implementation to run, e.g. List.Length parsed for List(Generic) fails). Follow-ups found:
- a bare type name as value (`Variable("count", Number)`) parses as `Number()`, needs type values;
- a `constant` passed to a `mutable` parameter is mutated in place (AdjustBrightness test images),
  the validator should reject it or the call should copy;
- AdjustBrightness should clamp channels (Number.Clamp) but that adds calls to the hot loop and
  breaks inlining tests, do it together with Phase C inlining work;
- interpreter call depth is limited to 128 (thread-pool stacks overflow at ~60–200 nested Strict
  calls), run interpreter work on threads with bigger stacks or make evaluation iterative (Phase C).

Phase A result (2026-10-09): A1.1–A1.11 done except VM constraint checks (moved to Phase C), A1.4
`Y`/`y` not reproducible, A1.5/A1.7/A1.9 no longer issues. A1.8 binaries are written atomically
and a busy cache file no longer fails a run. A2 specific exception types in runtime/bytecode
(compiler backends follow in B4). A3 binaries byte-identical across runs (fixed zip timestamps).

Done so far (2026-10-09): A1.1 inline then/else on mutables, A1.2 `is` comparison type check and
`is in` brackets, A1.3 subfolder examples + `from` member initialization, A1.5 no longer
reproducible (fixed by value piping fix), A1.6 `value` in number/Range loops is the current number
in parser and VM (VM used index + 1, README says `for 10` logs 0..9).
A2 Error quality: every failure path uses `ParsingFailed` / `InterpreterExecutionFailed` /
   `RuntimeError` with clickable `file:line` stack traces back to `.strict` source; remove remaining
   `NotSupportedException` / `InvalidOperationException` throws (AGENTS rule). One test per type.
A3 Determinism: binaries byte-identical for same input (sorted type/method order), so caches and
   differential tests are stable.

### Phase B — C# code cleanup (≈4 sessions, before porting so we port clean code)
Porting messy C# to Strict multiplies the mess; clean first, then mirror.
B1 Split files >500 lines per AGENTS (2x Limit rule), keeping high-level flow in the main file:
   - `BinaryGenerator.cs` → generator core + `ExpressionCodegen`, `LoopCodegen`, `ConditionCodegen`,
     `ListCodegen` (similar to Executor split rule).
   - `VirtualMachine.Methods.cs` → invoke core + `NativeTextMethods`, `NativeProcessMethods`,
     `NativeDirectoryMethods`, `NativeFileMethods` (one table-driven dispatcher instead of
     `MethodName ==` chains; interpreter reuses the same native table → one implementation of
     Directory/Process/Text natives instead of two).
   - `Interpreter.cs`, `MethodCallEvaluator.cs`, `Type.cs`, `TypeParser.cs`,
     `MethodExpressionParser.cs`, `ValueInstance.cs`, `BinaryExecutable.cs`.
B2 Remove special cases: `TestInterpreter.ShouldSkipKnownDummyBaseType/Method`,
   `Interpreter.ShouldSkipKnownStrictBaseMethodValidation`, VM "historical" Boolean native hacks,
   `IsFakeBodyForMemberInitialization`, text-based heuristics (`Body` value piping substring check,
   `TypeValidator` double-bracket line scan, `Method` IsTestExpression line checks) → AST based.
B3 Resolve the 73 TODOs (fix or delete with reason), review the 413 `ncrunch: no coverage`
   markers: either cover with a test or delete the dead code. Delete unused methods
   (`//TODO: unused again?` in BinaryGenerator etc.).
B4 Duplicate code: one `Registry`, one operator table, one platform/tool table shared by
   Compiler.Assembly, LlvmLinker, MlirLinker, NativeExecutableLinker; one cache-validity helper.
B5 Strict code cleanup (current `.strict` packages): delete demo-only types that duplicate tests
   (`*Demo.strict`, `*Tests.strict` that only log) once real inline tests cover them; consistent
   naming (no `C#`-style getters), every method with tests (MethodRequiresTest), apply the static-like
   type guide everywhere.
Verification: all suites green, file size report (no C# file >500 lines), TODO count 0 or justified.

### Phase C — Performance (≈5 sessions, measured before/after, numbers go into the plan file)
Baseline first: a `Benchmarks` Slow test set (BenchmarkDotNet already referenced) recording
parse time/package, test-run time, bytecode gen, optimize, VM run, allocations, exe size.
C1 Allocation budget (`RunAdjustBrightness` failing): allocation profiling of ValueInstance,
   CallFrame, list copies; flat float32/double backing for numeric-only types (AGENTS VM collection
   rules), preallocated register arrays, remove string/dictionary frame lookups (CallFrame symbol
   ids everywhere, no FrameKey fallbacks). Goal: test green with margin.
C2 VM hot paths: table dispatch instead of `switch` on method names, cached method instruction
   lookup (`GetPrecompiledMethodInstructions` tries 5 lookups per call), no per-call frame
   allocation (pool exists, verify), loop state without dictionaries (`SavedCustomValues`).
C3 Lists: copy-on-write or in-place `Mutable(List)` Add so Strict loops building lists are O(n)
   (now `list + x` copies → O(n²), visible in all Strict packages).
C4 Parser/loader: `FindTypeCount = 1069` for a tiny demo → cache by name per Context, avoid
   regex scans for dependency detection per file, parallel package loading (package level only,
   per AGENTS), lazy method body parsing kept.
C5 Interpreter (inline tests): avoid re-running tests for already validated methods across runs
   (cache test results keyed by source hash), faster value equality.
C6 Native output: register allocation (no spilling bugs beyond xmm14), keep values in registers
   across lines, constant data dedup, measured exe size/speed vs C# `InstructionsToAssembly`.
C7 Strict-on-Strict speed (needed for self-hosting): time to run the Strict parser over the whole
   repo and SourceCompiler on all Examples; target seconds, not minutes (Strict goal: millions of
   lines evaluated in real time).
Verification: each item shows measured improvement (AGENTS: real numbers, not flag changes).

Phase C1–C4 result (2026-10-09), AdjustBrightness, Release unless noted:
- C1 16x9 Debug run allocations 405 KB → 170 KB, cached binary loading 417 KB → 374 KB; the Slow
  allocation test is green (measures a warmed-up run, cache regeneration is compile cost).
  Fixes: flat numeric FieldLoad/ConstructValueType without materializing, binary members and invoke
  names cached, no development path probing when loading binaries.
- C2 320x180 VM run (min of 7, noisy machine): 525 ms → ~390 ms. Lock-free symbol ids, access paths
  cached once per block, Type.IsTrait and constant member values cached in the VM. Register
  save/restore was measured at ~7 ms total, not worth changing. Still ~7 µs and 850 B per pixel:
  remaining cost is per-invoke scope setup (name based Frame.Set), the `for image.Size` iterator
  materializing every Vector2 although the body only uses `index`, and ~50 ns per variable load.
- C3 `list = list + element` compiles to the in-place append of List.Add: 40000 appends 19069 ms /
  65 GB → 22 ms / 2 MB.
- C4 Context.FindType already caches hits; package loading is now logged (Compiler package
  ~105 ms, 5.8 MB). ReadyToRun publishing only saved ~10% on source runs, nothing cached, skipped.
Follow-ups: parameter/member symbol ids precomputed per invoke and type, lazy custom iterators,
per-instruction cached access paths, VM constraint checks (from Phase A).

### Phase D — Self-hosting milestones (≈15–25 sessions)
D1 Real front end in Strict (Language + Expressions): full Type/Member/Method model, Package/Context
   lookup (parent + children, generics, plural types, traits), tokenizer + shunting-yard producing an
   expression tree (Number, Text, Boolean, List, Dictionary, MemberCall, MethodCall, ListCall, Binary,
   Not, To, If/then-else, For, Return, Declaration, MutableReassignment, Variable/Parameter calls,
   value/index/outer) and canonical `ToString` round trip. Differential Slow test: Strict parser
   output == C# parser output for every `.strict` file.
D2 Validators in Strict over the tree (TypeValidator, ConstantCollapser). Differential test:
   same diagnostics as C#.
D3 Bytecode generation from the tree (mirror cleaned BinaryGenerator): Invoke with params/instance,
   fields, constructors, lists, text, nested blocks (label ids), for over lists, return types;
   `.strictbinary` writing via `BytesWriter.strict` (stored zip first). Differential test:
   Strict-generated binaries run in C# VM with identical output for all Examples.
D4 Optimizer parity: CompactType, MethodInlining, ConstructorToFieldMutations,
   LoopInvariantCodeMotion, JumpThreading over labels, RedundantLoad with register remap
   (like `Strict.Optimizers/RedundantLoadEliminator.cs`). Measured reduction per example vs C#.
D5 VM in Strict (Runtime/): full instruction set, frames, loops, lists, text, natives via host hooks.
   Differential test: same output as C# VM for all Examples.
D6 Native compiler for the full subset: calling convention + stack frames, Run(numbers) from argv,
   lists, text + printing, Linux/macOS/Windows entry points verified in CI.
D7 Bootstrap: Strict compiler compiles all packages (equal to C# output); Runner stages switch one
   at a time to Strict (parse → validate → test → bytecode → optimize → run), C# stage deleted once
   its differential test is green; finally the Strict compiler compiles itself natively and the
   produced exe compiles Examples (stage-2 bootstrap). Dashboard "C# replaced" tracked per phase.

D1 progress (2026-10-09): syntax layer done. `Expressions/Tokenizer` (character level, escaped
quotes), `SyntaxNode` (tree, C# precedences, canonical printing incl. `(not x) and y`),
`SyntaxParser` (operators incl. `is not`, `is in`, `is not in`, then/else, unary, member and call
chains, lists vs grouping), `StatementParser` (let/constant/mutable, reassignment, return, if,
else if, selector if lines, for). `RoundTrip.strict` re-prints every method line; the Slow test
`StrictParserRoundTripsEveryLine` asserts 0 mismatches for all 18 folders (runs on the VM).
Name resolution and type layer: `HeaderTokens` (member and method header tokens: names, declared,
value, parameter and return types incl. `Mutable(T)`, `List(T)`, `Package/Type`), `TypeShape`
(member scope and method/constant scope of a type, one tokenizer pass each), `KnownTypes` (all
package types with their scopes: plural lists, element types, members reached through member types
like File → TextReader or RegisterBank → CallFrame → Texts, single value member wrappers like
Degrees keep their type), `TypeInference` (types of literals, identifiers, calls, list calls,
members, operators, conversions, conditionals, implicit `value` calls), `MethodBodies` (flattens
the body trees into statements with the scope valid at each line: members, `value`, parameters,
declarations, loop `value`/`index`/`outer`), `FileCheck`/`ResolveCheck` (every name in a method
line resolves) and `TypeReport` (type of every statement line). Slow tests for all 18 folders:
`StrictParserRoundTripsEveryLine` (0 mismatches, whole bodies re-printed from the tree),
`StrictResolvesEveryName` (0 unresolved) and the differential test
`StrictInfersSameTypesAsCSharp` (every line the C# parser types gets the same type from the
Strict front end; dictionary key/value types are compared by generic name only). ResolveCheck on
Expressions runs in 2.7s on the VM (precomputed type scopes, iterative tokenizer).
Still missing for D1: typed nodes (each call resolved to a method, member, variable or parameter
in the tree itself), generic arguments beyond list element types.
Bugs found by ResolveCheck (each with a test): VM loop over a Range returned by a method,
ConstructorToFieldMutations mapped `from` arguments to the wrong members (Range's iterator trait
first), VM negative list index (`List.Last`), Boolean `let` lines were taken as inline tests and
dropped from the body, VM list call on a computed list (`node.children(0)` on a method result).
Bugs found by the type differential test (each with a test): `let` lines piped their value into
the loop `value`, implicit test instances got number 0 for list members, the C# printer dropped
brackets of an equal precedence right operand (`5 - (3 - 1)` could not be written), For, If and
selector if line numbers pointed to the end of their body (also wrong VM loop source lines), the
validator did not visit `return` values, constants created by a constructor were compiled to their
source text, a method lookup cycle (`List(Keyword).from(Texts)`) overflowed the stack because the
cycle guard compared argument type arrays by reference, a conditional line starting with `(a) and`
and ending with `)` lost its brackets, the constant collapser deleted declarations only used through
a member or list call (`let condition = x.children(0)` + `condition.Kind`).
Open C# issues: constants of another type (`KnownTypes.Sample`) or calling methods of another
package's type in member initializers fail when that type is parsed later; a constant named like a
type (`keywords` → `Keyword`) is typed by its name instead of its value; `mutable x = MethodNames`
looks for a plural type `MethodNames` instead of calling the method; unused variables in nested
bodies are only checked when the method body itself has variables.

D2 progress (2026-10-09): `Validators/` rewritten over the tree. `UsageRules` checks one method
(unused, hidden and never changed variables and parameters; a mutable counts as changed when it
is reassigned, called with a method returning `Mutable(...)` or passed to a `mutable` parameter,
`HeaderTokens.Mutations` lists those per header), `TypeValidator` adds unused private members
(same exemptions as C#: data types, traits, single value member types). Rule names are the C#
exception names. `ValidateCheck` runs over a folder; Slow tests: `StrictValidatesEveryFile` (0
issues in all 18 folders) and `StrictValidatorReportsSameRuleAsCSharp` (7 invalid samples, C#
TypeValidator and the Strict validator report the same rule). The old text based Visitor,
DeclarationRules and ValidateDemo are deleted. Fixed on the way: a `constant` line mentioning
"generic" made its type generic. `ConstantCollapser` now folds over the tree like C# (number arithmetic, text
concatenation, `to Number`/`to Text` of literals, boolean and/or) and TypeValidator reports
`UseConstantHere` for members computed from constants (8 rules in the differential). Also fixed:
the interpreter test mode evaluated declarations using a member reached through another member
(`node.children(0)`) on the empty implicit instance. Still missing for D2: the double bracket
list argument rule (C# checks it on the line text).

D3 progress (2026-10-09): binary primitives in Strict. `Bytecode/ByteEncoder` writes little
endian integers, 7 bit encoded integers, UTF-8 texts with length prefix, IEEE 754 doubles and
CRC32 (arithmetic XOR, Strict has no bit operators), `ZipEntry`/`ZipWriter` write a stored ZIP.
Slow test `StrictZipWriterOutputOpensWithZipArchive`: the bytes produced on the VM open with the
.NET ZipArchive (CRC checked) and give back the entry contents. VM fixes found on the way (each
with a test): `^` compiled to an endless `Number.^` invoke (new Power instruction), `text(index)`
gave a Text instead of a Character, `Character("h")` stayed a Text, members of a computed
Character/Text (`Character(value).number`) were loaded as a variable named after the expression.
A single element for a list parameter (`ZipWriter(ZipEntry(..))`, `GenerateReturningBody("x")`)
is now wrapped into a list by the parser, the old `(("x"))` workaround is gone (A1.7); the
ConvertingNumbers example passed only because `ConvertingNumbers(3)` stored a Number in its
`numbers` member and iterated it as a count, it now passes the list `(0, 1, 2)`.
The `.strictbinary` format in Strict: `BinaryNames` (69 prefilled names like the C# NameTable plus
custom names), `InstructionEntry` (Print payload), `MethodEntry`, `TypeEntry` (magic byte, version,
name table, members, method groups) and `BinaryFile` (one `.bytecode` ZIP entry per type). Slow
test `StrictWrittenBinaryRunsOnTheVirtualMachine`: a Hello program written by Strict runs on the
C# VM. Single elements are now also wrapped for constructors with more parameters
(`SingleElementForListParameterAmongOthersIsWrapped`), as a fallback after normal overload
resolution so operators like `list + element` keep their own overloads.
Next: all instruction payloads, then tree codegen with the VM output differential.
Flaky once in a full parallel solution run (passes alone and in reruns):
`InterpreterTests.ParserParsesExistingTextStrictFile` and
`LoadStrictExamplesPackageAndUseBasePackageTypes`, likely package files changing while
Strict.Tests runs programs there (Phase F thread safety item).
Bugs found and fixed on the way (each with a test): parser stack overflow on `.`/`)` inside text,
text-unaware member splitting, interpreter return type check on nested bodies, VM IndexOf
startIndex, list to Text truncation at 10 items, register reuse per statement (64 register limit),
constant folding / strength reduction / LICM / inlining assumptions broken by register reuse, VM
and interpreter short-circuit and/or, negative and out of range list indexes, VM list constants and
own `to Text`, VM endless recursion as StackOverflow, test lines with ` then ` in text, VM loop
state on re-entrant calls, inline conditional values on the VM.
Fixed C# parser issues: escaped quotes in one of several arguments, `\\` before `\"` and bracketed
comparisons at the start of a line all came from seven text literal scans, now one
`TextLiteral.Advance`. Still open: redundant brackets in arguments and then/else are accepted
although the printer drops them. VM: `FindJumpEndInstructionIndex` scans linearly on
every jump, `DeadStoreEliminator.FindProducerInstruction` may pick a side effecting producer.

### Phase E — Usability and product quality (≈4 sessions)
E1 CLI: clear usage, `strict run|test|build|decompile|check` commands, consistent exit codes,
   `-Windows/-Linux/-MacOS`, diagnostics flag shows stage times + instruction reduction.
E2 Errors: human-readable messages (AGENTS error guidelines), suggestions for common mistakes
   (double brackets, `value` in Range loops, precedence), source excerpts with caret.
E3 Docs: README sections for language rules discovered here (Range loops use `index`, `is`
   precedence, cross-package `Package/Type` references only in declarations, no static-like types),
   updated conversion guide, examples for native compilation.
E4 Tooling: LanguageServer diagnostics use the same validators; VS Code extension smoke test;
   `strict check` used by CI.
E5 CI: Windows + Linux runs of all suites incl. Slow and native compile tests; nightly benchmarks
   with regression thresholds (fail on >10% slowdown or allocation growth).

### Phase F — Hardening (continuous, ≈2 sessions final pass)
- Fuzz the parser with mutated `.strict` files (no crashes, only ParsingFailed).
- Thread safety of Repositories/package cache under parallel tests (AGENTS multithreading rules).
- Memory/time limits in VM (stack overflow detection, step limits) with clear RuntimeErrors.
- Binary format versioning + compatibility tests (old cache → clean regenerate).


## Native loops — 2026-10-09 (late night, part 4)

- LineGenerator handles `mutable` declarations, plain reassignments and `for N` loops (`index`
  store, start label, `index < N` check, body, increment, back jump, end label). Blocks close
  through `CloseBlock` for if, else and loops (one nesting level, `index` only, ponytail).
- `Examples/NativeLoop` (sum of 0..9) compiles natively and returns 45.
- Slow test `StrictSourceCompilerBuildsAndRunsNativeExecutable` compiles NativeArithmetic (20),
  NativeConditions (30) and NativeLoop (45) with the Strict compiler and checks the exe results.
- Not yet compiled natively: method calls, parameters, lists, text, nested blocks, `for list`.

## Native if blocks — 2026-10-09 (late night, part 3)

- LineGenerator: `if` emits the condition + `JumpToIdIfFalse ifN`, the block end emits `JumpEnd ifN`
  (id based, so optimizers removing instructions keep jumps valid; one nesting level). `else`
  emits `Jump ifNEnd` + `JumpEnd ifN` and closes with `JumpEnd ifNEnd` (NASM `jmp`).
- Open bug: the interpreter returned None for `openId is "" then current else current.Append(..)`
  in LineGenerator.GenerateBody (rewritten as if/return); needs a minimal repro and root fix.
- NASM: comparisons produce 1.0/0.0 via ucomisd + setcc, `JumpToIdIfFalse` tests against zero,
  `JumpEnd` is a label, every `Return` emits its own `ret`. UnreachableCode keeps code after a
  `JumpEnd`. `Examples/NativeConditions` compiles natively and returns 30 (1 with limit 3).
- Parser fix: Body value piping retyped `value` after control flow lines and mutable reassignments
  (`if flag` / `next.Add(value)` inside a loop saw value as the list), now only plain lines pipe.

## Strict optimizers in the native pipeline — 2026-10-09 (late night, part 2)

- Optimizers use `Bytecode/BytecodeInstruction` + `Bytecode/InstructionList` (OpList/OptimInstruction
  removed, InstructionList took over the list API). SourceCompiler runs `AllOptimizers` before NASM:
  `Examples/NativeArithmetic` goes from 10 to 2 instructions (`(10, 2)`), the exe still returns 20.
- ConstantFolder propagates single-assignment constants and folds Load/Load/Binary anywhere in the
  list (cascading). RedundantLoad only drops identical consecutive loads (the old version removed a
  load into a different register and broke later reads; upgrade path: remap reads like the C# one).
- Runner treats a cached binary as outdated when a used package (`Strict/Optimizers`, base types, ..)
  has a newer `.strict` file, not just files next to the entry file.
- `Process.OperatingSystem` (native in interpreter and VM) feeds `Platform.Current`; SourceCompiler
  and CompilerDemo no longer hard-code Windows. NativeBuild uses the same link flags as the C#
  NativeExecutableLinker (no CRT, own entry point): NativeArithmetic.exe 17,408 → 3,072 bytes.
- Runtime uses the shared model too (`VmInstruction`/`InstrList` removed): Bytecode, Optimizers,
  Runtime and Compiler now all work on `Bytecode/BytecodeInstruction` + `Bytecode/InstructionList`.

## Strict compiles Strict source to a native exe — 2026-10-09 (late night)

- `Compiler/SourceCompiler.strict Examples/NativeArithmetic.strict` runs entirely in Strict:
  read file → Run body + constant members → `Bytecode/LineGenerator` → `Compiler/InstructionsToNasm`
  → nasm + gcc → runs the exe and logs `Run returned 20`. The compiled exe exits with the result.
- Compiler package shares `Bytecode/BytecodeInstruction` (cross package via full name in one member
  or parameter type). `CompInstruction`/`CompList` removed. Data section is generated from constant
  loads/stores and stored variables. Optimizers and Runtime still have their own instruction copies
  (`OptimInstruction`/`OpList`, `VmInstruction`/`InstrList`), next to unify.
- `ExpressionCodegen` has precedence levels (compare < additive < multiplicative), left associativity
  and recursive operands, no more duplicated instructions. `LineGenerator` resets registers per line
  and `GenerateReturningBody` returns the last value.
- `Package.Load` loads child packages (Directory.Directories), `HasType`/`FindType` search children.
- Runtime/parser fixes found on the way (all with regression tests): Text.in/IndexOf/LastIndexOf missed
  a match at the end, Text.Trim only trimmed spaces, VM list + list concatenates without mutating the
  left list, VM skips loops over empty lists, dynamic list literals in declarations, implicit calls in
  loops use the method instance, relative process paths resolve against the current directory,
  `x.member(index).y` keeps the index, auto-wrapped list arguments print without double brackets
  (matches the validator), single-element lists may keep double brackets, test-only runs skip
  declarations that use parameters.
- Limits (ponytail): numbers only, no if/for/calls in compiled code, Windows entry point hard-coded,
  virtual registers map 1:1 to xmm0..xmm14 (registers reset per line). Subfolder examples
  (Examples/Parsing, Examples/CompactTypeTest) are not in the program suite and still fail.

## All Strict projects run — 2026-10-09 (night)

- `RunStrictProgramFromSourceAndCachedBinaryInFreshProcess` passes for every `.strict` file in
  Examples, Language, Expressions, Validators, TestRunner, HighLevelRuntime, Bytecode, Optimizers,
  Runtime and Compiler (source run with validation and inline tests, then the cached binary).
  Only the Slow allocation budget test (`RunAdjustBrightnessAllocatesBelowHalfMegabytePerRun`) fails.
- Unused members and variables are rejected when files are loaded (Repositories, Runner).
- VM sets every valued instance member in the call frame, a stubbed `List(T)` from a binary looked
  like a trait and its members were skipped (`parameters` unresolved in cached ExecutableTests).
- This is "every file parses, validates, passes its tests and runs", not self-hosting: the Strict
  implementations are still line-level subsets and no C# layer is replaced (see dashboard).

## Static-like types removed — 2026-10-09 (evening)

- All `has dummy Number` types are gone (see "Converting C# static classes" under Rules for the
  guide). Builders became factories on the built type (`OptimInstruction.LoadConstant`,
  `VmInstruction.ReturnOp`, `CompInstruction.BinaryOp`, `BytecodeInstruction.SetNumber`); passes and
  executors own their data (`ConstantFolder(ops).Optimize`, `VirtualMachine(ops).Run`,
  `InstructionExec(state, instruction).Execute`, `ArithmeticExec(left, right).Compute(op)`,
  `NasmFormat(platform).Format`, `EntryPoint(name, platform).Build`, `ToolRunner(name).Execute`,
  `DeclarationRules(line).IsDeclaration`, `NumberLiteral(text).IsNumber`). Removed OpBuilder,
  IdentityRules, InstrBuilder, CompBuilder, RegisterMap, InstructionBuilder.
- Optimizers, Runtime and Compiler folders pass fully from source and binary; demo outputs now
  compute correct results (constant folding `5 + 3` → `8` in r2, Windows linker command).
- Runtime fixes found on the way (each with a test): VM number comparisons used as values
  (`<`, `>=` in `and`/`or` recursed forever), `>=`/`<=`/`in` in if conditions compiled to Equal,
  Boolean `for` loops return true when any iteration is true, inline `a then b else c` as last
  expression returns its value, RedundantLoadEliminator only remaps reads until the register is
  rewritten (and remaps Invoke/FieldLoad/ConstructValueType/WriteToList), `value = ...` inside a
  method no longer leaks into the caller's loop `value`, type-qualified calls never borrow an
  unrelated caller instance, method last line is never a test, `Type.None` resolves to the type's
  method, constructor args like `Holder(code).Name, x` tokenize, text brackets are ignored when
  matching brackets, `TypeName(value)` is a constructor not a generic type, cached binaries older
  than the runtime regenerate.
- Remaining standalone failures (16): HighLevelRuntime (BodyResult, Evaluators, ExecutionContext,
  ForEvaluator, Interpreter, RuntimeStatistics, RuntimeValue), Bytecode (BinaryExecutable,
  BinaryTypeData, BytecodeValue, Decompiler, ExecutableTests, InvokeInfo, NameTable, Registry) and
  Validators/TypeValidator: unused data members, methods without tests, for-loop round trips and a
  few logic bugs in the generated code.

## Example and project suite — 2026-10-09 (later)

- New Slow test `RunStrictProgramFromSourceAndCachedBinaryInFreshProcess` runs every `Examples/*.strict`
  and every `.strict` in the nine project folders in a fresh process (source, then cached binary for
  programs with `Run`; library types must parse, validate and pass tests up to "No Run method").
- All 37 examples pass from source and binary. All `Language` files pass. Every demo/test program
  passes except `Bytecode/ExecutableTests` (VM: unresolved `parameters`).
- Runtime fixes: `Run(numbers)` without args uses an empty list; binaries stub generic bases
  (`ErrorWithValue(Number)`) and nested generic args (`List(Mutable(Text))`) on demand; entry-package
  types load into their own child package so `Language/Type` no longer merges with `Strict/Type`;
  `IsFileInstance` no longer needs File in the binary; Examples is loaded as a package (sibling types);
  RunExpression reuses an already loaded type.
- Language/validator fixes: `Type.IsMutable` only for `Mutable`/`Mutable(...)`; existing `List*` types
  resolve; same-named method on another typed member is not recursion; own type name call is the
  constructor; an uppercase member wins over a same-named type in `Member.x`; text literals with
  spaces are unescaped; `()` inside text is allowed; `Mutable(...)`-returning calls (`list.Add`)
  count as mutation; implicit loop `value` stays implicit; ConstantCollapser leaves non-literal `to`
  alone; test lines calling their own method are no stack overflow; `Type` usable-member cache is
  thread safe; Text.Substring clamps like the VM.
- Remaining standalone failures (~80): ~30 `has dummy Number` helper types (design decision pending),
  data members never read, methods missing tests, and a few interpreter/VM issues.

## Phase 1 continuation — 2026-10-09

- Added `Package.Load`, `ReadTypes`, and `ReadType` in Strict plus `Language/PackageTests.strict`.
  Local directories load type names and source lines through Directory.Files and File.ReadLines.
  Child packages, lookup validation, and integration with full expression parsing remain pending.
- Fixed Path.FileName for Windows separators, interpreter File(Path), and dynamic console
  concatenation (only literal prefixes use the optimized Print instruction).
- Fresh-process source/binary loader regression passes. Fixed declaration order for generic
  constants, ConstructValueType payload serialization, and reconstructed embedded member layouts.
- Bytecode format is version 4. Version 3 stored signature-only trait methods so cached
  ImageLoader width/height dispatch works, and restored primitive constructor defaults such as
  `Byte(255)`. Version 4 also stores `IsConstant` on members. Cached Path concatenation uses that
  flag so `path.RemoveExtension + "_output.jpg"` stays a path instead of `(path)_output.jpg`.
  Runner invalidates a cache when any sibling `.strict` file in the entry directory is newer.
- `NativeImageLoadProcessSavePipeline` passes for a fresh compile and for the cached binary.
  Non-slow `Strict.Tests`: 122 passed. `RunAdjustBrightness` passes. The Slow allocation budget
  test is still over its limit and is not a functional blocker.
- Validator walks list elements, counts mutable reassignment nested in `for`/`if`, and ignores
  implicit loop `index`/`value`. Double parentheses are rejected only when the called method has
  a single list parameter, so `NumberSummer((1, 2, 3))` stays valid (`from` also takes `logger`).
  `ConstantCollapser` keeps locals that are used inside `for`/`if`. `GcdCalculator` uses `for 100`.
  `Pixel.blue` is used. `MemoryPressure` assigns `values = values.Add(index)`. Empty `List` `from`
  builds a real list. Calls written as `Type.Method(...)` keep that prefix in expression text.
- CLI runs that now pass: `GcdCalculator`, `NumberSummer`, `NumberStats`, `Pixel`,
  `MemoryPressure` (`allocated numbers: 20000`), `Validators/ValidateDemo`, `TestRunner/TestDemo`.
  Non-slow validator tests: 51 passed. No production C# layer has been replaced.

Next: rerun the example suite, then Phase 1 package children and lookup. Package loading alone
does not complete self-hosting. The Slow allocation budget test is still over its limit.
## Verified checkpoint — 2026-10-09

Phase 0 base types are 100% verified across all base types (`Boolean`, `Number`, `Text`, `List`,
`Dictionary`, `File`, `Directory`, `Range`, `Character`, `Error`, `Any`, `Mutable`).
All tests pass in `Examples/BaseTypesTest/BaseTypesTest.strict` and `DirectoryTests.strict`,
running both from source (HLR inline test runner) and precompiled `.strictbinary` on the VM.

Key fixes completed:
- Native Directory: added `DirectoryEvaluator` for Exists/Create/Files, unified Path/Text argument
  handling via `FileValue.TryGetPathText`, fixed `Method.IsTestExpression` parsing classification,
  and fixed static method dispatch in VM.
- Native File: made stream opening lazy in `NativeFileRegistry` so `File("nonexistent")` does not
  create empty files on disk and `File.Exists` evaluates accurately.
- HighLevelRuntime: `Interpreter.GetFromConstructorValue` now handles primitive/wrapper types
  (e.g., `Name` with `TypeKind.Text`) returning primitive-backed instances instead of complex
  `ValueTypeInstance`.
- Validators: `ConstantCollapser` now handles `Boolean to Text` and avoids collapsing mutable variables
  that are reassigned prior to assertions.

Pre-existing failures (unchanged, verified independently): 6 Transpiler tests and
`RunAdjustBrightnessAllocatesBelowHalfMegabytePerRun`.

Next: Phase 1 — Package loading in Strict (`Language/Package.strict` loading `.strict` files via
`Directory.Files` and `File.ReadLines`), followed by parsing expressions and methods.

## Verified checkpoint — 2026-10-08

The conversion is **not functionally complete**. Parallel Strict implementations exist, but
many are line-level subsets. No production C# layer has been replaced; native I/O alone will
not bring all ten phases to 100%. File-count completion is not self-hosting completion.

- Restored directory entry points: `Examples/BaseTypesTest` resolves its same-named source
  file and loads sibling types through the existing package loader (including trailing slash).
- Fixed standalone bytecode execution of this package: preserve generic List identity,
  initialize intrinsic Any before generic reconstruction, bind constructor arguments by
  serialized parameter names, and restore collection/text member scopes from stored values.
- `RunBaseTypesTestPackageFromDirectory` covers both directory spellings and launches a fresh
  process for `.strictbinary` execution. Fresh-process checking caught an Any lookup failure
  hidden by shared parser caches in the test process.
- Verified `Language/Parser.strict Examples/HelloLogger.strict`: file read and console output
  work under the bootstrap VM; Members contains logger and Methods contains Run.
- Resolved LanguageServer notification merge remnants: use required constructor arguments,
  remove unused duplicate notification helpers, retain extended payload fields and TestState.
  All 44 LanguageServer tests passed; no unresolved Git entries remained after resolution.
- No new Strict files or replaced C# files in this checkpoint. Phase 0 still needs explicit
  base-type assertions; its greeting/arithmetic/list smoke test is not full base-type coverage.

Next: verify native Directory/File behavior in both interpreter and VM, then implement local
package loading in Strict. `Language/Package.strict` currently only declares Name/Children.
Full AST parsing, evaluator/codegen parity, binary serialization, and pipeline integration
remain substantive work after the base features; async/HTTP/reflection remain deferred.

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
| `Boolean` | `Boolean.strict` | ✅ `TestBoolean` | 100% |
| `Number` | `Number.strict` | ✅ `TestNumber` | 100% |
| `Text` | `Text.strict` | ✅ `TestText` | 100% |
| `List` | `List.strict` | ✅ `TestList` | 100% |
| `Dictionary` | `Dictionary.strict` | ✅ `TestDictionary` | 100% |
| `File` | `File.strict` | ✅ `TestFile` | 100% |
| `Directory` | `Directory.strict` | ✅ `DirectoryTests.strict` (Exists, Files, Create) | 100% |
| `Range` | `Range.strict` | ✅ `TestRange` | 100% |
| `Error` / `ErrorWithValue` | `Error.strict`, `ErrorWithValue.strict` | ✅ `TestError` | 100% |
| `Character` | `Character(strict)` | ✅ `TestCharacter` | 100% |
| `Any` | `Any(strict)` | ✅ `TestAny` | 100% |
| `Enum` | `Enum(strict)` | ✅ Smoke tested via base packages | 100% |
| `Mutable` | `Mutable(strict)` | ✅ `TestMutable` | 100% |
| `Iterator` | `Iterator(strict)` | ✅ List/Text iterations tested | 100% |

**Current state:**
- `Examples/BaseTypesTest/` exists with 3 `.strict` files (`BaseTypesTest.strict`, `TextHelper.strict`, `DirectoryTests.strict`)
- Tests: `RunBaseTypesTestPackageFromDirectory` and `RunDirectoryTestsUsesNativeDirectoryInTestsAndVirtualMachine` in `Strict.Tests` pass
- Phase 0 verification complete (100%): explicit tests for Boolean, Number, Text, List, Dictionary, File, Directory, Range, Character, Error, Any, Mutable.

**Target:** Complete. All base types verified via source validation, HLR inline tests, bytecode generation, optimization, and VM execution (both live and cached `.strictbinary`).

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
| 20 | `Package.cs` | Package = directory of types | `Language/Package.strict` | 🚧 60% (local loading; children/lookup pending) |
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
| `.strict` files created | 23 | 22 | 96% |
| Test methods written | 335 | 37 | 11% |
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
| `InstructionOptimizer.cs` | Abstract base + optimizer chain | ✅ `OptimInstruction` + `OpList` surface |
| `TestCodeRemover.cs` | Remove test-only instructions | ✅ `TestCodeRemove.strict` |
| `ConstantFoldingOptimizer.cs` | Fold constant binary ops | ✅ `ConstantFolder.strict` (simple prefix pattern) |
| `StrengthReducer.cs` | Replace expensive ops with cheaper | ✅ `StrengthReduce` + `IdentityRules` |
| `DeadStoreEliminator.cs` | Remove never-loaded stores | ✅ `DeadStore.strict` |
| `RedundantLoadEliminator.cs` | Remove duplicate loads | ✅ `RedundantLoad.strict` |
| `JumpThreadingOptimizer.cs` | Simplify redundant jumps | ✅ `JumpThread.strict` |
| `UnreachableCodeEliminator.cs` | Remove code after unconditional jumps | ✅ `UnreachableCode.strict` |
| `AllInstructionOptimizers.cs` | Compose all optimizers in order | ✅ `AllOptimizers.strict` (7-pass pipeline) |

**Also in C# (beyond plan's original 9):** CompactType, MethodInlining, ConstructorToField, LoopInvariant, MutableFieldMutation — deferred; C# chain still runs those for production.

**Phase 7 status (parallel Strict package):**
- Package `Strict/Optimizers` loads with line-level instruction list optimizers over `OptimInstruction` / `OpList`.
- Pipeline: TestCodeRemove → ConstantFolder → StrengthReduce → DeadStore → RedundantLoad → JumpThread → UnreachableCode.
- Demos green under VM: `OptimizerDemo`, `FolderTests`, `StrengthTests`, `DeadStoreTests`, `UnreachableTests`, `PipelineTests`.
- `StrictOptimizersConversionTests` 6/6 green.
- C# `AllInstructionOptimizers` remains bootstrap for production Runner (full instruction graph + advanced optimizers).

**Progress table:**

| Metric | Target | Actual | % |
|--------|--------|--------|---|
| `.strict` files created | 9 | **19** (core optimizers + helpers + 6 demos) | 100%+ |
| Test methods written | 59 | 6 VM demos + 6 C# conversion tests | ~20% |
| C# files replaced | 9 | 0 (bootstrap still C#; Strict package parallel) | 0% |

---

## Phase 8 — `Strict` (VirtualMachine + Runner) → Strict Package

**Goal:** Convert 6 C# files (1,611 lines) + 8 test files (107 test methods) to `.strict`.

This is the execution engine — the capstone of the self-hosting effort.

| C# File | Description | Status |
|---------|-------------|--------|
| `RegisterFile.cs` | Fixed-size register array | ✅ `RegisterBank.strict` (via CallFrame named slots R0..) |
| `CallFrame.cs` | Variable scope per method call | ✅ `CallFrame.strict` (names/kinds/numbers/texts) |
| `Memory.cs` | Registers + frame per VM | ✅ `VmMemory.strict` |
| `VirtualMachine.cs` | Execute bytecode instructions | ✅ `VirtualMachine` + `InstructionExec` + `ArithmeticExec` (line-level) |
| `Runner.cs` | Orchestrate parse→validate→compile→run | ✅ `RunnerPipeline.strict` (run expression/stored helpers) |
| `Program.cs` | CLI entry point — keep in C# or convert last | 🚧 Deferred (C# CLI stays) |

**Phase 8 status (parallel Strict package `Runtime/`):**
- Package `Strict/Runtime` loads with VmValue, RegisterBank, CallFrame, VmMemory, instruction model, VM, pipeline.
- **Executes:** LoadConstant/Variable, StoreConstant/Register, arithmetic (+−*/%), comparisons, Return, simple jumps.
- **Working under VM:** register set/get, frame bind/lookup, full binary expression run, store-then-load.
- Demos: `VmDemo`, `RegisterTests`, `FrameTests`, `VmTests`, `CompareTests` green.
- `StrictRuntimeConversionTests` 6/6 green.
- C# VirtualMachine + Runner remain production bootstrap (full instruction set, invoke, loops, plugins, binary cache).

**Progress table:**

| Metric | Target | Actual | % |
|--------|--------|--------|---|
| `.strict` files created | 6 | **17** (core + helpers + 5 demos) | 100%+ |
| Test methods written | 107 | 5 VM demos + 6 C# conversion tests | ~15% |
| C# files replaced | 6 | 0 (bootstrap still C#; Strict package parallel) | 0% |

---

## Phase 9 — `Strict.Compiler` + `Strict.Compiler.Assembly` → Strict Package

**Goal:** Convert 5 C# files (918 lines) + 1 test file (49 test methods) to `.strict`.

| C# File | Description | Status |
|---------|-------------|--------|
| `Strict.Compiler/Platform.cs` | Enum: Windows/Linux/MacOS | ✅ `Platform.strict` |
| `Strict.Compiler/ToolNotFoundException.cs` | Exception for missing NASM/gcc | ✅ `ToolInfo.strict` (messages/URLs; no throw) |
| `Strict.Compiler/InstructionsCompiler.cs` | Abstract compiler interface | ✅ `CompilerPipeline` + `InstructionsToNasm` |
| `Strict.Compiler.Assembly/InstructionsToAssembly.cs` | Bytecode → NASM x64 assembly | ✅ `InstrToAsm` + `InstructionsToNasm` + `EntryPoint` (line-level) |
| `Strict.Compiler.Assembly/NativeExecutableLinker.cs` | Invoke NASM + gcc/clang | ✅ `LinkerPlan` / `NativeBuild` + Strict `Process.RunTool` |

**Phase 9 status (parallel Strict package `Compiler/`):**
- Package `Strict/Compiler` loads with platform, register map (R→xmm), instruction emit, NASM body/entry, linker plans, **native tool spawn**.
- **Emits:** load const/var, store, return, add/sub/mul/div, compare with NASM `[rel …]` memory operands and real `\n` line breaks.
- **Process/Directory natives:** `Process.strict` + `ProcessResult.strict` + VM hooks via shared `NativeProcessRunner`; C# `ToolRunner` delegates to the same runner.
- **CompilerDemo end-to-end:** generate `add.asm` → `nasm` → `add.obj` → `gcc` → `add.exe` entirely from Strict.
- Demos green: `CompilerDemo`, `ProcessProbe`, `PlatformTests`, `EmitTests`, `LinkerTests`.
- C# NASM/MLIR/LLVM compilers remain production bootstrap for full pipelines; tool invocation is no longer C#-only.

**Progress table:**

| Metric | Target | Actual | % |
|--------|--------|--------|---|
| `.strict` files created | 5 | **17** (core + helpers + 4 demos) | 100%+ |
| Test methods written | 49 | 4 VM demos + 6 C# conversion tests | ~20% |
| C# files replaced | 5 | 0 (bootstrap still C#; Strict package parallel) | 0% |

---

## Overall Progress Dashboard

Counts verified on 2026-10-09. Counts include demos/tests and exclude root base types;
Language's root Method.strict is also excluded. Earlier totals of 51 files and 12% were stale.
The percentage below measures production C# replacement, not existence of parallel files.

| Phase | Project | Actual `.strict` Files | Current scope | C# replaced |
|-------|---------|------------------------|---------------|-------------|
| 0 | Base types verification | 3 | Base assertions verified; cached-runtime compatibility under retest | N/A |
| 1 | Language | 22 | Local package loading verified; full parsing/lookup pending | 0% |
| 2 | Expressions | 33 | AST models, classifier/tokenizer subset | 0% |
| 3 | Validators | 6 | Line-level validation subset | 0% |
| 4 | TestRunner | 7 | Simple assertion evaluator | 0% |
| 5 | HighLevelRuntime | 21 | Line-level evaluator subset | 0% |
| 6 | Bytecode | 30 | Line-level generation; ZIP serialization pending | 0% |
| 7 | Optimizers | 19 | Simplified instruction passes | 0% |
| 8 | Runtime | 17 | Partial VM; production orchestration remains C# | 0% |
| 9 | Compiler | 19 | NASM subset and tool invocation | 0% |
| **Total** | | **177** | **No phase verified fully self-hosted** | **0%** |

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
| `Directory.Exists` | 1 (Language) | 🔴 Critical | ✅ Added (VM + HLR verified 2026-10-09) |
| `Directory.GetFiles(path, pattern)` | 1 (Language) | 🔴 Critical | ✅ Added (`Directory.Files`, VM + HLR; Strict-level test pending) |
| `Directory.CreateDirectory` | 1 (Language) | 🟠 High | ✅ Added (`Directory.Create`, VM + HLR; Strict-level test pending) |
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
| Process execution (`Process.Start`) | 9 (Compiler) | 🟡 Medium | ✅ `Process.Find` / `Process.RunTool` + `NativeProcessRunner` (shared with C# ToolRunner) |

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
9. **No static-like types.** Never add `has dummy Number` (or any unused member) to get past
   "types without members must be traits". See the guide below. Enforced when files are loaded:
   `Type.ValidateMembersAndVariablesAreUsed` (Repositories and Runner) rejects private members
   and declared variables never used; single member value wrappers using `value` or `from` (Degrees, HashCode) are fine.

### Converting C# static classes and helpers

Strict has no static methods: every method runs on an instance whose members are its data. A 1:1
port of a C# static class gives a member-less type, which only traits may be. Decide per helper:

| C# shape | Strict shape | Example |
|----------|--------------|---------|
| Static class whose methods all take the same argument `X` | Type with `has x X`, methods drop that parameter; call `Pass(x).Run` | `ConstantFolder.Optimize(ops)` → `ConstantFolder(ops).Optimize` |
| Factory/builder creating instances of `T` | Methods on `T` itself, called type-qualified (`T.Create(...)`), no members used | `OpBuilder.LoadConstant(0, 5)` → `OptimInstruction.LoadConstant(0, 5)` |
| Helper used by only one type | Private method of that user type | `IdentityRules` → inside `StrengthReduce` |
| Pure function on a base value (Number/Text) | Method on the type that owns the data it needs, or the caller | `Compute(op, left, right)` stays in the pass that owns the ops |
| Pipeline/orchestrator calling several passes | Type holding the input (`has ops OpList`), chaining `Pass(ops).Optimize` | `AllOptimizers(ops).Optimize` |
| Constants-only class | `constant` members on the type that uses them | `InstructionNames` → constants on the instruction type |

Also drop the `Empty` = `X(0)` factory that only existed for the dummy member, and the
`for 0 / list.Add(...)` workaround: `mutable list = List(Mutable(T))` is an empty list.

---

## How to Run the Current Baseline

```bash
# Run all current C# tests
dotnet test Strict.Tests/Strict.Tests.csproj

# Run the multi-file package BaseTypesTest example
dotnet run --project Strict/Strict.csproj -- Examples/BaseTypesTest

# Run a single .strict file
dotnet run --project Strict/Strict.csproj -- Examples/SimpleCalculator.strict
