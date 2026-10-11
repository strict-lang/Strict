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
   Decided 2026-10-10: values stay immutable in the language, runtimes append in place to a shared
   backing and hand back the same memory (the previous version stays a valid shorter view), only
   changing an old version again copies (bad code). No mutable language features for speed.
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
Follow-ups: member symbol ids precomputed per type, VM constraint checks (from Phase A). Done in
the Phase C dev loop below: per-instruction cached access paths, lazy custom iterators (parameter
symbol ids were measured without gain).

Phase C7 progress (2026-10-10), Strict compiler (Bytecode/FileCompiler on the C# VM, Debug):
- Per Example run time: HelloLogger 130 → 111 ms, FizzBuzz 520 → 281 ms, ProcessProbe 850 →
  457 ms; the Slow differential over 20 Examples 9 s → 4 s. ZIP CRC32 was 75% of FizzBuzz (402
  bytes): bitwise XOR per bit (~320 loop iterations per byte) is replaced by nibble XORs with a
  256 entry nibble table (`Bytecode/Crc`, ~80 per byte). Linking stops at a fixed point.
- New `-profile` CLI option: the VM records inclusive time and calls per invoked method and prints
  the top 20 (`VirtualMachine.Profile`). First findings on ResolveCheck (Bytecode folder):
  `List(Text).in` ran its Strict loop 107,659 times (now a VM native, like Length), and the
  tokenizer made two method calls per character (now inlined into WordLength). ResolveCheck on
  Bytecode 4.4 → 2.4 s. Next hotspot: every line is tokenized about 4 times (TypeShape scopes,
  BodyParser); tokenize once per file.
- Remaining costs: inference over all base files (~100 ms per run, rebuilt for every file),
  linking recompiles used base types each round (ProcessProbe ~200 ms), CRC still ~150 ms.
- `Number.Floor` and `List.Index` run as VM natives (Floor ran its Strict body 45,295 times per
  FizzBuzz compile): FizzBuzz 263 → ~206 ms, CRC 154 → ~72-86 ms. Method headers are still
  tokenized twice (MethodScope + ParameterScope, ~10% of a run), left as is.
- Finding, needs a decision: every `List(Number)` uses the flat float32 backing, so integers
  above 2^24 change silently (a CRC table entry 1996959894 reads back as 1996959872 in the
  interpreter, the VM path happened to differ). Crc avoids storing 32 bit values in lists; the
  general fix (double backing, or float32 only for numeric data types like ColorValue) changes
  memory numbers of the C1 work and is left for the user to decide.

Phase C5 result (2026-10-10): no test result cache needed. Cached binaries skip inline tests
entirely and a source run only tests the changed main type: FileCompiler from source spends 29 ms
of 336 ms in tests (load packages 118, generate 53, optimize 25, run 93).

Phase C6 result (2026-10-10), native output correctness before speed:
- Before: the default MLIR backend built 1 of 14 numeric Examples (invalid IR for if/else, loop
  bodies emitted once, SSA values used across joins, `logger` passed as a number argument), the
  LLVM backend built 6 but printed wrong results (AreaCalculator perimeter 110 instead of 30,
  texts as empty lines), NASM silently dropped `%`.
- Now MLIR keeps variables in stack slots when a function branches or loops (LLVM promotes them
  back to registers, verified: the loop becomes phi nodes), emits real count and range loops
  (both directions, index/value restored), jumps to the matching JumpEnd, passes instances as
  their number members (constructor arguments by member name, defaults like the VM) and gives
  overloads unique symbols. Windows without CRT: `_fltused`, an `fmod` for `%`, no dynamic allocas
  around prints (they needed `__chkstk`). Texts, lists and Boolean prints throw
  NotSupportedByBackend instead of producing wrong executables. NASM implements `%`.
- Slow test `NativeExecutableRunsLikeVirtualMachine`: 10 Examples print the same natively as on
  the VM. 10M iterations of `sum = sum + index % 2`: VM (Debug) 10.1 s, native 0.34 s, 2,560 byte
  exe.
- Open: native number printing only matches the VM for integers below 1e7 (Windows prints integer
  digits, Linux printf `%g`), Strict prints `2.9999994e7` and shortest round-trip fractions. NASM
  and LLVM backends still miscompile (duplicate labels, texts); delete them once MLIR covers
  lists and texts (D6).

Phase C dev loop (2026-10-10), `Strict.exe Compiler/CompilerDemo.strict` from the cached binary,
Debug, best/median of 15: wall 218/248 → 129/135 ms, Loading cached 30/31 → 13/14 ms (962 → 496
KB), Run 120/146 → 52/56 ms. Native links run binutils `ld` directly (the gcc driver started
collect2 and ld: 66 ms vs 20 ms), tools share Strict's console (a new hidden console per tool cost
~8 ms), a multicore JIT profile (`Strict.jitprofile`) compiles last run's methods in the background,
timing logs are culture-invariant (no ICU culture init), cache checks enumerate each folder once,
`[GeneratedRegex]` replaces Reflection.Emit compiled regexes, NameTable indices and Type caches are
lazy, generic stubs skip the TypeParser and zip entries are read in one call. Fixed on the way:
NativeProcessRunner returned empty output while the thread pool was busy (NCrunch). Left: Run is
mostly nasm, ld and add.exe (~35 ms) plus first-call JIT, the VM itself needs ~0.2 ms; EventSource
off (~11 ms, a Defender ETW session forces the runtime manifest), a ReadyToRun publish (~45 ms) or
NativeAOT (~100 ms, needs AOT fixes) would cut startup further. EventSource is now off (A/B -11 ms).
VM: parsed variable access paths are cached on the instructions instead of per-VM string
dictionaries: BenchBrightness 320x180 Run 482/498 → 436/450 ms (Debug, interleaved). Measured and
rejected: saving only the registers a callee writes (CPU samples blamed RegisterFile copies for 68%,
A/B showed no change) and symbol ids for parameters. Release runs it 19% faster than Debug (350 ms).
Lazy custom iterators: a loop without custom variables or a nested loop that never reads `value` or
`outer` (bare words binding to the value included) counts to the iterator's `Length` instead of
calling `for`, in C# BinaryGenerator and Strict LoopCodegen alike. BenchBrightness no longer builds
57,600 Vector2 (Size.for was 92 ms): Run 350/355 → 244/249 ms Release, 429/435 → 320/326 ms
Debug, allocated 48 → 27 MB, 121 → 103 instructions (interleaved, same VM). Next: the Process loop
body itself (~150 ms beyond its two calls, ~2.6 µs per pixel) and the 27 MB still allocated.

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
All instruction kinds encode through one payload rule (bytes, negative numbers point at names),
`MemberEntry` writes members/parameters with `Set` initializers; the HelloLogger type entry is byte
identical to C# except the source line flag. Tree codegen in Strict: `CodeBlock` (instructions, next
register, next jump id, scope), `ValueCodegen` (literals, variables, binary operators, if
conditions, types), `MethodCodegen` (declarations, reassignments, return, if, for, logger.Log,
inline tests removed), `TypeCodegen` (members, constants, parameters, methods) and the program
`FileCompiler` (source file to `.strictbinary` bytes). Slow differential
`StrictCompiledExampleRunsLikeCSharp`: HelloLogger, NativeArithmetic, NativeConditions and
NativeLoop compiled by Strict give the same output and return value on the C# VM as the C#
compiled binaries. Fixed on the way (each with a test): constants could not call methods of other
types (`constant X = Other.Make(1)`), a list element could not be a bracketed conditional,
`constant Invalid = Error` failed on the VM (`List(Stacktrace).from(list)` invoke), constant
folding and strength reduction treated `list + 0` as number math (`x + 0`, `x - 0`, `x * 0` now
need proven numbers), package dependency detection took `"Examples/Sum.strict"` inside a text as
a package reference (Compiler then resolved `InstructionType` to Examples/InstructionType), and
both printers dropped the brackets a conditional needs among several arguments or list elements
(`f(1, (a then b else c))`). Open: source lines in Strict written instructions, method overload groups,
else/else if, method calls (Invoke), lists, member access, `Optimizers/StrengthReduce.strict`
parity, and parser papercuts found while writing the codegen (a parameter named `value` silently
collides with the implicit value, `Method(0)` on a parameterless method result is parsed as a call
argument, `x to Number` two calls deep and `a then b else c` followed by more arguments fail).
Then: Invoke for own methods, constructors, static calls (`Directory.Exists`), base type methods
(`text.Length`, `number to Text`, `not`), field loads (`where.exitCode`), Range loops; `CallTarget`
builds the invoke signatures (full type names, parameter names from `KnownTypes.parameters`).
`FileCompiler` links used base types like C# does: invoked `Strict/X.Method`s that the VM does not
handle natively are compiled from `X.strict` into `Strict/X` entries (3 rounds deep). 15 of the 19
runnable Examples compiled by Strict run like the C# binaries (HelloLogger, NativeArithmetic,
NativeConditions, NativeLoop, Greeter, Fibonacci, AreaCalculator, SimpleCalculator,
TemperatureConverter, GcdCalculator, FizzBuzz, AutofilledMutable, Pixel, DirProbe, ProcessProbe).
Missing: generic list methods (Sum, Add, Length on List(Number) for Sum, NumberStats, NumberSummer,
MemoryPressure), list literals, else/else if. C# fixes found on the way (each with a test): a
method call left of a nested binary was loaded as a variable named after its text, a `for` loop
ending a Text method returned a list instead of the concatenated text, and constant lists of
computed elements (`(Range(0, 1), Range(1, 3))`) were read as constant data (new
`Value.ConstantData`). The name table of a type entry prefills the short type name (deduplicated)
like the C# NameTable. Error quality gaps seen: VM errors show only the innermost Strict frame,
codegen errors in Strict (Error values) silently become garbage bytes, and the C# generator
inlines whole constant trees (`ValueCodegen.Plain`) until it runs out of 64 registers.
Then: constant list literals, empty `List(T)` construction, `list.Add(x)` as WriteToList, a
trailing `for` in a Number/Text method sums its values (`forResult`, like C#), and the linker
instantiates generic base types (`Strict/List(Number)` from List.strict with Generic → Number).
All 19 runnable Examples compiled by Strict now give the same output and return value on the C#
VM as the C# compiled binaries (Slow `StrictCompiledExampleRunsLikeCSharp`, Sum with program
numbers). C# fix on the way: a member constant like `BodyParser(..).Block(1, 1)` was typed by its
first call (BodyParser) instead of the whole expression.
else/else if work like C# (condition flag set after the then body, JumpToIdIfTrue over the else
body), the new Examples/Grade.strict covers them in the differential (20 Examples).
Unsupported code now fails loudly (`FileCompiler` refuses bytes outside 0..255, e.g. from codegen
Error values), value-less enum constants (`constant Add`) get their numbers like C#. Codegen
coverage over Examples (top folder): 34 of 41 files compile; open: selector if
(`if operation is` cases), `* value` reductions, list-returning loops with a filter, if/else used
as a value. Found on the way: reading a missing file through the VM or interpreter created an
empty file (NativeFileRegistry opened reads with OpenOrCreate, now FileNotFoundException with a
test), which let the linker litter the repo root with empty `.strict` files for same-package
types; the linker now only links types whose source exists.
Next for D3: compile the compiler packages themselves (same-package types need their package
prefix and inference from the package folder, today they crash), source lines in instructions.
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

D4 result (2026-10-10): Strict-compiled main types of the 20 differential Examples went from 429
to 402 instructions, the C# optimizer pipeline produces 404 (all 11 C# passes run, only folding,
constant stores, returning branches and redundant loads change these Examples). Done at the tree
level where that is less code: literal assignments become `StoreConstantToVariable`, constant
members with literal values are inlined and number arithmetic folds (`Bytecode/ConstantFolding`),
a then branch ending in `return` needs no skip-else flag or JumpToIdIfTrue (Grade 55 → 46, C# 50).
Instruction level: `Bytecode/LoadReuse` mirrors RedundantLoadEliminator with register remapping,
decoding read/written registers per instruction kind. FileCompiler on FizzBuzz 206 → 185 ms
(fewer bytes to write and checksum). Found on the way: package types were sorted without the types
used in constant values (`constant A = Other.Create(Kind.X)`), the load order was random and a new
Bytecode type broke the whole package; TypeLines now records them. A local named `from` (or any
operator word like `and`) crashed the expression parser with "Stack empty", such names are now
rejected like keywords (CannotUseKeywordsAsName). The C#
RedundantLoadEliminator ignores StoreConstantToVariable between two loads of the same variable
(Strict's LoadReuse treats every store as a barrier).

D5 progress (2026-10-10): `Runtime/` is rewritten, the old BytecodeInstruction sketch is gone. The
Strict VM (`Machine`) runs the TypeEntry/MethodEntry/InstructionEntry lists produced by the
Strict compiler: `VmValue` (kind, number, text, items), `Locals`, `Frame` (registers, locals, the
condition flag and loop state as `#` locals), `PayloadDecoder` (small numbers, IEEE doubles,
texts, lists, 7-bit ids), `Storage`, `Operations`, `Flow`, `Looping` (count and range loops in
both directions, lists, texts, index/value restored), `Invocation` (Invoke payload) and `Natives`
(constructors by member name with defaults, to, Length/Count, Boolean logic, in/Index, Exists).
`Runtime/Execute.strict <file> <root>` compiles a program with the Strict compiler and runs it on
the Strict VM. Slow test `StrictVirtualMachineRunsLikeCSharp`: 19 Examples print the same as on
the C# VM, Process.Find/RunTool/Run/OperatingSystem are host hooks (the Strict VM calls the
host's Process, native on the C# VM) and instances print their automatic text like the C# VM.
Speed (Strict VM inside the C# VM, Debug, including the Strict compiler): Fibonacci 249 ms,
FizzBuzz 292 ms, MemoryPressure 15.3 s, because every list append copies the immutable `items`
(20,000 appends are O(n²), the C# VM appends in place). Open: in place list appends, program
arguments (Sum), loading `.strictbinary` files (moved to D7, needed when the run stage switches to
the Strict VM: stored ZIP, name table, members, methods and a slot spec per instruction kind).
C# bugs found on the way (each with a test): `ValueInstance.Equals` treated a type instance whose
`number` member is 0 as None, so the implicit instance of a method call was lost; the interpreter
left constructor members without a from parameter (like `logger`) uninitialized
(NullReferenceException); type lookup ignored the packages a package declares (Runtime found
`Examples/InstructionType` instead of `Bytecode/InstructionType` once Examples was loaded) and the
dependencies were only assigned after parsing the package. Strict papercuts still open:
`Method(not x)` (sole unary argument) is not parsed, `(a then b else c).Next` is parsed as a list.
Then (2026-10-10): list appends reuse memory (C3 decision). `ValueArrayInstance` versions share one
backing list, `list + element` on the newest version appends in place in the C# VM and the
interpreter, older versions stay valid shorter views and copy only when changed again. Strict VM
MemoryPressure 21.2 → 1.75 s (Release, allocated 15.9 GB → 297 MB), the Strict compiler allocates
23% less on FizzBuzz (time unchanged). Program arguments: `Runtime/Execute.strict <file> <root>
[numbers]` binds them like the C# Runner (a Run with that many parameters, else one list
parameter), Sum joins `StrictVirtualMachineRunsLikeCSharp` (20 Examples). Fixed on the way: a
camelCase member named like a type took a type of any loaded package (`constant limit = 10` in
Examples became Language/Limit once Language was loaded, a load order race in
RunAllTestsForAllStrictFilesInThisRepo), now only the own package, its parents and top level
packages count. Cost audit (Release): the 20,000 appends are ~1% of MemoryPressure's bytes on the
Strict VM, each append is 6 Strict VM steps or ~87 Strict calls on the C# VM (66.6 µs, 14.3 KB).
Directly on the C# VM it is 13 ms cold and 1.5 ms warm (77 ns per append), natively 0.7 ns. The C#
VM no longer allocates per call (argument arrays, an empty disposables List, a Dictionary per loop
start, a closure per subtraction, native Text argument arrays, dotted ListCall paths split again
per access): MemoryPressure 297 → 150 MB, FizzBuzz compile 22.2 → 11.3 MB, time unchanged (GC was
1.4%). Fixed on the way: the entry call `X((1, 2), "ab").Method` put both arguments into the list
member (the `X(1, 2, 3)` shorthand fired whenever the arguments were not a single list), and
`let target = …` plus `numbers(target) = 5` silently changed nothing: the validator never visited
assignment targets (false UnusedMethodVariableMustBeRemoved), DeadStoreEliminator dropped the
element write and the `let` (the index only appears inside the store name), and the C# VM stored
constants into a variable literally named `numbers(target)`. Still open: the Strict VM
(`Runtime/Storage.strict`) has no element writes at all, a C# VM element write that cannot be
resolved (index out of range) still falls back to such a variable instead of failing. Fixed:
after `mutable result = numbers` the element write `result(0) = 5` changed `numbers` too (C# VM
always, interpreter for parameters). Both compilers now emit `CopyList` before the first element
write to a list the variable does not own (before the loop when the loop does not share it) and
the interpreter copies such a list once into a Mutable list. With that the Strict VM stays
functional and still got faster (MemoryPressure on the Strict VM, Release: 1646 → 1255 ms, 138.9 →
117.4 MB): `Locals.Replaced` copies natively and writes one element instead of rebuilding the list
in a Strict loop, one `#loop` state (begin, length, iteration, outer state) replaces the backward
`LoopBeginOf` scan and the `#loop`/`#iteration` text keys, Storage/Operations/Flow get the
instruction instead of fetching it again, and `NextWithRegister`/`NextWithLocal` build one frame
instead of two. From Examples, as first measured: 1.75 s and 297 MB at the start of this round, now
1.36 s and 120 MB; FizzBuzz through the Strict compiler 330 ms and 22.2 MB, now 311 ms and 10.7 MB.
The rest of the gap to the in-place prototype (0.77 s, 89 MB) is one new Frame and
Locals per step, closing it needs the runtime to reuse an instance whose old version is dead. Next,
ranked by measured effect: ReadyToRun as one version bubble (FizzBuzz compile 348 → 195 ms), one
store and an in-place append for `x = x.Add(y)` in both compilers, and the Strict compiler
tokenizing, reading and linking each file once (1,602 tokenizations for 1,073 lines, 54
LinkedType calls for 12 types).

D6 progress (2026-10-10): `Compiler/NativeCompiler.strict <file> <root>` is a native compiler
written in Strict. It compiles the program with the Strict compiler (`FileCompiler.Compiled`),
emits MLIR from the TypeEntry/InstructionEntry lists (`MlirModule`, `MlirFunction`,
`MlirStatements`, `MlirLoops`, `MlirCalls`) and runs mlir-opt, mlir-translate and clang through
`ToolRunner`. Same design as the fixed C# backend (C6): registers, variables and instance members
live in stack slots that LLVM promotes, JumpEnd labels become blocks, count and range loops are
real loops, instances are their number members, every reachable method is a function, prints use
printf (links the C runtime), constants are exact IEEE hex literals and `main` returns the Run
result as exit code. Texts, lists and other calls fail with an "unsupported:" message. Slow test
`StrictNativeCompilerRunsLikeVirtualMachine`: 7 printing Examples run like the VM,
NativeArithmetic/Conditions/Loop exit with 20/30/45. Open: lists, fraction printing like the VM
(printf `%g`), Windows without the C runtime, Run(numbers) from argv, then delete the old
line-level SourceCompiler/NASM path.
Bugs fixed on the way (each with a test): the VM ran `list.Count(x)` as Length; a summing loop with
a filtering `if` aggregated on every iteration (C# generator; the Strict compiler's summing loops
now add inside the then branch too, `MethodCodegen.SummedLoop`); a test line comparing with `>` was kept as code (recursion and
register exhaustion); text literals with two spaces were rejected; BinaryGenerator now names the
method that runs out of registers. Strict papercuts: an implicit-instance method call counts as
constant (`let x = OwnMethod` must be `constant`), the inner `value` of nested loops keeps the
outer type, `List.Reverse` (`outer.value`) does not run on the VM.
Texts (2026-10-11): 14 of the 20 runnable Examples build natively with the Strict compiler (11
before), Grade, Greeter and FizzBuzz joined the Slow differential (10 cases, 5 s). A text value is
its pointer bit cast into the existing f64 slot, so slots, signatures and copies stay unchanged.
`MlirKinds` infers each register's kind (Number, Text, Boolean) from its writer (constant, variable
store, parameter, member, call return type, operator), `MlirTexts` prints texts with `%s` and
Booleans as `true`/`false` (the VM's text, not 1/0), joins with snprintf into a malloc'd buffer and
converts `to Text` with `%g`, `MlirConstants` emits text constants as globals. Instances now carry
their Text and Boolean members (`ValueMembers`): Greeter printed `Hello, (null)!` before, silently
wrong. Texts are never freed (ponytail: fine for short programs, arena or ownership later); other
operators on texts and text member defaults (`mutable text` without argument) fail with
"unsupported:". Exe size: HelloLogger 112 KB, Grade/Greeter 141 KB, FizzBuzz 146 KB (C runtime
linked). Seen once: mlir-translate (msys64 ucrt64) crashed with an access violation while four
native builds ran in parallel, the same Grade build passes alone. Left: lists (MemoryPressure,
NumberStats, NumberSummer), `Text.Length` (AutofilledMutable), Directory/Process natives (DirProbe,
ProcessProbe), fraction printing like the VM, Run(numbers) from argv.

D7 progress (2026-10-10): measured the Strict compiler (`Bytecode/FileCompiler`) on the 41 programs
with a Run method outside Examples. Before: 33 produced bytecode, 8 crashed (`-1` literals parsed as
`Negate` reached the binary operator path; a non-literal constant returned a bare Error where an
InstructionEntry was expected). Both fixed (`ConstantFolding.Negated`, unsupported constants become
an instruction with an Error kind), now 33 compile and 8 stop with "unsupported code". None of the
33 runs like C# yet: only root base types (`Strict/X`) are linked, types of the program's own
package and of declared dependencies are not resolved, so `BytecodeInstruction.ReturnOp(0)` becomes
a variable load and `ExpressionParser(...)` a call on the main type. Next: package-aware inference
and linking (folder files + declared dependency packages), then a Slow differential test over all
41 programs like `StrictCompiledExampleRunsLikeCSharp`.
Then: package-aware compilation. `SourceFiles` collects the program folder, the packages it
references (`Bytecode/InstructionList` tokens, transitively) and the root types; `CallTarget`
resolves short names to full names (own package first) and `FileCompiler.Linked` links package
types (same-package entries get short names like C#), deeper (8 rounds). `>=`/`<=` (compare plus
equal false), `and`/`or` (short circuit through a `logical{id}` variable) and `is in`/`is not in`
(an `in` call on the list) are generated by the new `OperatorCodegen`; jump ids are taken after a
condition's operands. A failed compilation now names the first unsupported type and method.
The differential tests loaded binaries into the shared source package, so a C# binary only ran
right when its source package was parsed earlier in the same process (and the Strict-compiled
side failed the same way, both sides equal). They now load self-contained like the CLI, which
needed the loader to always provide the base Text type.
Next rounds: conditional values, constants of other types and own enum constants folded through
the inference, list indexing as IndexCall, and a C# VM fix (parameters are bound after the implicit
instance members, a type-name call read the caller's member instead of its own parameter of the
same name, which made the Strict compiler write stores one register off). Now 18 of the 41
programs outside Examples run exactly like their C# compiled versions (all Optimizers tests, most
Bytecode tests, Linker/Platform tests, Evaluator/RuntimeValue tests), checked by
`StrictCompiledProgramRunsLikeCSharp` with 38 cases. Open: Compiler/NativeCompiler, Runtime,
Expressions, Validators, Language parser and ImageProcessing still hit unsupported code; some
HighLevelRuntime and TestRunner programs fail at runtime (missing linked methods, recursion in
Platform.Current, an unresolved implicit member).
Then: built lists (`listResult{id}` with InvokeWriteToList), member types and traits linked like C#,
inherited member methods (`Length` inside Text methods), list Remove, IndexCall-aware LoadReuse, and
loops aggregating into lists (`Bytecode/LoopCodegen`, split from MethodCodegen: sum for Number/Text,
list for plural and `List(...)` return types, filtered by a trailing `if`). Text literals are unescaped
like C# `Text.Unescape`, and only Text/List/File `Length`/`Count` are left to VM natives
(`Range.Length` is linked). Now 26 of the 41 programs run like C# (`StrictCompiledProgramRunsLikeCSharp`,
46 cases, path arguments resolved from the repository root).
Next round: all 41 programs now compile with the Strict compiler. Single elements passed to list
parameters are wrapped (C# `WrapSingleListElements`), own methods win over type names, own
constant lists are indexable, Range-typed loop iterators load Start/ExclusiveEnd, constants are
inlined at use sites like C# (any constant expression, constant data also written as member Set
values), `from` parameters skip generic members (C# `CreateFromMethodParameters`, generic types
detected with the same first-three-lines rule), and a bare word inside a `for` body binds to the
loop `value` first (C# `TryParseForBodyValueMethodCall`, loops declare `outer`). C# fixes found on
the way: in-place list appends/removes only on lists the variable owns (`mutable copy = other`
then `copy.Add(x)` changed `other` too, which cut `SourceFiles.Folders` after one level), the VM's
list minus removed only the first match (interpreter removes all), text arguments mentioning
`Generic` were parsed as generic type names, and VM errors now list the calling methods. 28 of
41 programs run like C# (ValidateCheck and Runtime/Execute added, 48 differential cases); the
Strict-compiled FileCompiler runs and writes a binary for HelloLogger.
Then: C# `Text.Unescape` is a single pass (`"\t"` was backslash+tab) and the Strict `Unquoted` splits
at escaped backslashes, the VM finds a type's own `to Text` by type name (full names depend on the
binary's folder), several arguments for a single list parameter are grouped like C#
(`ByteEncoder(MagicByte, FormatVersion)`), and `is in`/`is not in` test lines are stripped.
36 of the 39 tracked programs now run exactly like C# (TestInterpreter has no Run method,
BenchBrightness is untracked): RoundTrip, ResolveCheck, TypeReport, ValidateCheck, SourceCompiler,
CompilerDemo, EmitTests, NativeCompiler, Runtime/Execute and Bytecode/FileCompiler included. The
Strict compiler compiled by itself produces byte-identical HelloLogger bytecode to the C#-compiled
Strict compiler (Nightly case, the heavy self-hosting cases take ~19 min). Open: Language/Parser
(types only used as parameters, `Strict/Path`, are not linked), ImageProcessing (custom `for`
iterators of `Size`, nested loop aggregation, `outer.index`), `outer` as a local name crashes the
C# parser with a NullReferenceException.
Then: ImageProcessing/AdjustBrightness and ProcessImage run like C#. Loops over a type with its own
`for Iterator(...)` method call it first (`LoopCodegen.Iterated`, C# `GetLoopIteratorExpression`),
iterator methods aggregate into a list of their element type (`TypeCodegen.BodyType`), the inference
keeps `Iterator(Vector2)` (`HeaderTokens.TypeName`, `KnownTypes.ElementType`) so the loop `value` is a
`Vector2` and `Size.for` records `Strict/Iterator(Vector2)` like C#, generic implementations of traits
are not linked (C# never emits them), a nested `for` as the last statement aggregates into the same list,
`outer.index` is a variable path load, assignments store to the full target text
(`image.Colors(index)`), 2D list indexing flattens to `x + instance.Size.Width * y` like C#
`ListCall.CreateFlattenedIndex`, and `has x = 1` members keep their initial value (`ColorValue.Alpha`).
The Strict compiler's `Add`/`Remove` changes a list in place only when the variable owns it like C#
`OwnsList` (a variable declared from another variable, parameter or member gets a changed copy,
`CodeBlock.Sharing`/`Owns`).
Then: lists behave as values in both generators. A local list variable owns its list only when it
was declared with a new list (literal, built list or operator result) and was not shared since:
before each statement C# `BinaryGenerator.Disown` / Strict `ListCodegen.Disowned` end the ownership
of variables reassigned from another value or used as a whole value (assignment, argument, list
element, return, loop iterator, right operand except the list searched by `is in`), loops are
checked as a whole first and aggregated loops treat their last line as shared. `list.Remove(x)`
returns the list, VM `list - other` removes each element of `other` once like the interpreter. Open:
`Add` on immutable parameters and members still changes the caller's list in place, the interpreter
shares `Mutable` lists after the first `Add` (`firsts = result` in a loop sees later appends).
Cleanup round: Text natives (`Upper`, `StartsWith`, ...) are linked like C# (a bare `for` iterates
`value`, Text yields `Character` elements), the Strict VM intercepts them like the C# VM, a loop no
longer leaks `outer`/`value` into the following statements, member reassignments store the member
name like C# `MutableReassignment.Name`, and a type's own list member flattens 2D indexes too.
Open: leading Boolean method-call test lines (`"hello".StartsWith("hel")`) are kept as code.
Bootstrap fixpoint: the Strict compiler compiled by itself (FileCompiler.strictbinary) writes the
same bytes as the C#-run Strict compiler for all 66 tracked programs with a Run method, itself
included. Needed: a bare `Error` in any position (declaration value, return, then/else branch,
argument, last line) builds `Error.from(name, stacktraces)` like C# (was a load of the variable
`Error`), named after the declaration while its value is compiled, else after the method
(`CodeBlock.ErrorMarker` scope entry), and MethodCodegen is back to 15 methods. Open: the
stacktrace list stays empty (SyntaxNode has no line numbers, the compiler no file path),
`Error("text")`/`Error(value)` are not normalized like C# `NormalizeErrorArguments` yet, and C#
keeps a `mutable` declaration's name for later errors in that body (TryParseDeclaration returns
before resetting `CurrentDeclarationNameForErrorText`), the Strict compiler uses the method name.
Interpreter speed of the inline tests (`test Bytecode/TypeCodegen.strict`, median of 7, Debug):
TypeCodegen 1777 ms / 401 MB → 1293 ms / 82 MB, LoopCodegen 1690 ms / 369 MB → 1276 ms / 78 MB,
MethodCodegen 1041 ms / 222 MB → 754 ms / 50 MB. Bisect: the interpreter list value commits copy
only ~150 short lists per run, the growth came from the Strict compiler running 2-3x more
expressions (Grid test, ownership checks, Error positions) and from the number wrapper commit, which
checked `IsNumberLike` (a type compatibility walk allocating a closure) on every loop iteration
(+21-27% allocations). The check now runs once per loop (`outer` is still read per iteration, a loop
can change the member it resolves to), `DisposableValues` (interpreter and VM) no longer creates a List per call and Number loops fold each result instead
of keeping all of them. Next: `Text.Length` runs `List.Length` (`for elements / 1`), 3.3M of the
6.2M TypeCodegen expressions, the VM has native `Length`/`Count`; the context pool's
ConcurrentStack allocates a node per returned context (17 MB per TypeCodegen run).
Conversions: a value accepted only through `CanBeConvertedTo` (its `to` or the target `from(x)`,
like a `ColorValue` stored into a `Color` list element, in a list literal argument or as a parameter)
is wrapped in `Expressions/Conversion` (printed as the value), README typed collections convert. The
VM runs member initializing `from` bodies (`Method.InitializesMembers`, shared with the interpreter),
the Strict compiler converts reassigned values and own or constructor arguments
(`ListCodegen.Converted`) and its `from` methods return their type. AdjustBrightness logs and
compares its stored `Color` `to ColorValue`. Binary operator arguments convert too
(`colors + ColorValue(..)`), a list variable whose elements need a conversion becomes a `Conversion`
with a `For` over its elements (interpreter converts element by element, BinaryGenerator reuses the
list aggregation loop). The Strict compiler mirrors this in `Bytecode/ConversionCodegen` (`to` route,
explicitly typed `from`, operator and `Add`/`Remove` elements, element loops for list variables,
nested `Color(1, 0, 0)` is `Color(ColorValue(1, 0, 0))` like C#) and `KnownTypes.ParameterTypesOf`
knows the parameter types of other types' methods. Open: CompactTypeOptimizer stays parked (rounds
127.5 to 128, Alpha 0), return values are not converted, `to` methods returning a usable but not the
same type and a list member's `.Add` with a convertible element are only converted by C#.
Cost: BenchBrightness +34% run time and +15.8 MB allocated (two values per pixel go through
`ColorValue.from`). Merged 2026-10-11 for README-correct semantics, the speed is a Phase C item.
The conversion rule (`Converter`) now lives in `Expressions/TypeInference`, so the front end types a
converting reassignment as its target like C# (StrictInfersSameTypesAsCSharp, ImageProcessing).

Path to Strict without .NET (measured 2026-10-11). Today every stage exists in Strict but runs on
the C# VM: the Strict compiler compiling itself (FileCompiler.strict) takes 196 s and allocates
4.5 GB there (Debug). Leaving .NET needs a native Strict compiler, which the D6 native backend must
first be able to compile. Native survey over all 67 programs with a Run method: 15 build and run;
13 (FileCompiler, NativeCompiler, Runtime/Execute, Sum, ...) take arguments, `Run(path, root)` had
silently produced an empty exe and now fails with "unsupported:". Missing, in order of need:
1. Program arguments as Texts (`Run(path Text, root Text)`, `Run(numbers)`) from argv.
2. Instances as heap values (pointer in the 64-bit slot like texts): calls returning instances
   (`Registry.Empty`, `TypeEntry`), members that are instances or lists.
3. Lists: literals, `Length`, index, `+`/`Add` appending in place (C3 semantics), `in`, `Index`,
   loops over lists, lists of instances and texts. Dictionaries after that.
4. Text natives: Length, Substring, IndexOf, StartsWith, Split, Character, `to Number`.
5. Host natives through the C runtime: File read/write, Directory files/exists/create, Process
   run (mlir-opt, mlir-translate, clang), Error values with stack traces.
6. Stage 1: the native backend compiles FileCompiler and NativeCompiler; stage 2: those exes compile
   themselves, byte-identical output (fixpoint), and all Examples run like the C# VM.
7. Tests without C#: inline tests run by a natively compiled test runner (TestRunner/TestInterpreter
   is still small) or compiled into the binaries.
Only then can C# go: keep the C# projects as the bootstrap and reference for the differential
tests until stage 2 holds, record golden outputs, check in a native bootstrap compiler, then delete
the C# layers one by one (the plan's "C# replaced" column). LanguageServer and the VS Code extension
need their own Strict versions later.

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

E progress (2026-10-10): `Strict [run|check|test|build|decompile] <file|folder> [-options] [args]`,
exit codes 0 success, 1 failed, 2 wrong usage (unknown option, missing file), negative numbers are
program arguments, not options. `check`/`test` use the new `Runner.Check(runTests)`, which always
parses and validates (inline tests used to run only with diagnostics). Strict errors print only
type name, message and the clickable .strict locations; .NET stack traces only for non-Strict
exceptions or with `-diagnostics`. Files outside the Strict root load their folder as a package, so
error links point to the real file instead of `<root>/<Type>.strict`. `a is 1 and b is 2` fails with
`LogicalOperatorNeedsBooleans`, which explains the `is` precedence and the bracket fix. A package
loaded earlier (Bytecode) was dropped from the declared dependencies of the next package (Runtime),
so after a fresh build `InstructionType` resolved to `Examples/InstructionType` depending on test
order; declared dependencies now always include loaded packages.
E5: `.github/workflows/ci.yml` runs every test project except Cuda (needs an NVIDIA driver) with
Manual/Slow/Nightly excluded on windows-latest and ubuntu-latest (Slow needs mlir-opt, clang and
nasm, it stays a local run before commits). Verified on Ubuntu 26.04 (WSL): the checkout folder must
be named `Strict` (the base package name comes from it). Linux fixes: the checked-in ImageSaver.so
was built from the old 5 parameter `ImageSaver_Save` (rebuilt both .so files from the current
sources, glibc 2.29 or newer); test helpers re-copied the loaded plugin .so on every test because
Linux file copies truncate the write time to seconds, overwriting a mapped library crashed the test
host (copy only when the content differs); `Marshal.SizeOf`, `
` and `file:////tmp` URIs in
three tests. Open: the rare `CachedBinaryOlderThanRuntimeIsRegenerated` failure (2 in about 180
runs, regenerated cache not saved, cause not found yet). The LanguageServer runs
TypeValidator and ConstantCollapser like `strict check`, so validator errors become diagnostics.
README: "Command line" and "Rules worth knowing" sections (limits, `is` precedence, conditional
arguments, single-element lists, reserved names, test lines, package references, constants).

### Phase F — Hardening (continuous, ≈2 sessions final pass)
- Fuzz the parser with mutated `.strict` files (no crashes, only ParsingFailed).
- Thread safety of Repositories/package cache under parallel tests (AGENTS multithreading rules).
- Memory/time limits in VM (stack overflow detection, step limits) with clear RuntimeErrors.
- Binary format versioning + compatibility tests (old cache → clean regenerate).
- 2026-10-10 test isolation: RunnerTests no longer write, delete or re-time repo files (Color.strict
  rewrite, BytecodeInstruction.strict re-time, SimpleCalculator.asm and program-suite .strictbinary
  deletes, 4x4_output.png/test_image_output.jpg). They use temp copies or a fresh process on a runtime
  copy whose Strict*.dll write time decides cache freshness (used packages always map to repo folders).
  `CachedBinaryWithOlderVersionIsRegenerated` covers the InvalidVersion catch (passed at once, fails
  without it). C# Version 4 = TypeEntry.FormatVersion 4, BinaryTypeData.strict has unused Version 1.
- Review fixes: the program suite passed on a stale repo .strictbinary when the source run could not
  save it (binary held open: "not saved" is only logged), it now asserts the binary was rewritten.
  Runtime copies hold only the 20 files Strict.deps.json loads (1.3 MB, was 96 files/28.6 MB, which
  included Strict.jitprofile that fresh processes write: IOException). The NCrunch %TEMP% build is
  behind a named mutex (abandoned one handled), cold 5-7 s, warm ~1 s. NCrunch console run on
  Strict.Tests.csproj (cold and warm): 1310 fast tests passed, 5 allowlist ignores.

Parser fuzzing (2026-10-10): Slow `ParserFuzzTests` mutates all 299 .strict files 40 times (seed 7,
11960 parses, 3 s); a .NET exception (also inside ParsingFailed) or a parse over 5 s fails it.
Found 1 hang and 19 crash classes at 100 mutations/file, 20 root causes fixed (GetMemberType endless
loop, Stack empty in Binary, unwrapped constraint/generic test errors, Type.Dispose keeping
List(Type) alive, index bounds); seeds 7/13/21 at 400/file are green. Open: UnterminatedString,
CannotUseKeywordsAsName and other parse errors still only get wrapped into ParsingFailed.
Fuzz review: every file gets its own seed from its repo path, new or edited files no longer change
other files' mutations. 9 fixes got their missing fast test (`for in x` is now MissingVariableNameBeforeIn,
`if is`, `then else`, `(1, (2)`, empty parameter, unclosed `Range(`, `=` in member names, conditional
error line). Per-file seeds at 400/file found `for (five)` crashing (no comma); TryParseNumber took
`(5` as -75 (double to uint saturates), so `constant result = ((5)` parsed. Both fixed, seeds 7 and
13 at 400/file green (44 s each). Open: `logger.Log((5)` reports an argument mismatch, not the bracket.
VM limits (2026-10-10): endless recursion crashed `Strict test` (Debug) with a real .NET stack overflow
(every body caught and rethrew); `CallDepthExceeded` now passes bodies, lists the Strict call chain and
guards small thread stacks (~5.5 KB per Strict call in Debug). VM `StackOverflow` is an
`InstructionExecutionFailed` with file:line and `Deeper (255 times)` callers. Raw Overflow/OutOfMemory
(`Range(1, 3e9)`, `Length is 3e9` lists) become Strict errors at the line, BenchBrightness 271 vs 275 ms
(noise). No step limit: only `LoopEnd` jumps back, count fixed at start.
Review fix: a failure 120 calls deep (legal, limit 128) still crashed Debug `Strict test` (0xC00000FD,
110 was fine), each body rethrew it. Bodies now wrap an error once with all Strict callers
(`ListsCallers`) and let it pass. `for 1e12` and `Range(3e9, 3e9 + 2)` silently ran int.MaxValue or 0
times, `GetLoopBound` now fails in VM and interpreter naming value and line. The 256 KB thread test hit
NCrunch's stack guard (<480 KB left) and is gone; NCrunch green (606 HighLevelRuntime, 1314 Strict.Tests).
Type identity (2026-10-10): List(Color) was cached by simple names, a temp ImageProcessing copy's List(Color)
replaced Strict/ImageProcessing/List(Color) for the whole process; now keyed by full names and living next
to Color (List(Number) stays Strict/List(Number), binaries keep Strict/List(Color)). That exposed Language
passing List(Language/Variable) to the base Method: Language/Variable (a copy of root Variable) is gone,
MethodParser builds root Variables (ConstantCollapser folded `X(...).IsMutable` to its default). RunnerTests
unload temp packages, Package.Types is a snapshot, no static Any methods or lastType cache. Medians of 7
(plain folders): test TypeCodegen 1616 → 1590 ms, SyntaxParser 450 → 449 ms, Language.Tests 254 → 263 ms.
Review fix (2026-10-11): implementations were still added to a package by simple name, so List(Widget) of
two root packages or Dictionary(Color, Color) with Colors of two packages threw TypeAlreadyExistsInPackage;
now only the generic's cache finds them (parent package kept for FullName). Language/Method.strict is gone
again. Medians vs f-type-identity (7 interleaved): TypeCodegen 1594 → 1592 ms, SyntaxParser 453 → 448 ms,
Language.Tests (434 shared tests, 11 runs) 256 → 256 ms; 428 Slow RunnerTests/differential tests green.


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
  → nasm + ld → runs the exe and logs `Run returned 20`. The compiled exe exits with the result.
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
| 11 | `Variable.cs` | Variable | Root `Variable.strict` (`Language/MethodParser.strict` builds them) | ✅ 75% |
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
- ✅ **Language package `.strict` files** — TypeLines, NamedType, Parameter, Member, Expression, ConcreteExpression, ExpressionParser, TypeParser, TypeFinder, MethodParser, Context, Package, Type, Body, Parser + constants. Root `Method.strict` is data-only (`Name`/`Type`/`Parameters`) with root `Variable`s as parameters; parsing lives in `MethodParser.strict`.
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

**Also in C# (beyond plan's original 9):** CompactType, MethodInlining, ConstructorToField, LoopInvariant — deferred; C# chain still runs those for production.

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
- **CompilerDemo end-to-end:** generate `add.asm` → `nasm` → `add.obj` → `ld` → `add.exe` entirely from Strict.
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

Counts verified on 2026-10-10 (files and lines per folder, root base types excluded). "C# replaced"
stays 0% until D7 switches a Runner stage to the Strict implementation and deletes the C# stage.

| Phase | Project | `.strict` files / lines | Current scope (differential test) | C# replaced |
|-------|---------|-------------------------|-----------------------------------|-------------|
| 1 | Language | 22 / 472 | Type/Member/Method model, header tokens, package lookup | 0% |
| 2 | Expressions | 49 / 1584 | Tokenizer, syntax tree, statements: every line of 18 folders round-trips (D1) | 0% |
| 3 | Validators | 5 / 235 | Same rule as C# for each validator case (D2) | 0% |
| 4 | TestRunner | 7 / 196 | Simple assertion evaluator | 0% |
| 5 | HighLevelRuntime | 20 / 560 | Line-level evaluator subset | 0% |
| 6 | Bytecode | 52 / 2772 | Compiles Examples to .strictbinary running like C# (D3), 402 vs 404 instructions (D4) | 0% |
| 7 | Optimizers | 15 / 261 | Instruction passes; codegen folding and load reuse live in Bytecode | 0% |
| 8 | Runtime | 12 / 574 | Strict VM runs 20 Strict-compiled Examples like the C# VM (D5) | 0% |
| 9 | Compiler | 25 / 1038 | Native compiler via MLIR, 14 Examples incl. texts like the VM (D6) | 0% |
| **Total** | | **207 / 7692** | **Every stage exists in Strict, Runner still uses C#** | **0%** |

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
