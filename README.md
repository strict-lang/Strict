# Strict

<video src="https://strict-lang.org/img/SCrunchDemo.mp4" controls width="100%"></video>

[![Strict Programming Language](https://strict-lang.org/img/StrictBanner.png)](https://strict-lang.org/img/SCrunchDemo.mp4)

[Watch the SCrunch demo](https://strict-lang.org/img/SCrunchDemo.mp4): how `.strict` code is written, tested while typing and run.

## A fast, low level engine for AI to write

Strict is a small programming language that AI can write, read and change as reliably as people can. Every type is one file, every method starts with its own tests, values are immutable and there is exactly one way to solve each problem, so generated code is checked by `strict check` and its tests within seconds instead of being reviewed line by line.

Large language models write code by imitation and their output still has to be verified by someone. Strict is meant as the DNA of a program: a compact, exact and fast low level layer whose rules a computer can fully understand, verify and evolve. The long term goal is a higher level language on top of it in which computers design and write whole programs, generating, testing and discarding millions of lines in real time.

What Strict does today:
- **Self-hosted**: parser, type inference, validators, the Strict VM and the native code generator (MLIR and LLVM) are written in Strict, `strict` builds itself.
- **Instant feedback**: `strict test` runs the inline tests on the Strict VM in memory, the VS Code extension shows passing lines and the values of each line while you type.
- **Native speed**: SIMD, loops spread over all CPU cores, NVIDIA GPU kernels (tensor cores, no CUDA toolkit needed), AMD GPUs through HIP, Metal on macOS (experimental), WebAssembly for the browser.
- **Real workloads**: GPT-2 and Qwen3 inference (Qwen3-8B, Qwen3-Coder-30B-A3B, Qwen3.8 27B) with the transformer logic written in Strict, image processing, neural network training, the DeltaEngine and SoulCraft games, C and C# libraries with generated bindings.

Latest performance results, docs (Getting Started, Language Guide, Cheat Sheet) and the blog: https://strict-lang.org

![Strict Layers](https://strict-lang.org/img/StrictLayers.svg?sanitize=true)

# Strict library

This repository is the compiler, the runtime and the standard library in one. Any folder containing `.strict` files is a package. The root folder holds the types with special language treatment, imported by default everywhere:
- Boolean, Number, Text, Character, Byte
- List (`Sorted`, `Merged`, `Slice`, `Insert`, `Unique`, `Average`, `Median`, `Shuffled`, ...), Dictionary, Range
- Error (with stack traces), Logger
- Time, Duration, PseudoRandom (deterministic, seeded), MovingWindow, PriorityQueue
- traits like Any (every type is Any), App, Generic, Mutable, File, Directory

More specific types live in packages like `Math`, `ImageProcessing`, `NeuralNetworks`, `MachineLearning`, `Compression`, `Json`, `Xml` and `Examples`, each with a short README. Strict is not a general purpose language: websites, databases or complex apps belong into external code, which Strict calls through native packages (C) or which calls Strict through generated libraries (C, C#).

An optional value uses `or None`, it can be checked but never assigned:
```
has Parent Context or None
if Parent is None
	logger.Log("This is the root")
```

# How to use
The base types are the documentation: every method starts with tests showing how to use it. Tests are `is` comparisons (`==` elsewhere, `=` assigns), every expression must be true or execution stops with an Error.

## Command line
`strict` (`strict.exe` on Windows) is in this folder next to the base types, put it on the PATH. No .NET or other runtime is needed (the C# implementation that bootstrapped Strict is on the `dotnet` branch).
```
strict Examples/Sum.strict 5 10 20        run on the Strict VM, arguments go to Run
strict test Examples/Sum.strict           run the inline tests on the Strict VM (a folder: every file and package sub folder)
strict build Examples/Sum.strict          run the tests, then build the native executable
strict build Examples/Sum.strict linux    also the Linux executable (on Windows through WSL)
strict build Examples/Sum.strict web      .html, .js and .wasm for the browser (Emscripten, installed when missing)
strict build Examples/Sum.strict android  arm64-v8a shared library and assets in Release/Android (Android NDK)
strict build Examples/Sum.strict ios      arm64 iOS static archive and assets in Release/iOS (native packages need Xcode)
strict build Examples/Sum.strict small    optimized for size instead of speed
strict build Policy.strict library        Policy.dll (.so, .dylib) exporting the public methods for C and C#, plus Policy.cs
strict build Games/App/App.strict         a windowed program goes into Games/App/Release with its Content (macOS: a signed .app)
strict check Examples                     validate a folder (type rules, unused members, limits)
strict format Examples/Sum.strict         print the file in its canonical form
strict test Examples/Sum.strict lines     run the tests natively, then show how often and how long each line ran
strict test Examples/Sum.strict values    run the tests on the Strict VM, then show the values each line produced
strict update .                           copy or download the packages of a folder again
strict version                            version and build time of this strict
```
- A type without `Run` builds its tests: `strict build Character.strict` runs `Character.test.exe` as the test gate (exit code: the number of failed tests), the Strict VM only when it does not build or crashes. A program with `Run` tests on the Strict VM unless that took more than twice its native build, then `Name.test.exe` gates later builds (delete it to go back). `web` and `library` builds always test on the Strict VM. Each test prints `Test passed: <test> in <Method> at <File>.strict:line <N>` or `Test failed: ...` with the value it got.
- `strict build` and `strict test ... lines` run `Compiler/NativeCompiler.exe`; when a file of `Compiler/`, `Bytecode/` or `Expressions/` or a `Compiler/NativeRuntime*.mlir` is newer than it (an edit or a pull), they build the compiler again first and note that in `Compiler/NativeCompiler.checked` (the runtime files are read by every build, the compiler logic is fixed in the executable, both must match).
- Exit codes: 0 success, `test` and `build` return the number of failed tests, 2 wrong usage or a missing file, an unhandled Error from `Run` exits with 1.
- `strict check` prints each issue as `<File>.strict:line <N>: <Rule>: <detail>`.
- Files whose tests passed on the Strict VM get a `Name.strictbinary` (bytecode of all used types), while it is up to date `strict test` reports them as passed right away and running and building reuse it. Passing native tests write none, `strict test` still runs them on the VM.
- Types are looked up in the current folder (and its sub folders in a project), next to `strict` (base types and packages like `Math`), in the projects next to the current one, then in GitHub repositories linked in the project's README and github.com/strict-lang (copied into the project with a `.origin` file).
- Errors before anything runs: `TypeNotFound`, `ParsingFailed`, `NameNotFound`, `MethodNotFound`, each with file and line. At runtime an Error prints its name and the stack trace back to the `.strict` source (at most the first 40 and last 10 lines), a call nested too deep returns `StackOverflow`.

Needed tools: `mlir-opt`, `mlir-translate`, `clang`, `opt` and `llc` of LLVM 22 (on Windows the msys2 ucrt64 `mlir` package and the LLVM release). When they are missing `strict` says so and installs them (Windows: winget and pacman, Linux: the LLVM release into `Tools/`, macOS: conda-forge with micromamba into `Tools/`).

`strict` runs, tests and checks on the Strict VM and hands `build` and `test ... lines` to `Compiler/NativeCompiler` (`.exe` on Windows), the native compiler next to it. Bootstrap a fresh checkout from the checked-in seed with `Compiler\Bootstrap.cmd` (Windows) or `sh Compiler/bootstrap.sh` (Linux and macOS): it builds `Compiler/NativeCompiler` from the seed, which then builds `strict`. After changing the compiler rebuild both, `strict build Compiler/NativeCompiler.strict` then `strict build Strict.strict`; building twice gives identical output.

## Rules worth knowing
- Limits: 120 characters per line, 13 lines per method (header and tests included), 4 parameters, 16 methods, 15 members, nesting depth 5, names 2-50 characters. Files in a `Generated` folder get 10 times as much.
- Precedence from loosest to tightest: `then`/`else`, `not`, `or`, `xor`, `and`, comparisons (`is`, `is not`, `is in`, `is not in`, `<`, `>`, `<=`, `>=`), `in`, `+`/`-`, `*`/`/`/`%` (left to right: `a % b / c` is `(a % b) / c`), `^`, `to`.
- A conditional as one of several arguments needs brackets: `Text(value, (isShort then 2 else 4))`.
- Constraints follow `with`: `has Width Number with value > 0`, `Run(frames = 0 with value >= 0)`. A value that does not fit is `Error("ConstraintNotMet: ...")`.
- A decimal is a Number with a step: `has price Number with Step 0.01` rounds every value going into it (half to even), plain Number arithmetic stays binary.
- Sums in a `for` may be added in any order (vectorized, parallel), compare sums of fractions with `IsNearlyEqual`.
- `(x)` is a list with one element, brackets around operators only group: `(5 + 3) * 2 is 16`.
- `and`, `or` and `xor` of Numbers are bitwise: `12 and 10 is 8`.
- Keywords (`from`, `to`, `is`, `in`, `and`, `or`, `not`, `constant`, `mutable`, `let`, ...) can't be names, `value` and `index` can't be declared in a method with a loop.
- The innermost name wins: a method, member or variable named like a type hides it, `outer.Marker` is the type.
- Members don't repeat their type when the name says it: `has number`, not `has number Number`.
- A member default is a literal, a constant, an empty `List(...)`/`Dictionary(...)` or a one-member type with literal arguments.
- Test lines are Boolean comparisons at the start of each method, every method needs one. A failing test prints what it got as Strict code to paste as the expected value.
- Other packages are named `Package/Type` only in declarations, method bodies use the plain type name.
- `constant` for values computed from constants, `let` for everything else that doesn't change, `mutable` only when a value really changes.
- An Error is handled where it shows up (returned, kept in a variable or checked with `is Error`), any other use returns it to the caller with one more stack trace line. See `Examples/ErrorHandling.strict`.
- A call of a method returning `Mutable(...)` as a statement (`numbers.Add(5)`) changes the `mutable` variable it is called on.
- Values never change: `list + element` gives a new list. The runtime reuses the old memory in place since the old version is not used anymore (using it again forces a copy, that is bad code).

## How it is built (from low to high level)
| Package | Responsibility |
|---------|----------------|
| root types | `Number`, `Text`, `List`, `Dictionary`, `Error`, `File`, ... define the language and are its documentation, `Strict` is the `strict` command |
| `Expressions` | Tokenizer, syntax tree, statements, type shapes, known types and type inference over all packages |
| `Validators` | Unused and never changed variables, constant folding, limits (`strict check`) |
| `Bytecode` | Compiles a type into register based instructions, inlines constants, removes tests, links used types, the `.strictbinary` cache |
| `Runtime` | The Strict VM running those instructions (`strict <file>` and `strict test`) |
| `Compiler` | Instructions to MLIR, the runtime in `NativeRuntime*.mlir`, GPU kernels, mlir-opt, mlir-translate and clang |

Details in the README of each package. Performance results of a package go into its README, the latest overall numbers are on https://strict-lang.org.

## Open work
See [improvements.md](improvements.md), in priority order.

### Key design decisions
1. **Immutable expression tree**: static analysis only, no runtime state sneaks in.
2. **Validation before execution**: catch the trivial stuff early, let the VM handle the hard parts.
3. **One-time tests**: tests live where the code lives, run once, then disappear from the bytecode.
4. **Strict rules first, then hardware speed**: exactly one way to solve each problem, then performance and parallelism as close to the hardware as possible (CPU, GPU, NPU, distributed). Immutable values reuse memory in place instead of copying.

Only code that is actually called is parsed, validated, tested and executed, so millions of generated lines can be checked and discarded almost immediately.

## Examples (from Number.strict, Text.strict and List.strict)
```
	3 + 4 is 7
	2 is not 3
	5 to Text is "5"
	2.4.Floor is 2
	"Hey" + " " + "you" is "Hey you"
	"Your age is " + 18 is "Your age is 18"
	"5" to Number is 5
	"lo" is in "hello there"
	"hello".StartsWith("hel")
	(1, 2) to Text is "(1, 2)"
	(1, 2, 3) + (4, 5) is (1, 2, 3, 4, 5)
	("1", "2") to Numbers + (3, 4) is (1, 2, 3, 4)
	(1, 2, 3) - 3 is (1, 2)
	(2, 4, 6) / 2 is (1, 2, 3)
	(1, 2) * (3, 5) is (3, 10)
	(1, 2, 3).Sum is 6
	3 is in (1, 2, 3)
	(1, 2, 3).Index(9) is -1
	("Hi", "Hello", "Hi").Count("Hi") is 2
```

## Members, variables, constants
Members are defined at the top of a type file. A type is composed of its members and can be used as any of them, if a member type is a trait (methods without implementation) the type implements its methods. Members are `has` (the default), `constant` (precomputed, anything built only from constants is a constant too) or `mutable` (rare, only for optimizations). A member or parameter can have a default like `has activation = Activation.Sigmoid`. A copy with some members replaced names them: `Unit(unit, health = unit.health - damage)`.

A member of type `Generic` makes a type generic, one type argument per `Generic` member: `Couple("a", 2)` with `has first Generic` and `has second Generic` is a `Couple(Text, Number)`. Methods are compiled per instance, so a call on a `Generic` member reaches the concrete type (`Examples/Couple.strict`, `Examples/Holder.strict`).

Any is the basis of every type and gives each one `from`, `to Type`, `to Text`, `to HashCode`, `is(other)` and `Members`. Some base types are tiny:
```
has number
from(text)
	Character("b") is "b"
	Character(7) is Tab
	Character("ab") is Error
	text.Characters(0)
```
A Text is a list of characters (`has characters`), a Name is a Text with constraints (`has text with Length > 1 and " " is not in value`). A type with one member is that member: `Byte(5)` prints as `5`, calling it with a value of that member wraps the value unless the type has its own `from` taking it: `Float8(0.3)` is `0.3125` (E4M3: the nearest value, ties to even, at most 448, the same on the Strict VM, the CPU and the GPU).

Variables live in a method body: `constant` for constant values, `let` for anything else that does not change, `mutable` only when it has to change:
```
TrimStart Text
	mutable startIndex = 0
	for characters
		if value is " "
			startIndex = startIndex + 1
		else
			return Substring(startIndex, characters.Length - startIndex)
	""
```

A computed name is one expression evaluated again on every use, only for very simple things (`Examples/ComputedRectangle.strict`):
```
	mutable position = 3
	next(add = 1) = position + add
	next is 4
	next(2) is 5
```

## if expressions
`if` comes in block, single-line and selector form, all of them are expressions:
```
if value > 0
	value
```
```
value > 0 then value else 0
```
The selector works like switch or match elsewhere, without an `else` a value no case matches is an Error at runtime:
```
if operation is
	"add" then value
	"subtract" then 0 - value
	"multiply" then value * value
```

# for
`for` is the only loop. It iterates anything that implements Iterator (List, Number, Range, Enum, Dictionary, ...). `value` is the current element, `index` its position and `outer` the value outside the loop. Loops whose body only calculates run in parallel on the CPU cores or on the GPU automatically.
```
	for (1, 2, 3)
		logger.Log(index + ": " + value)
```
`for 10` counts from 0 to 9, `for Range(2, 5)` gives 2, 3 and 4, iterating an enum (a type of only constants) gives its constants. `for as long as condition` repeats while the condition holds:
```
GCD Number
	GcdCalculator(48, 18).GCD is 6
	mutable current = first
	mutable modulo = second
	for as long as modulo is not 0
		let remainder = current % modulo
		current = modulo
		modulo = remainder
	current
```

## Dictionaries
A Dictionary is a list of key and value pairs, `value.Key` and `value.Value` read them in a `for`. `prices("Apple")` reads a value (a missing key is the `KeyNotFound` Error), `prices("Apple") = 4` on a `mutable` dictionary sets it:
```
mutable counts = Dictionary(Text, Number)
for words
	counts(value) = (value is in counts then counts(value) else 0) + 1
```
There are no other collections: List and Dictionary pick their internal representation at runtime from how they are used.

## for as a value
The last line of a method is its result and a `for` is an expression: the values of its body are collected into a list, which becomes a Text, a Number or a Boolean when the method returns one.
```
in(key Generic) Boolean
	2 is in Dictionary((1, 1), (2, 2))
	for keysAndValues
		value.Key is key
```
```
GetElementsText Text
	(1, 3).GetElementsText is "1, 3"
	for elements
		(index is 0 then "" else ", ") + value
```
```
Count(searchFor Generic) Number
	("Hi", "Hello", "Hi").Count("Hi") is 2
	for elements
		if value is searchFor
			1
```
Filter, map and reduce like a pipe in functional languages (the even numbers up to 9 times 10, summed):
```
for Range(0, 10)
	if value % 2 is 0
		value * 10
```
The same in C#: `Enumerable.Range(0, 10).Where(n => n % 2 == 0).Select(n => n * 10).Sum()`.

From `Examples/RemoveParentheses.strict`:
```
Remove Text
	RemoveParentheses("example(unwanted thing)example").Remove is "exampleexample"
	mutable parentheses = 0
	for text
		if value is
			"(" then parentheses.Increase
			")" then parentheses.Decrease
			else if parentheses is 0
				value
```

## History
The idea for my (Benjamin Nitschke, CEO of Delta Engine) own programming language started in the 1990s after reading "Gödel, Escher, Bach" by Douglas R. Hofstadter. A first iteration in 2008-2010 worked but was dropped for Lua. This repository started 2020-06-16 for editor plugins and new parsing ideas based on the [old Strict sdk](https://github.com/strict-lang/sdk), with some help from employees in 2020-2022 until the parent project ran out of funding. Since mid 2025 I am working on it again in the evenings and on weekends, towards the original goal: a base layer language computers can actually understand and build on, the way DNA and proteins are the building blocks of life.
