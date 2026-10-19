# Strict

![Strict Programming Language](https://strict-lang.org/img/StrictBanner.png)
<div style="text-align: center;">Create well-written and efficient code quickly</div>

## Strict is a language computers can understand

Strict is a simple-to-understand programming language that not only humans can read and understand, but also computers are able to understand, modify and write it.

The long-term goal is NOT to create just another programming language, but use Strict as the foundation for a higher level language for computers to understand and write code. This is the first step towards a future where computers can write their own code, which is the ultimate goal of Strict. Again, this is very different from other programming languages that are designed for humans to write code but not for computers to understand it. Even though LLMs can be trained to repeat and imitate, which is amazing to see, they still don't understand anything really and make the most ridiculous mistakes on any project bigger than a handful of files. Strict needs to be very fast, both in execution and writing code (much faster than any existing system), so millions of lines of code can be thought of by computers and be evaluated in real time (seconds).

More at https://strict-lang.org

## Layers

![Strict Layers](https://strict-lang.org/img/StrictLayers.svg?sanitize=true)

# Strict library

This repository is not just the runtime, but contains the standard library. It is one and the same. Any folder or GitHub repository folder that contains .strict files is a package. The main folder contains all types that have special language treatment and are required to build strict packages. It is imported by default in every source file, here are the most used types:
- Boolean
- Number
- Text
- List (also `Sorted`, `Merged`, `Slice`, `Insert`, `SwapRemoved` (the last element takes the removed place), `Unique`, `Average`, `Median`, `Normalized`, `IndexOfMaximum`, `IndexOfMinimum`, `Pick` and `Shuffled` with a `PseudoRandom`)
- Dictionary
- Range
- Error (with Stacktraces, Type and Methods support)
- Log
- Time (seconds since 1970 UTC: `Time.Now`, `Time.Day`, calendar `Date` and `Clock`, ISO texts and `Parse`, `RoundUp`, `IsWithin`, `Elapsed` like "about 3 minutes ago", `Time - Time` is a `Duration`, `Time + Duration` a `Time`)
- PseudoRandom (the multiply-with-carry generator of the C# DeltaEngine: `Seeded` takes a nonzero integer, keep the state returned by `Next`, then read `Uint` or `Get(minimum, maximum)` or `WeightedIndex(weights)`, none of them advance it; zero seeds are rejected instead of reading the clock, so simulations stay deterministic)
- Duration (seconds with `Milliseconds`, `Minutes`, `Hours`, `Days`, `+`, `-`, `*`)
- MovingWindow (the last values of a fixed capacity with their running `Sum` and `Average`, `Minimum`, `Maximum`) and PriorityQueue (a binary min-heap of numbers, `Pushed`, `Popped`)
- common traits like Any (applies to all types), App, Generic (replaced by concrete type in implementation), Mutable, File, Directory, FileReader, FileWriter, etc.

That is pretty much it, there are not many more things needed to write code. For more specific things see the Math folder, ImageProcessing, Examples, etc. here in this repository. Keep in mind that Strict is NOT a general purpose language. It is supposed to be writeable, readable and understandable by computers. If you want to build a website, a complex app, database code, a game, etc. you should use external code for that (you can keep the logic in Strict, but it is unlikely that Strict will have those things unless some people work on it and need it). It is quite easy to interface with external code, you can use any web api, C++, C#, Java, Python, etc. libraries. Those are threated as black boxes (thus not longer computer understandable, here we would be limited to LLMs) and will not follow the strict rules in Strict. The main issue here would be if an external library changes, then all your calls to it also have to change, which is less of a problem with Strict code calling Strict code as any changes can be fixed in a more automated manner.

In the rare case you need an optional value you can specify it with the or None type constraint (any other type also works). Keep in mind you still cannot assign None to anything, but we can check if something is None and simply not assigning it will keep it at the None state.

Context.strict
```
has Parent Context or None
has Name
[...]
if Parent is None
  logger.Log("This is the root")
```

Parent can just be used as Context and as any derived type (anything that uses Context), but if it was not set and thus is still None, an error will be thrown. You can of course always check for None, but this is a very rare pattern in Strict. An error is also thrown when using a type as a higher level type, which it is not. Let's say our Context is a Package, but we try to use it as a Type (which also is a Context, but clearly not a Package). An Error is thrown telling us we can't use it as a type (like a cast in any c like language would do).

# How to use
The best way to learn how any expression or base type works is to look at the code, it is very compact and every single method starts with tests on how to use it. This base library is not just defining how the language works, but the documentation in source code form. All tests use the is comparsion (similar to == in c style languages, = is used for assignments), "is true" at the end of any expression is never used. All expressions (not just tests) must evaluate to true, otherwise the execution stops with an error. If an expression is constant (like all tests, but also many other expressions), it will be removed and optimized out after the first run.

`.strict` files must not end with trailing blank lines or whitespace-only tab lines.

## Command line
Strict is written in Strict and compiles itself to a native executable, no .NET or other runtime is needed (the C# implementation that bootstrapped it lives on the `dotnet` branch).
```
strict Examples/Sum.strict 5 10 20        run on the Strict VM, the fastest, everything stays in memory
strict build Examples/Sum.strict          run the tests, then the native executable (on Windows also the Linux one through WSL)
strict test Examples/Sum.strict           run the inline tests of every method on the Strict VM (a folder: every file in it and its package sub folders)
strict build Games/App/App.strict         a Run program that opens a window goes into Games/App/Release (kept between builds, only the executable is replaced), Games/App/ContentBuild.strict builds Release/Content when it exists (strict ContentBuild.strict Content Release/Content), otherwise Content is copied incrementally (new and changed files, removed ones deleted)
strict build Examples/Sum.strict windows  the same without the Linux executable (when the native code only exists for Windows)
strict build Examples/Sum.strict small    optimized for size (-Os) instead of speed (-O2, the default), also its C natives, combines with windows
strict build Policy.strict library        Policy.dll (Linux libPolicy.so) exporting its public methods for C# and C, plus Policy.cs
strict check Examples                     validate a folder and its package sub folders (type rules, unused members, limits)
strict format Examples/Sum.strict         print the file in its canonical form, each method body as the parser reads it
strict test Examples/Sum.strict - < A.txt test A.txt (standard input) as Examples/Sum.strict, the file itself is not touched
strict test Examples/Sum.strict lines     run the tests natively, then print how often and how long each line ran (ms with callees, self ms without)
strict test Examples lines                the same for every file of a folder and its package sub folders in one native build (Examples/Tests.test.exe)
strict test Examples/Sum.strict values    run the tests on the Strict VM, then print Value <file>:line <n> hit <count> times per executed line of a method (test lines print nothing, a passed test is true): with the same value on every hit followed by : <name> = <substituted> = <value> for a let, mutable or assignment line (: sum = 5 + 3 = 8, : <name> = <value> for a literal) or : <substituted> = <value> for an expression (5 + 3 = 8, nothing for true), otherwise , each: <value>, ... with the value of up to 10 hits, a for line lists its iterated values (, each: 0, 1, 2, 3), values in code form (texts cut after 80 characters, lists after 5 elements)
strict update .                           copy or download the packages of a folder again from their source
```
A library exports every public method with Number, Text, Boolean, Numbers or record values as `int Policy_Start(const char* mode, const char* enabled, const char** result)`: 0 on success, 1 for an Error whose text `strict_last_error()` gives; a text result stays valid until `strict_release(result)`. A list of numbers crosses as a pointer to `{count, items...}` doubles (a result is released with `strict_release`, C# sees `double[]`). A type whose members are all Number or Boolean crosses as a C struct of doubles in member order (its methods get the instance as first argument, `Trade_Total(const Trade* instance, double* result)`). A list of those records crosses like a list of numbers, `{count, fields of each record...}` (C# `Trade[]`). The generated `Policy.cs` wraps all of it (`Policy.Start("preview", "false")`, `TradeMethods.Total(trade)`, an Error becomes an InvalidOperationException).

`strict` is `strict.exe` (on Linux `strict`) in this folder, next to the base types, put this folder on the PATH to use it everywhere. `strict` alone explains each command, `strict version` shows the version number and when this `strict` was built. Without a command the file is run, like `python file.py`. A build keeps the executable and its `.mlir` (the input of the tools, also the bootstrap seed), the other tool files are removed. On Windows `App.ico` next to `App.strict` becomes the icon of `App.exe`, an icon group named `GLFW_ICON` (GLFW uses it for its windows) compiled by `llvm-rc` in every build, so a changed icon is always picked up (a program without its own icon uses the one named like its folder, `Games/Blocks/Blocks.ico` for `Games/Blocks/Tetris.strict`, and otherwise `Strict.ico` next to `strict.exe`, which is also the icon of `strict.exe` itself) (`Examples/NativeAdd`). A type without a `Run` method builds its tests instead: `strict build Character.strict` gives `Character.test.exe`, its exit code is the number of failed tests (this is how the native code generator is tested). Every test prints one line, `Test passed: <test> in <Method> at <File>.strict:line <N> (0.02 ms)` with the time the test took or `Test failed: ...` followed by the value it got, on the Strict VM and natively. `strict check` prints every issue as `<File>.strict:line <N>: <Rule>: <name or detail>`. Every file whose tests passed gets a `Name.strictbinary` next to it, the bytecode of all used types (a type whose own `.strictbinary` in the same folder has the same methods is only referenced, `strict test <folder>` repacks the files it wrote at the end): while it is up to date (no source file of a type it contains, no data file it names and no file the tests read, `Process.FilesRead`, has a different modification time, so editing another file of the folder does not count, neither does a new `strict.exe`) `strict test` reports the tests as passed right away, and running and building use that bytecode instead of compiling again. Types are looked up in the current folder first (in a project folder also in all its sub folders, so `Shapes/Square.strict` is just `Square`; when two sub folders define the same type name, a declaration like `has scene WebWars/Scene` chooses one, otherwise the one in the file's own folder is used and anything else is an `AmbiguousType` error), then next to `strict.exe` (the base types and packages like `Math`, `Vector2` finds `Math/Vector2.strict`, and the types those packages use), then in the package folders of the projects next to the current one (`../DeltaEngine/Input` for a type `SoulCraft` uses, the local clones are the newest while several projects change together: their files and `native/` folders are used in place, paths in their `.flags` are relative to that project), then in the GitHub repositories linked in the project's `README.md` and in github.com/strict-lang: the package holding the type is copied into the current folder (only for files inside it, a file elsewhere reports the type as missing instead), one sub folder per package with its own sub folders, from a local clone in `C:/code/GitHub/owner/repository` or downloaded with `curl` (GitHub is never asked when `C:/code/GitHub/strict-lang` exists, its API is called with the token of `gh auth token` when the GitHub CLI is logged in). A `.origin` file in the package folder names its source: `strict update .` copies or downloads every such package again, a package that lacks a newly used type is refreshed on its own. A type found nowhere (declared or only used in a method body) is a `TypeNotFound` error with file, line and tips. A line that does not parse is a `ParsingFailed` error and an unknown name or method (in the file or in any type it uses) a `NameNotFound` or `MethodNotFound` error, also with file and line, before anything runs. At runtime an Error prints as `Error: Name` followed by `at Type.Method in file:line N` for the method that created it and every method it was returned from, on the Strict VM and natively, a call nested too deep returns a `StackOverflow` Error (the Strict VM after 20000 calls, natively a method that calls others when less than 1 MB of the main stack is left), a method calling itself as the result of a line (`number is 0 then total else Counted(number - 1, total + 1)`, `Examples/Countdown`) jumps back to its start instead on both, so it never runs out of stack, `strict check` reports a method calling itself inside an expression with an argument stepping down from a literal over 1000 as `DeepRecursion`, an unhandled Error from `Run` exits with code 1 (a windowed Windows executable without a console writes it and stack overflows to `App.exe.crash.txt` next to it as well). A folder with `Any.strict` after the file overrides where the base types are, any other argument goes to `Run`. Exit codes: 0 success, `test` and `build` return the number of failed tests, 2 wrong usage or a missing file.

Needed tools: `mlir-opt` and `mlir-translate` (LLVM MLIR, on Windows the msys64 ucrt64 package `mingw-w64-ucrt-x86_64-mlir`) and `clang` (LLVM 22, its `llvm-rc` for application icons on Windows). When they are missing `strict` says so and starts installing them (Windows: winget and pacman in a new window, Linux: the LLVM 22 release archive into `Tools/`).

Bootstrap a fresh checkout from the checked-in seed (the compiler itself as MLIR) with `Compiler\Bootstrap.cmd` (Windows) or `sh Compiler/bootstrap.sh` (Linux), then let it build itself: `strict build Strict.strict` twice gives identical output (fixpoint). On Windows an existing executable is linked to `name.new.exe` first and then swapped in (`File.Rename`), the running `strict.exe` stays as `strict.old.exe` until the next build.

## Rules worth knowing
- Limits: 120 characters per line, 13 lines per method (header and tests included), 4 parameters, 15 methods, 15 members, nesting depth 5, names 2-50 characters. Files in a folder named `Generated` (written by tools, like texture atlas data) may have 10 times the characters per line, lines per method, methods and members, anything bigger belongs into a data file.
- Precedence from loosest to tightest: `then`/`else`, `not`, `or`, `xor`, `and`, the comparisons (`is`, `is not`, `is in`, `is not in`, `<`, `>`, `<=`, `>=`), `in`, `+`/`-`, `%`, `*`/`/`, `^`, `to`. So `1 < 2 and 3 > 2 or 4 is 0` is `((1 < 2) and (3 > 2)) or (4 is 0)` and `value > 31 and value is not in (34, 92)` needs no brackets, but a test of `and` on numbers needs them: `(12 and 10) is 8`.
- A conditional used as one of several arguments needs brackets: `Text(value, (isShort then 2 else 4))`.
- A member or parameter can have a constraint after `with`: `has Width Number with value > 0`, `has text with Length > 1`, `Run(frames = 0 with value >= 0)`. `value` is the member or parameter, other names its type has (`Length`) are read from it, anything else from the type (`Max`). Constructing an instance (a type with constraints gets a generated `Validated` method that every construction calls) or calling a method with a value that does not fit gives `Error("ConstraintNotMet: Width with value > 0 is false for 0")`, on the Strict VM and natively. `where` is not a keyword (`ConstraintNeedsWith`).
- A decimal is a Number with a step: `has price Number with Step 0.01` (also on parameters and return types, `with Step 0.01 Floor` or `Ceiling` to round down or up). Every value going into it is rounded to the step, half to even (`0.125` becomes `0.12`, `0.135` becomes `0.14`), and a comparison with a stepped side compares both sides rounded to the finest step, so `Prices(0.1, 3).Total > 0.3` is false (see `Examples/Prices.strict`). Values are exact up to 2^51 steps, beyond that the rounding is `Error("DecimalOverflow")`. Plain Number arithmetic stays binary: `0.1 * 3` is `0.30000000000000004` unless it goes into a stepped member, parameter or return value.
- A `for` that adds up numbers (`sum = sum + value * other(index)`, `Sum`) may add them in any order, vectorized and later in parallel: the last bits of a sum of fractions can differ between the Strict VM and native code, compare such sums with `IsNearlyEqual`. Everything else is calculated in the written order on both.
- `(x)` is always a list with the one element `x`: `("abc").Length is 1`, `("abc") is not "abc"`. Brackets around an operator only group: `(5 + 3) * 2 is 16`, `(true and true) is true`. Returning one element where a list of it is declared (`("a" + name)` in a `Texts` method) is a `ReturnTypeMismatch` error, write `List(Text) + ("a" + name)`.
- `and`, `or` and `xor` of two Numbers are bitwise on their integer values (64-bit two's complement): `12 and 10 is 8`, `12 xor 10 is 6`.
- Keywords and operator words (`from`, `to`, `is`, `in`, `and`, `or`, `not`, `constant`, `mutable`, `let`, ...) can't be names. A parameter called `value` collides with the implicit instance `value`.
- The innermost name wins: a method, member, constant or variable named like a type (`Marker`) is that method, member, constant or variable in its scope, `outer.Marker` is the type (outside of a `for`, inside one `outer` is the outer value). Using such a method like the type says so: `MethodNotFound: Number.Unplaced, local Marker hides the type Marker, use outer.Marker`, also for a call with arguments the method does not take (`Size(number, 1)` in a type with a `Size Size` method, `constant Size = Size(0.5, 2)`). `Size.Width` where `Size` is both an own method and a type is `AmbiguousName`: write `outer.Size.Width` for the type or `value.Size.Width` for the method. `outer.Size(0.5, 2)` constructs the type, also in constants. A parameter named like a member is the parameter, `strict check` asks for another name (`ParameterHidesMemberUseDifferentName`).
- Members don't repeat their type when the name already says it: `has number`, not `has number Number`.
- A member default is a literal, a constant of any type (`has pulse SinusCurve = SinusCurve.Flat`, every constructor call leaving it out passes the constant), an empty `List(...)` or `Dictionary(...)` or a one-member type with literal arguments (`has start Vector2 = Vector2(0, 0)`, `has label Text = "Names"`), anything else (`= Size(1, 2)`, `= Created(data)`) is an `UnsupportedDefault` error and a constant the type does not have is `ConstantNotFound`.
- Test lines are Boolean comparisons (`is`, `is not`, `>`, `<`) at the start of each method; every method needs at least one. A failing test prints its value after `got:` as Strict code (`value.Code` of Any: `"ab"`, `("a", "b")`, `Bag(("a"), 1)`, `Error("Zero")` followed by its stack lines), so it can be pasted as the expected value.
- Other packages are referenced as `Package/Type` only in declarations (members, parameters, return types), method bodies use the plain type name. Two same-named types from different packages cannot be used in one file (`SameNamedTypes`), types are keyed by their plain name in a build.
- `constant` for values computed only from constants, `let` for everything else that doesn't change, `mutable` only when a value really changes (loop counters).
- An Error is handled where it shows up: it can be returned, kept in a `let` or `mutable` variable, or checked with `is`/`is not` against an Error (`x is Error` matches any Error, `x is KeyNotFound` or `x is Error("Name")` a named one, `x is in (KeyNotFound, DuplicateKey)` several). Any other use of a value that turned out to be an Error (arithmetic, `<`, a condition, calling a method on it, passing it as an argument, storing it in a member or list, dropping it as a statement) stops the method right there and returns that Error to the caller with one more stack trace line, up to the test line (which fails with the Error as `got`) or `Run` (prints the Error with its stack trace, exit code 1). Inside `if x is Error` the variable `x` is an Error (`x.Name`, `x to Text`). See `Examples/ErrorHandling.strict`.
- A call of a method returning `Mutable(...)` as a statement (`numbers.Add(5)`, `prices.Add("Plum", 5)`, `count.Increment`) changes the `mutable` variable it is called on, a `DuplicateKey` Error from it stops the method. On anything else the call would change nothing and is a `MutatingCallNeedsMutableVariable` error.
- Values never change: `list + element` (or `list + otherList`) gives a new list, a `mutable` variable can only be reassigned to such a new value. Internally the runtime keeps the same list and appends in place, the new list is the same memory and the previous version is not used anymore. Changing an old version again forces a copy, that is bad code: a list is either fixed, or growing and then only its latest version is used.

## How it is built (from low to high level)
| Package | Responsibility |
|---------|----------------|
| root types | `Number`, `Text`, `List`, `Dictionary`, `Error`, `File`, ... define the language and are its documentation, `Strict` is the `strict` command |
| `Expressions` | Tokenizer, syntax tree, statements, header tokens, type shapes, known types and type inference over all packages |
| `Validators` | Unused and never changed variables, constant folding, limits (`ValidateCheck` behind `strict check`) |
| `Bytecode` | Compiles a type into register based instructions (`FileCompiler`), inlines constants, removes tests, links used types, reads and writes the `.strictbinary` cache (`BinaryFile`) |
| `Runtime` | The Strict VM running those instructions (`Execute` behind `strict <file>` and `strict test`) |
| `Compiler` | Instructions to MLIR, `NativeRuntime*.mlir` (texts, lists, files, processes, collector), mlir-opt, mlir-translate and clang |

Native model: every value is one 64-bit slot, numbers are doubles (a list of numbers that C code hands out as 32 bit floats, 16 bit halves, 8, 6 or 4 bit quantized numbers stays packed like that, a `for` adding it up reads it directly (also at an offset, `row(first + index)`) and calculates in 32 bit floats, packed numbers are never more precise than that, any other use unpacks it into doubles once; `Float8` is a number that lists keep in one byte where they are packed: a GPU loop collecting `Float8` values like `Float8.Packed(numbers)` stores 8 bit floats, E4M3 with 3 mantissa bits and at most 448, which read back as numbers, the Strict VM and loops on the CPU keep doubles; a 32 bit GPU kernel keeps every number next to its float as an exact fraction of two 64 bit integers, `MlirExact`: the index, constants, `+ - * / %`, `Floor`, `Length` and the parameters (whose doubles it splits with integer operations, `MlirBits`, the RTX 4090 runs double operations at 1/64 of the rate) are exact, the rest only when its float is a whole number, and list reads take their position from it, so a 32 bit kernel stays exact past 2^24 positions; the parts no list read needs are dropped by LLVM), texts, lists and instances are pointer bits, Error values carry a NaN tag in the top bits so `x is Error` works for a value of any type (a number from C code carrying that tag becomes NaN). A type with one member is that member (`Name` is its text, `Degrees` its number), a type of 3 to 16 numbers of 32 bits that only reads them by `X`, `Y`, `Z`, `W` and `Length` (`Vector3`) is one block of its count and floats, other instances are heap blocks with one slot per member. Lists append in place while they are the newest version, an element write to a list that is used elsewhere too writes in place as well: the old header keeps the replaced value and points to the new one, so an older version that is used again copies its elements out on first use. A `for` that collects one value per element (like `for weights` in `Layer.Sums`) runs on several threads when it has at least 256 elements and its body only calculates: numbers, reading lists and members, conditional values, calling methods that do the same (no logging, collecting only some elements, files, processes, `Random`, writing to lists, building texts or lists, nested collecting loops). A pool of a quarter of the logical processors takes chunks of the elements, the collector does not run while they work. With an NVIDIA GPU a collecting `for` (also over an ascending `Range`) whose body only does arithmetic on numbers (including `%`, `^`, `Floor`, `Logarithm` and methods of types holding one number like `Activation.Tanh.Output`), reads list elements, picks values with conditionals (`x > 0 then x else 0`) and calls methods doing the same runs there when it reads at least 4M numbers (elements, or rows times their length for a list of rows like `weights`, also when it passes the rows to dot products), a loop collecting `Dot(rows(index % n), 0, input, (index / n).Floor * width)` (`Batch.Sums`, a matrix times the numbers of several tokens) gets all its dot products from one runtime kernel (`NativeRuntimeTokens.mlir`, only in programs that use it): from 2 tokens on a block of 8 warps takes 8 rows and up to 32 tokens on the int8 tensor cores (`mma.m8n8k16`, each warp an eighth of the row, its sums added up in shared memory, for more than 32 tokens 64 rows and each warp the whole rows of its 8 so they share the token numbers; a lane reads 8 bytes of every 32 numbers of its row and of its token in the same order, so Q4_K, Q6_K and 8 bit rows go in as they are, the token numbers as 8 bit numbers with a scale per 32; from 64 tokens on Q4_K and Q6_K rows take `strict_device_tiles`: a block stages the blocks of 256 numbers of 64 rows and of 64 tokens in shared memory with their scales, minimums and token sums and each warp computes 32 rows by 16 tokens of them, so every row is read once per 64 tokens and the scaling is shared by more products; each thread loads its part of the next 256 numbers into registers before the warps compute the current ones, two blocks of 128 registers per SM, Q4_K rows are unpacked to 8 bit numbers when they are staged and ldmatrix loads the mma operands; `Batch.Sums` passes the list of `+ biases(index % biases.Length)` to the matrix products, which add it when they store, so no second kernel reads the products again, `MlirBiased`), a single token goes row by row (dp4a), each part of a row is decoded once for all tokens, `Attention.Merged` of a prompt (`MlirAttention`, a type with queries, keys, values, heads, groups, size and sequences whose `Merged` uses `Softmax.Shares` and `Softmax.Weighted`) is one FlashAttention-2 style runtime kernel (`NativeRuntimeAttention.mlir`, `mma.m16n8k8` with 16 bit inputs and 32 bit sums, keys and values in shared memory 64 positions at a time, the softmax online so no score goes to memory, a warp per 16 rows of tokens and heads of one group of keys, positions after a token skipped; when up to 4 warps hold all tokens of a group, like decoding or verifying up to 16 drafted tokens of Qwen3-8B, each warp of tokens gets its own blocks, the 8 warps of a block and up to 16 blocks per warp of tokens split the positions 8 at a time, read keys and values straight into the mma operands and merge their outputs, the last block writes them), a list that is already on the GPU or packed numbers while lists live there: each loop is a kernel (PTX built by clang into `Name.ptx` and `Name.older.ptx` next to the executable, read at the first launch, without them the loops run on the CPU, loaded through the driver `nvcuda.dll` or `libcuda.so.1`, no CUDA toolkit needed to run), a loop adding up numbers inside it (`Layer.Summed`, also several sums and `let` values in it) runs on a warp of 32 threads per element, a counted loop whose one such call only depends on `index` through `(index / size).Floor` (a group, `HeadScale(input, start + (index / size).Floor * size, size)` in `Qwen3Block.Normalized`) runs one thread per element in blocks of up to 256 that compute the call of their group together (the sum adds up the warps in shared memory), on the CPU the call runs once per group, so does a loop changing a number only through a method that returns one of its two numbers by comparing them (`largest = Larger(largest, x)`, a maximum or minimum), a sum of `value * other(index)` over a row or of `x(a + index) * y(b + index)` is one dot product where every lane decodes 32 numbers of a 4 bit row or 16 of a 6 bit row per step, a list of rows can be indexed there too (`gateUps(expert * size + row)` passed to such a dot) and `Length` of a list works. A loop reading packed numbers (32 bits or fewer) runs as a 32 bit kernel and its result stays on the GPU as a list of 32 bit floats: the next loops read it there and the launches do not wait for the GPU, the CPU copies the numbers only when it reads the list itself (the logits of a language model), appending such lists (`keys + keyed`) happens on the GPU too, in place while the list is the newest version, a list from the CPU appended with one there moves to the GPU first (as 32 bit floats). Loops over doubles run as 64 bit kernels and wait for their result, `^` needs the CPU there. Launches are captured into CUDA graphs of up to 256 (8 after a download that waits for the GPU, doubling) that run again with updated arguments, about 2 instead of 6 us per launch (`STRICT_GRAPHS=0` launches directly), a download of a small result (up to 256 numbers) waits only for the graph that wrote it, not for the launches after it. A dot product of a packed row (4, 6 or 8 bits) with 32 bit numbers that the loop only reads in dot products uses 8 bit numbers instead (a scale per 32, made by a quantize kernel per launch) and `dp4a`, which multiplies 4 bytes at once (`STRICT_DP4A=0` keeps 32 bits). Host lists stay on the GPU after their first use when they have at least 64 KB, smaller ones from their second use (until a collection finds them unreachable, at most 65536 lists), packed rows keep their own widths (4 bit query rows next to 6 bit value rows). Results are the same as on the CPU up to the order of sums and 32 bit rounding (`^` is approximated with `ex2.approx` and `lg2.approx` there), `STRICT_GPU=0` keeps everything on the CPU, `STRICT_GPU=all` sends every such loop to the GPU (`Examples/Squares`). Memory is managed by a conservative mark-sweep collector with a young generation (64 KB pages, size classes, interior pointers; marks stay set so objects that survived a collection are old, a write into an existing object marks a 512 byte card, a young collection after 64 MB allocated or 256 MB taken on the GPU marks only from the roots and marked cards, a full one after 16 MB or 8x the live data or when results on the GPU took half of its free memory; GPU blocks of 64 MB and more are allocations of their own, before an allocation would leave less than 512 MB free the collector runs fully and the free ones go back to the driver, since the Windows driver pages memory out instead of failing and every kernel got 5 to 10 times slower, `STRICT_YOUNG` sets the young limit in MB, 0 only collects fully), 1M loop iterations on the Strict VM use 22 MB. Every heap object also counts the references that other heap objects hold to it (an i32 and a flag byte per object on its page, locals are not counted): objects of 256 bytes or more, and the small ones referencing them (a view of a long list, an instance holding one), go into a table; after 1 MB allocated a reclaim scans only the stack and frees each one with no count and no stack reference, its references give back their counts in cascade and its block is the next one allocated of its size, so a loop doing the same steps again reuses the same memory hot in the cache without tracing what survives (784-64-6 MNIST training 834 to 520 ms). A local that only ever holds fresh values (constructions, list literals, collected lists, constants, results of methods returning such values) and never leaks into another local frees its old value when it is overwritten (`MlirOwned`), so `position = position + velocity` takes the same block again; lists built by literals, constructors and collecting loops append in place. `STRICT_RC=0` leaves everything to the collector, `STRICT_RC_CHECK=1` recomputes the counts at every full collection and reports any count that is too low, and before each reclaim marks from the roots and keeps (and reports) anything the counts would free that is still reachable.

Native packages: a type without members whose methods have no body (a trait like `Examples/NativeAdd/Adder.strict`) is implemented in C in the `native/` folder next to it. `Adder.Add(1, 2)` calls `double Adder_Add(double self, double left, double right)` (`Type_Method`, `self` is 0 for calls on the type) from `native/Adder.c`. A type with one member works the same and holds what C hands out, like a window or texture handle: its bodyless `from(...)` is the C constructor (`Counter(10, 5)` calls `Counter_from(0, 10, 5)`, the result becomes the member) and its methods get the member as `self` (`counter.Next` calls `Counter_Next(self)`). Every argument and the result is one double: a Number, or a Text or Numbers that `strict.h` reads with `strict_text(value)` and `strict_numbers_of(value)` (borrowed for the call, Numbers as a flat copy `{count, items[]}`) and returns with `strict_text_result` and `strict_numbers_result` (the caller copies them right away), `strict_packed_result` returns numbers as 32 bit floats, 16 bit halves (`strict_half` rounds a float) or 8 bit quantized numbers (`strict_quantized`: a signed byte per number and a scale per 32 numbers, like Q8_0 of ggml), a `List(Numbers)` result is a Numbers whose items are such results (one per row), `strict_fail("Name")` returns an Error. `STRICT_EXPORT` marks the functions. `strict build` compiles each C file once into an object next to it (again when a C or header file of the folder, `strict.h` or the flags are newer) and links them into the executable together with the flags in `native/Windows.flags` or `native/Linux.flags` (libraries to link), the Strict VM builds them into `native/<Folder>.dll` (`.so` on Linux) when a source is newer and calls the functions through the root type `Library`. Packages copied into a project keep their `native/` folder.

Strict VM model: when the types are loaded `Resolver` numbers the locals and registers of every method into slots, resolves every call target and jump, decodes number constants and gives every instruction the slot its result goes to (`InstructionEntry.Target`). A call runs in one list of slots that its `Frame` owns and writes in place (the compiler keeps a list owned when it is only passed to methods of the same type that neither store nor return it), member values come from one decoded instance per type, constant instances like `VmValue.None` are built once natively. A call nested more than 20000 calls deep returns a `StackOverflow` Error with the Strict stack, the other tests of the file still run. Training `NeuralNetworks/Iris.strict` on the VM takes 2.1 s (4.0 s with a new frame per instruction), `ImageProcessing/ProcessImage.strict` 27 s (82 s).

Speed (1M loop iterations): native executable 40 ms (269 KB), Strict VM 105 ms, C# Strict.exe 420 ms. `strict` builds itself in 14.5 s, most of it in mlir-opt and clang. The neural network samples (`NeuralNetworks/`) log their training time, the MNIST digits of [Mnist](https://github.com/strict-lang/Mnist) train a 256-32-4 network on 128 images for 100 epochs in 0.49 s natively.

## Open work
See [improvements.md](improvements.md), in priority order.

---

### Key Design Decisions

1. **Immutable Expression Tree**
	 Static analysis only; no runtime state sneaks in.

2. **Validation Before Execution**
	 Catch the trivial stuff early, let the VM handle the hard parts.

3. **VM Is King**
	 Only the VirtualMachine sees real values and side effects — everything before it is compilation.

4. **One-Time Tests**
	 Tests live where the code lives, run once, then disappear from the bytecode.

5. **Strict Rules First, Then Hardware Speed**
	 First priority: clean strict rules with exactly one way to solve each problem. Second: performance and parallelism as close to the hardware as possible (CPU, GPU, NPU, TPU, CUDA, distributed over the network). Immutable values are implemented by reusing memory in place, not by copying.

---

### Why This Matters

Only needed code that is actually called is ever parsed, validated, tested and executed in the VM. This is a very different approach and allows generating millions of .strict lines of code and discarding anything that is not needed or doesn't work pretty much immediately.
* **Fast Feedback** — formatting and type errors fail instantly, long before the VM.
* **Deterministic Builds** — immutable trees + one canonical code style = no drift.
* **Lean Bytecode** — by the time code hits the VirtualMachine, dead weight is already gone.

---

Happy coding with **Strict** — where there’s exactly one way to do it right.

## Examples (all from Strict/Number.strict)
```
	3 + 4 is 7
	2 is not 3
	5 to Text is "5"
	123.digits is (1, 2, 3)
	5 to Text is "5"
	2.4.Floor is 2
	"Hey" + " " + "you" is "Hey you"
	"Your age is " + 18 is "Your age is 18"
	"5" to Number is 5
	"A" is "A"
	"hi" is not in "hello there"
	"lo" is in "hello there"
	"hello".StartsWith("hel")
	"hello".StartsWith("hel", 1) is false
	"yo mama".StartsWith("mama") is false
	"abc".Count("a") is 1
	"Hi".ToBrackets is "(Hi)"
	"1, 2, 3".ToBrackets is "(1, 2, 3)"
```

List.strict usage (all lists are simply surrounded by brackets, basically any method call uses a list as the arguments)
```
	(1, 2) to Text is "(1, 2)"
	(1, 2).Length is 2
	(1, 2) is (1, 2)
	(1, 2, 3) is not (1, 2)
	(1, 2, 3) + (4, 5) is (1, 2, 3, 4, 5)
	("Hello", "World") + (1, 2) is ("Hello", "World", "1", "2")
	("1", "2") + (3, 4) is ("1", "2", "3", "4")
	("1", "2") to Numbers + (3, 4) is (1, 2, 3, 4)
	("3", "4") + (1, 2) to Text is ("3", "4", "(1, 2)")
	("3", "4") + (1, 2) to Texts is ("3", "4", "1", "2")
	(1 + 2) * 3 is not 1 + 2 * 3
	(("1", "2") + (3, 4)) to Numbers is (1, 2, 3, 4)
	3 + (4) is (3, 4)
	(1) + ("Hi") is Error("Cannot downcast Text to Number for list: Text \"Hi\"")
	(1, 2, 3) + 4 is (1, 2, 3, 4)
	("Hello", "World") + 5 is ("Hello", "World", "5")
	(1, 2, 3) - (3) is (1, 2)
	(1, 2, 3) - 3 is (1, 2)
	(1, 2, 3) - 4 is (1, 2, 3)
	(2, 4, 6) / (2) is (1, 2, 3)
	(1, 2) / (2, 4) is (0.5, 0.5)
	(1) / (20, 10) is (0.05, 0.1)
	(2, 4, 6) / 2 is (1, 2, 3)
	(1, 2) * (3, 5) is (3, 10)
	(1, 2) * 3 is (3, 6)
	(1, 2, 3).Sum is 6
	(1, 2, 3).X is 1
	(1, 2).Y is 2
	(1, 2, 3).Z is 3
	(1, 2, 3, 4).W is 4
	(1, 2, 3).First is 1
	(1, 2, 3).Last is 3
	3 is in (1, 2, 3)
	3 is not in (1, 2)
	"b" is in ("a", "b", "c")
	"d" is not in ("a", "b", "c")
	(1, 2, 3).Index(2) is 1
	(1, 2, 3).Index(9) is -1
	(1, 2, 3).Remove(2) is (1, 3)
	(1, 2).Count(1) is 1
	(1, 3).Count(2) is 0
	("Hi", "Hello", "Hi").Count("Hi") is 2
```

## Members, variables, constants, let, mutable
Members are always defined at the top of any type file. Members are basically fields in other languages, but can do much more. For example almost all strict types are composed of more basic types and can be used as such. This kind of allows multiple inheritence (as you can have multiple members) and also gives you any feature from another type you want by just adding it is a member. If a member type is a trait (kind of interface, traits have no implementation), the type must implement all the trait methods.

Members can be either `has` (default), `constant` (for anything that does not change, think of static readonly, this includes way more than other programming languages and propagates upward, anything composed out of constants is a constant too, constants are always precomputed and optimized away, even if this is a complex operation, one calling methods runs once: when the types are loaded on the Strict VM, on its first use natively) or `mutable` (rarely used, this means the value is allowed to change, mostly for optimization purposes, should be avoided otherwise as parallization will struggle with mutables). Members and parameters can have a default value: a literal or a constant like `has activation = Activation.Sigmoid` (a type with one member, here a number, is that member), constants of types with several members are passed by every constructor call that leaves the member out. A copy of an instance with some members replaced is the type called with the instance followed by the replaced members by name, `Unit(unit, health = unit.health - damage)`, the other members come from `unit` (a name that is not a member is `MemberNotFound`).

A member of type `Generic` makes a type generic: every use is an instance of its own, with one type argument per `Generic` member in member order, taken from the constructor arguments. `Couple("a", 2)` (with `has first Generic` and `has second Generic`) is a `Couple(Text, Number)`, its `first` is a `Text`, it prints as `(a, 2)` and compares member by member. A method declared to return its own type returns what its body constructs: `Swapped Couple` with the body `Couple(second, first)` gives a `Couple(Number, Text)` here. Methods are compiled for each instance, so a call on a `Generic` member reaches the concrete type: with `has item Generic` and `Area Number` returning `item.Area`, `Holder(Size(2, 3)).Area` calls `Size.Area` (a member typed with a trait means the type implements that trait, holding any implementation is what `Generic` is for). See `Examples/Couple.strict` and `Examples/Holder.strict`.

Some examples:

Any.strict is the basis of every other type, it doesn't have to be added as a member, all types are always Any. You can never use Any directly, but it implements these methods automatically in every type.
```
from
to Type
to Text
to HashCode
is(other) Boolean
Members Variables
```
`Members` gives every stored member as a `Variable` (name, type and current value), `to HashCode` combines the hash codes of the members, the compiler writes both for a type when they are used.

A character is just a number, one byte of a text: texts are UTF-8, so letters outside of ASCII take several characters. It can be created from a single letter text or from its value:
```
has number
from(text)
	Character("b") is "b"
	Character(7) is Tab
	7 to Character is "7"
	Character("ab") is Error
	text.Characters(0)
```

A text ist just a List of characters:
```
has characters
```

A name is just a Text with some constraints
```
has text with Length > 1 and " " is not in value
```

NamedType is the base type for Members and Variables internally and this is all the code for it:
```
has Name
has Type
has IsMutable
has IsConstant
to Text
	NamedType("count", Number) to Text is "count Number"
	Name + " " + Type
```

Logging a value or adding it to a text uses its own `to Text` (`Process(C:\Tools\nasm.exe)`), a type without one shows its members in brackets, `(1, 2)`, also a type with one named member like `has step Number`, `(1)`. A value type whose one member is just its type (`has number`, `has text`) shows that value, `Byte(5)` is `5`.

Variables are always created in a method body scope. They look almost the same to members, but have a much smaller scope. They can be `constant` if the value is purely constant and never changes, this is always optimized away. Or it can be `let` (if the incoming value can be different, e.g. coming from a has member value that is different for each instance). `mutable` is also allowed here and rare. Only needed if a value needs to change, e.g. counting up in a loop. These kinds of mutable values are not dangerous as long as they don't leave the scope (then the same applies as for members, use rarely, only for specific optimizations, avoid otherwise).

Example:
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

`name = expression` (or `name(parameter = default) = expression`) for a name that is not a variable, member or parameter is a computed name: a single expression whose type is inferred, every use evaluates it again with the current values (`let` keeps a calculated value, `mutable` a reassignable one), effects run on every use. Assigning to it is a `ComputedNameMustNotBeAssigned` error. Only for very simple, obvious things (square roots, lengths, small sums of local values), one expression is not a tested method. Uses are replaced by the expression with the arguments for the parameters, the same on the Strict VM and natively. On a type, `Area = Width * Height` (or `Doubled(factor = 2) = Area * factor`) is a method without tests whose return type is inferred from the expression (`Examples/ComputedRectangle.strict`):
```
	mutable position = 3
	next(add = 1) = position + add
	next is 4
	position = 10
	next is 11
	next(2) is 12
```

## if expressions
Strict only has `if` for conditional expressions, which can be in block and single-line forms, plus a selector-style variant for compact matching. Everything is an expression, `if` and `for` always return an expression as well, which can be used as an automated return expression as well. In for loops the `if` expression is often used to filter values like piping in functional languages, see below for more advanced examples.

```
if value > 0
	value
```

Single-line conditional are pretty much the same (as '?' in C, C# or Java, just using "then" like in F# or Haskell). This is often used to return or assign simple expressions based on a condition, it all must fit in a line (as always max. length is 100).
```
value > 0 then value else 0
```
C#:
```
var result = value > 0 ? value : 0;
```
Scala or Haskell:
```
val result = if value > 0 then value else 0
```

Finally, we have the selector `if`, it starts pretty much the same as above and also uses "then". The if line ends with "is" or "is not" (and other keywords or expressions that can be compared). It works like a switch or match expression in other languages, but is more powerful as you can use any expression and aretmetic or comparison operator or type check here.
```
if operation is
	"add" then value
	"subtract" then 0 - value
	"multiply" then value * value
```
C#:
```
var result = operation switch
{
	"add" => value,
	"subtract" => 0 - value,
	"multiply" => value * value
	_ => throw new InvalidOperationException("Unknown operation: " + operation)
};
```
Scala:
```
val result = operation match
	case "add" => value
	case "subtract" => 0 - value
	case "multiply" => value * value
	case _ => "Unknown operation"
```
Haskell:
```
let result = case operation of
	"add" -> value
	"subtract" -> 0 - value
	"multiply" -> value * value
	_ -> "Unknown operation: " ++ operation
```
It is important to note that if there is no fallback else case at the end an Error is automatically generated and thrown at runtime if no case above it matches.

# for
The only iterator in Strict is the "for" expression, it is usually used to enumerate through lists, but anything that implements Iterator can be enumerated (Enum, List, Number, Range and anything that implements List like Dictionary). The syntax for "for" is quite powerful and not much like c-style languages, it is more like in functional languages. A for loop is also usually the only spot where mutable variables are used and make sense, so a lot of optimizations are done here to avoid actual mutable usage at runtime and thus allowing almost all code to be executed without side effects and in parallel. Remember that you don't have to write any parallel code, threads, processes, async, etc. it is all done automatically be the runtime and way more powerful than any other programming language could provide. All the following examples are automatically executed in parallel and even on the GPU if available and if it would be faster to do so.

Unlike the above examples or any other code in strict for expressions are not as self-explanatory, so each use case is explained here in a bit more detail.

## Normal iteration
```
has logger
...
	constant someList = (1, 2, 3)
	for someList
  	logger.Log(index + ": " + value)
```
Outputs this into the logger (defaults to the console):
0: 1
1: 2
2: 3

The for expression requires an iterator to go through each element, which is what someList is. There is no need to specify an enumerator index or value as this is done automatically, and you can access these via "index" (0, 1, 2, 3, etc.) and "value" (whatever value is at this index). If you need to access the outer value you can use "outer.value".

`for as long as condition` repeats its body while the condition is true, the condition is checked before every iteration and `index` counts the iterations (this is what a while loop does elsewhere). A `for` over a value that can not be iterated, like a Boolean or a type without a `for` iterator, is a `NotIterable` error:
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

You can iterate over a number if you just want a quick for loop for let's say 10 iterations. This outputs 10 lines starting from 0 to 9, index is available and the outer value stays unchanged and isn't used or needed here:
```
  for 10
    logger.Log(value)
```

Iterating over a Range will start at the Range.Start and end before the Range.ExclusiveEnd is reached. This outputs 2, 3, 4 into 3 lines:
```
	for Range(2, 5)
  	logger.Log(value)
```

Or you can enumerate over Enum values, lets say you have Connection.strict (if there are only constants in a type, it is an enum, you don't have to give them values, but here they are all Text values)
```
constant Google = "https://google.com",
constant Microsoft = "https://microsoft.com"
```
Now we can iterate over this:
```
for Connection
  logger.Log(value)
```
Notice that iterating over an Enum value is something else, if the Enum is a number, it will just iterate to that number like in the above example "for 10". If the Enum value is a Text string, it will iterate over each letter as Text is just a List of Character.
```
for Connection.Google
  logger.Log(value)
```

## Custom enumerator variables
```
has logger
LogNumbers
	for element in (1, 2, 3, 4)
		logger.Log(element)
```
Outputs 1 to 4 in 4 lines. Index still can be used, but value wasn't used or overwritten, so you can access the outer value this way. A more complex example would be:
```
for x in Range(1, 10)
  for y in Range(1, 10)
    logger.Log((x, y))
```
which would output 100 values starting at (0, 0) ending at (10, 10). Notice that Strict will evaluate the fastest execution, so for image processing it is not needed to iterate y first. Here it makes no difference. Since output logging is a side effect this can't be executed fully in parallel (the logger will still get all values at once, but in order and output them at once as pretty much no time will have passed from start to finish).

## Dictionaries
For dictionaries, you might want to use index, key and value variables, but again: you don't have to. This works fine:
```
for someDictionary
  if value.Key is "Apple"
    return value.Value
```

Dictionaries are just Lists of key+value pairs, so each enumeration just returns one of these pairs, and we can grab the first value as the key with value(0), which is exactly what value.Key will do. Same goes for value(1) for value.Value.

`prices("Apple")` reads the value of a key (`get`, a missing key is the `KeyNotFound` Error), `prices("Apple") = 4` on a `mutable` dictionary replaces the value of an existing key or adds the pair (`set`), like an element write of a list it updates the dictionary in place. Any type with a `get(key)` method is read the same way (`json("bones")` calls `Json.get`, a chained call needs `.get("bones")`):
```
mutable counts = Dictionary(Text, Number)
for words
	counts(value) = (value is in counts then counts(value) else 0) + 1
```

```
Index(other Generic) Number
	(1, 2, 3).Index(2) is 1
	(1, 2, 3).Index(9) is -1
	for elements
		if value is other
			return index
	-1
```
Unlike the for examples below that use a more compact syntax, here we want to manually abort the for loop iteration when an element is found. Since the list is not sorted by the other value we search for here, we have to iterate the whole elements list and then abort when we found the value. With the index method we are interested in the index of that value, so that is what is returned here (or -1 if nothing is found).

If you look at the source code of Dictionary.strict you might be confused how simple it is, all the optimizations happen at runtime, the code in Strict.Base just shows how the types are functionally working. At runtime everything is optimized. That is also why there are no other collections needed yet, no Stack, LinkedList, Queue, SortedDictionary, HashTable, whatever. You can just use List and Dictionary as is and at runtime Strict will figure out what internal structure makes most sense based on how it is used. This of course means if you mix and match different types like using a queue or stack, but also randomly adding or removing elements in the middle, it can't be as fast. The biggest impact on optimizations is if mutable types are used or not, by default nothing should be mutable in strict (and most mutables can be removed in the Executor so code stays functional and fast), but sometimes you might want to handcraft a mutable iteration like for image processing. If Strict can figure out that your input image stays unchanged and your output is just assigned back to the input, it can remove all mutable automatically and just swap two images (double buffering).

## Image processing
```
has brightness Number with value is not 0
Process(image) Image
	constant testImage = Image(Size(1, 1), (Color(0, 1, 2)))
	AdjustBrightness(1).Process(testImage) is Image(Size(1, 1), (Color(1, 2, 3)))
	AdjustBrightness(5).Process(testImage) is Image(Size(1, 1), (Color(5, 6, 7)))
	mutable adjusted = image
	for adjusted.Size
		adjusted.Colors(index) = GetBrightnessAdjustedColor(adjusted.Colors(index))
	adjusted
GetBrightnessAdjustedColor(currentColor Color) Color
	AdjustBrightness(1).GetBrightnessAdjustedColor(Color(0, 1, 2)) is Color(1, 2, 3)
	AdjustBrightness(5).GetBrightnessAdjustedColor(Color(0, 0, 0)) is Color(5, 5, 5)
	AdjustBrightness(-5).GetBrightnessAdjustedColor(Color(0, 0, 0)) is Color(0, 0, 0)
	Color(current.Red + brightness, current.Green + brightness, current.Blue + brightness)
```

If the iterator is multidimensional like Size, it can be used in a for loop and multiple variables can be named (value would here be a list of 2 values like for Dictionary):
```has iterator Iterator(Vector2)
has Width Number with value > 0
has Height Number with value > 0
for Iterator(Vector2)
	for Height
	for Width
		Vector2(x, y)
...
```
Notice that Size(0, 0) or Size(-1, 5) would not be valid as all dimensions must be positive numbers as defined above. The iterator is also returning a Vector2, which is just a list with 2 numbers.

## For return value
You might already have noticed that for loops can be used as the return value for a method. In the Vector2.for implementation we just have 2 for loops and inside just the Vector2(x, y) is constructed. The method however returns an Iterator(Vector2) which is used in the image processing example to iterate through all possible pixel values of an image.

Here a few more examples as this is used in most useful for loops
```
in(key Generic) Boolean
	2 is in Dictionary((1, 1), (2, 2))
	3 is not in Dictionary((1, 1), (2, 2))
	for keysAndValues
		value.Key is key
```
Here the dictionary with keysAndValues is iterated. On each iteration the for body is executed, which here is just a comparision if the value.Key is the given key value. If that evaluation is true, the for loop is aborted and true is returned (we found the key). If the whole loop finishes and no key was found, the whole for expression returns false. In strict the last line in a method is always automatically the return expression. "return" is only needed when it happens in any line above it.

```
GetElementsText Text
	(1, 3).GetElementsText is "1, 3"
	for elements
		(index is 0 then "" else ", ") + value
```
This method goes through the elements and does a quick check if we are at the first element via "index is 0", if yes, then an empty Text string is used, else ", " is used. Then that value is concatenated via "+ value", where value is the current value from the for enumeration. The result of the for body expression is a Text (either just the value as Text for the first iteration or ", " + value otherwise). The same way for expression boolean value was used in the previous example, this Text value is now used to create a new list (since we didn't specify + at the beginning of the for expression, see below for more examples). So the return value of the for expression is a list of Text, which is not what we asked for, we wanted a Text, so Strict automatically concatenates the list into a single Text value, which is then returned. Btw: This implementation would be stripped out of the code and be replaced with the existing Text.Combine method!

```
Length Number
	(1, 2).Length is 2
	for elements
		1
```
The same concept can be used to count up some numbers, here the for expression results in a number 1 for each iteration. The for expression returns a list of numbers, all 1, which is then concatenated into the Length Number that we wanted to know.

Sum and Count work the same way, see if you can figure it out:
```
Sum Generic
	(1, 2, 3).Sum is 6
	for elements
		value
```
```
Count(searchFor Generic) Number
	(1, 2).Count(1) is 1
	(1, 3).Count(2) is 0
	("Hi", "Hello", "Hi").Count("Hi") is 2
	for elements
		if value is searchFor
			1
```

You can of course also use any other operator instead of the default creation of a list and concatentation. The above Sum method would return the same value if it were written as:
```
Sum Generic
	for elements
		+ value
```
However, Strict is VERY picky and strict (hence the name) on how to do things and would immediately replace it with the easier above version (as it needs less code).

## Functional programming
But maybe you want to do some other operation in the for body, like multiplying:
```
Product Generic
	for elements
		* value
```
As you can see this is more like a pipe operator in functional languages (Lisp, Clojure, F#, Haskell, etc.). Let's say we want to do some filtering, mapping and reducing. Let's look at some different languages, all of them pretty much do the same thing, the syntax is just slightly different:

C#
```
int[] numList = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
int result =
    numList
        .Where(n => n % 2 == 0)
        .Select(n => n * 10)
        .Sum();
```
Haskell
```
numList :: [Int]
numList = [0..9]
result :: Int
result =
    sum
      . map (* 10)
      . filter even
      $ numList
```
Clojure
```
(def num-list [0 1 2 3 4 5 6 7 8 9])
(def result
  (->> num-list
       (filter even?)
       (map #(* 10 %))
       (reduce + 0)))
```
F#
```
let numList = [| 0..9 |]
let result =
    numList
    |> Array.filter (fun n -> n % 2 = 0)
    |> Array.map (fun n -> n * 10)
    |> Array.sum
```
Javascript
```
const numList = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
const result = numList
    .filter(n => n % 2 === 0)
    .map(n => n * 10)
    .reduce((acc, n) => acc + n, 0);
```
Python
```
num_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
result = sum(n * 10 for n in num_list if n % 2 == 0)
```

The python example is a bit more compact, but you have to get used to the syntax as it does the mapping first and the filtering last, unlike all other examples.

Here is the same example in Strict (creates a list of numbers, which is then concatenated automatically for us using List.Sum or the identical output via + concatenation as in the example above):
```
for Range(0, 10)
  if value % 2
    value * 10
```

## More examples
Let's look at a more complex example (not really useful, but is shows how we can append more things to for enumerations), think in similar terms to the functional examples above.
```
  for numbers
		to Text
		Length * Length
		if value > 3
			4 + value * 3
			to Text
```
Here we start iterating through a list of numbers, then we convert each value to Text (value is always used by default on any call, no matter if we are in a method were it defaults to our type instance or in a for loop, where it is the current iteration value).Then we use Length on the new value inside the for body, which is now a Text, so that would be the length of that Text and multiply it by itself. That length is a number again, the if expression now checks if that value is above 3. If so we go into the inner if body and do a quick calculation, and finally we convert the resulting number to Text, which is then added to the for expression result. If the return value is a List of Text that would be returned, if it is a Text, then everything is concatenated. Otherwise, an error is given with the incompatible type. Also note that some of these things could be done in one line like (value to Text).Length * (value to Text).Length, but that would be longer and is not preferred. (4 + value * 3) to Text is probably slightly shorter and would be preferred, but the point here is you can call whatever code you like to transform any Iterator.

Here is the final example from Strict.Examples/RemoveParentheses.strict
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

## Blog

Details at https://strict-lang.org/blog

## History

The idea to write my (Benjamin Nitschke, now CEO of Delta Engine) own programming language originally started in the 1990s after reading my favorite book "Gödel, Escher, Bach" by Douglass R. Hofstadter as a child. Due to lack of time and experience, it wasn't really much more than a bunch of small programs and proof of concept ideas. In 2008–2010 the first iteration was built, and while it worked, it lacked integration and was discarded in favor of Lua. This repository was started 2020-06-16 just for experimentation to create editor plugins and rethinking some of the low-level parsing, using the [old Strict sdk](https://github.com/strict-lang/sdk) as a guide. In mid 2025 I am back working on it alone after a few years of pause while I had some help in 2020–2022 from some employees, in the end the parent project was abandoned. There wasn't a need for creating neural networks and doing image processing with this. There were too many ready-to-use solutions to compete against, and the project ran out of funding and man power.

Now it is just a project to work a little on in the evenings and weekends if I find the time for it. I will continue to the original goal of creating a base layer language computers can actually understand and work on top of. I am thinking more in the way DNA and proteins are the base building blocks of life, but day to day we don't really think about them that much.
