# Strict

![Strict Programming Language](https://strict-lang.org/img/StrictBanner.png)
<div style="text-align: center;">Create well-written and efficient code quickly</div>

## Strict is a language computers can understand

Strict is a simple-to-understand programming language that not only humans can read and understand, but also computers are able to understand, modify and write it.

The long-term goal is NOT to create just another programming language, but use Strict as the foundation for a higher level language for computers to understand and write code. This is the first step towards a future where computers can write their own code, which is the ultimate goal of Strict. Again, this is very different from other programming languages that are designed for humans to write code but not for computers to understand it. Even though LLMs can be trained to repeat and imitate, which is amazing to see, they still don't understand anything really and make the most ridiculous mistakes on any project bigger than a handful of files.) Strict needs to be very fast, both in execution and writing code (much faster than any existing system), so millions of lines of code can be thought of by computers and be evaluated in real time (seconds).

## Language Server Protocol

The LSP (Language Server Protocol) implementation here adds support for VSCode, VS2019, IntelliJ and many more IDEs as possible Editors for Strict.

## Layers

![Strict Layers](https://strict-lang.org/img/StrictLayers.svg?sanitize=true)

## Architecture at a Glance

| Layer | Responsibility | When It Runs | Rejects / Optimises |
|-------|----------------|-------------|---------------------|
| **Strict.Language** | Load *.strict* files, package handling, core type parsing (structure only) | File load | Bad formatting & package errors |
| **Strict.Expressions** | Lazy-parse method bodies into an immutable tree of `if`, `for`, assignments, calls… | First method call | Structural/semantic violations |
| **Strict.Validators** | Visitor-based static analysis & constant folding | After expression parsing | Impossible casts, unused mutables, foldable constants |
| **Strict.Runtime** | Execute expression tree via ~40 instructions | On demand by the caller | Value/type errors, side-effects |
| **Strict.TestRunner** | Run in-code tests (`method must start with a test`) | First call to a method | Failing test expressions |
| **Strict.Optimizers** | Strip passed tests & other provably dead code | After first successful run | Dead instructions, test artefacts |

## 1 · Strict.Language
* Parses package headers and type signatures only.
* Keeps every package cached in memory for instant access.
* Any malformed file is rejected before deeper compilation.

## 2 · Strict.Expressions
* Method bodies are parsed _lazily_ the first time they’re needed.
* Produces an **immutable** expression tree — never mutated at runtime.
* Early structural checks ensure only valid Strict constructs survive.

## 3 · Strict.Validators
* Runs constant folding & simple propagation (e.g. `"5" to Number → 5`).
* Flags impossible casts (`"abc" to Number`) and other static mistakes.
* Separation of concerns: validation without touching runtime state.

## 4 · Strict.Runtime
* The Runtime is the single source of truth for dynamic behaviour.
* Executes ~40 byte-code-like instructions as statements, all in memory for now.
* Very fast, the goal is to run this in Strict itself and go through millions of lines per second.

## 5 · Strict.TestRunner
* Every method must open with a self-contained test.
* Tests are executed once; passing expressions become `true` and are pruned.
* Guarantees that only verified code reaches production.

## 6 · Strict.Optimizers
* Final clean-up pass: Remove passed test code.
* Eliminate unused variables created solely for testing.
* Preserve original line numbers for precise error reporting.

Note: There are older projects like the VS Code LanguageServer, Cuda Compiler, Roslyn, Grammar, etc. but these are no the focus here, we just want to get the Runtime working, switch all the code to Strict itself and let it rip then ^^

---

### Key Design Decisions

1. **Immutable Expression Tree**
   Static analysis only; no runtime state sneaks in.

2. **Validation Before Execution**
   Catch the trivial stuff early, let the Runtime handle the hard parts.

3. **Runtime Is King**
   Only the VM sees real values and side-effects — optimisation happens post-execution.

4. **One-Time Tests**
   Tests live where the code lives, run once, then disappear.

---

### Why This Matters

Only needed code that is actually called is ever parsed, validated, tested and runs in the Runtime. This is a very different approach and allows generating millions of .strict lines of code and discarding anything that is not needed or doesn't work pretty much immediatly.
* **Fast Feedback** — formatting and type errors fail instantly, long before runtime.
* **Deterministic Builds** — immutable trees + one canonical code style = no drift.
* **Lean Runtime** — by the time code hits the Runtime, dead weight is already gone.

---

Happy coding with **Strict** — where there’s exactly one way to do it right.

## Blog

Details at https://strict-lang.org/blog

## History

The idea to write my (Benjamin Nitschke, now CEO of Delta Engine) own programming language originally started in the 1990s. Due to lack of time and experience, it wasn't really much more than a bunch of small programs and proof of concept ideas. In 2008–2010 the first iteration was built, and while it worked, it lacked integration and was discarded in favor of Lua. This repository was started 2020-06-16 just for experimentation to create editor plugins and rethinking some of the low-level parsing, using the [old Strict sdk](https://github.com/strict-lang/sdk) as a guide. In mid 2025 I am back working on it alone after a few years of pause while I had some help in 2020–2022 from some employees, in the end the parent project was abandoned. There wasn't a need for creating neural networks and doing image processing with this. There were too many ready-to-use solutions to compete against, and the project ran out of funding and man power.

Now it is just a project to work a little on in the evenings and weekends if I find the time for it. I will continue to the original goal of creating a base layer language computers can actually understand and work on top of. I am thinking more in the way DNA and proteins are the base building blocks of life, but day to day we don't really think about them that much.
