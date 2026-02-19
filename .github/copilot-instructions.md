# Copilot Instructions

## Project Guidelines
- In Strict, type instance equality should check type compatibility and then compare member values (including list/dictionary members) rather than reference equality.
- Aim for a proper root-cause fix rather than a quick workaround or “duct-taping.”
- Apply Limit.cs rules with a ~2x multiplier for C#. Classes should ideally be between 100-400 lines, but classes above 500 lines should be split for better maintainability.
- When splitting `Executor.cs`, keep high-level methods there (Execute, exceptions, DoArgumentsMatch, stackoverflow detection, arguments/instances/parameters handling, RunExpression), and move expression evaluators into separate classes unless they are single-line/simple.
- Do not add new methods to `SpanExtensions` without asking first; keep refactors focused and fix one issue at a time.

## Project-Specific Rules
- Strict is a simple-to-understand programming language that not only humans can read and understand, but also computers are able to understand, modify and write it.
- The long-term goal is NOT to create just another programming language, but use Strict as the foundation for a higher level language for computers to understand and write code. This is the first step towards a future where computers can write their own code, which is the ultimate goal of Strict. Again, this is very different from other programming languages that are designed for humans to write code but not for computers to understand it. Even though LLMs can be trained to repeat and imitate, which is amazing to see, they still don't understand anything really and make the most ridiculous mistakes on any project bigger than a handful of files.) Strict needs to be very fast, both in execution and writing code (much faster than any existing system), so millions of lines of code can be thought of by computers and be evaluated in real time (seconds).
- Use dotnet build to build, dotnet test to test, do not try to use VS building or ReSharper build, you always fail trying to do that.

## Strict Semantics
- When asked about Strict semantics, derive behavior directly from README.md and TestPackage.cs examples; re-check cited examples before answering and avoid contradicting them.
