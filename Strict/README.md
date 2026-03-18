# Strict executable

Runs though all stages in the Strict workflow to execute a strict file. Can also create an optimized executable for any supported platform with parallel CPU and GPU (CUDA, more later) execution support. Will give you statistics and performance metrics in debug mode, in release just gives the final output.

## Purpose

This tool demonstrates and tests the complete Strict workflow from source code to execution (step 2-5 are skipped if diagnostics is off in release mode):

1. **Load Strict Package**
   - Loads the Strict base package from the main directory
   - Parses type signatures and members
   - Validates package structure

2. **Load Package and Parse Types**
   - Loads .strict files from a package directory
   - Parses type signatures and members
   - Validates package structure

3. **Parse Method Bodies** (Strict.Expressions)
   - Lazy-parses method bodies into immutable expression trees
   - Handles `if`, `for`, assignments, calls, and all Strict constructs
   - Performs early structural validation

4. **Run Validators** (Strict.Validators)
   - Static analysis and constant folding
   - Detects unused variables, impossible casts, and other issues
   - Ensures code quality before execution

5. **Run Tests** (Strict.TestRunner + HighLevelRuntime)
   - Executes all inline tests (every method must start with tests)
   - Uses interpreter-style execution for fast feedback
   - Verifies all test expressions evaluate to true

6. **Generate Bytecode** (Strict.Bytecode)
   - Converts expression trees to ~40 bytecode-like instructions
   - Optimizes for execution speed
   - Prepares for low-level runtime

7. **Optimize Bytecode** (Strict.Optimizers)
   - Removes passed test code
   - Performs constant folding, dead store elimination
   - Eliminates unreachable code and redundant loads

8. **Execute Bytecode** (Strict)
   - Runs optimized bytecode via VirtualMachine
   - Provides final output and return values
   - Very fast execution speed, but not as fast as precompiled CPU and GPU optimized executable below.

9. **Create optimized executable** (Strict.Compilers.Assembly)
   - Create executable for Windows, Linux or MacOS
   - Maximum execution speed, parallelizes any complex code automatically, runs on GPU if available.

## Usage

```
Usage: Strict <file.strict|.strictbinary> [-options] [args...]

Options (default if nothing specified: build .strictbinary cache and execute in VM)
  -Windows     Compile to a native Windows x64 optimized executable (.exe)
  -Linux       Compile to a native Linux x64 optimized executable
  -MacOS       Compile to a native macOS x64 optimized executable
  -mlir        Force MLIR backend (default, requires mlir-opt + mlir-translate + clang)
               MLIR is the default, best optimized, uses parallel CPU and GPU (Cuda) execution
  -llvm        Force LLVM IR backend (fallback, requires clang: https://releases.llvm.org)
  -nasm        Force NASM backend (fallback, less optimized, requires nasm + gcc/clang)
  -diagnostics Output detailed step-by-step logs and timing for each pipeline stage
               (automatically enabled in Debug builds)
  -decompile   Decompile a .strictbinary into partial .strict source files
               (creates a folder with one .strict per type; no tests, optimized)

Arguments:
  args...      Optional text or numbers passed to called method
               Example to call Run method: Strict Sum.strict 5 10 20 => prints 35
               Example to call any expression, must contain brackets: (1, 2, 3).Length => 3

Examples:
  Strict Examples/SimpleCalculator.strict
  Strict Examples/SimpleCalculator.strict -Windows
  Strict Examples/SimpleCalculator.strict -diagnostics
  Strict Examples/SimpleCalculator.strictbinary
  Strict Examples/SimpleCalculator.strictbinary -decompile
  Strict Examples/Sum.strict 5 10 20
  Strict List.strict (1, 2, 3).Length

Notes:
	Only .strict files contain the full actual code, everything after that is stripped,
	optimized, and just includes what is actually executed (.strictbinary is much smaller).
  Always caches bytecode into a .strictbinary for fast subsequent execution.
  .strictbinary files are reused when they are newer than all of the used source files.
```

### Example

```
Strict Examples/SimpleCalculator.strict
```

## Output

The runner provides detailed progress through each pipeline stage.

```
Strict.Runner: Sum.strict
Parse: 42 methods, Total expressions: 156
...
```

## Exit Codes

0 on success, execution or build successful.

1 if there is a failure, error details are in console output

## Use Cases

- **Strict runtime**: This contains everything needed to run strict, run tests, compile it, build executables.
- **IDE**: Used by any ide that supports Strict. See Strict.LanguageServer for details.
- **Testing the pipeline**: Verify all components work together
- **Debugging**: See exactly which stage fails and why
- **Performance analysis**: Measure time spent in each stage
- **Learning**: Understand how Strict processes code end-to-end
- **Development**: Quick way to test changes across all pipeline stages

## Architecture

See the main [README.md](../README.md) for detailed architecture documentation, including:
- Immutable expression trees
- Validation before execution
- One-time test execution
- Runtime optimization strategies