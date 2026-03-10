# Strict executable

Runs though all stages in the Strict workflow to execute a strict file. Will give you statistics
and performance metrics in debug mode, in release just gives the final output.

## Purpose

This tool demonstrates and tests the complete Strict workflow from source code to execution:

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

6. **Generate Bytecode** (Strict.Runtime)
   - Converts expression trees to ~40 bytecode-like instructions
   - Optimizes for execution speed
   - Prepares for low-level runtime

7. **Optimize Bytecode** (Strict.Optimizers)
   - Removes passed test code
   - Performs constant folding, dead store elimination
   - Eliminates unreachable code and redundant loads

8. **Execute Bytecode** (Strict.Runtime)
   - Runs optimized bytecode via BytecodeInterpreter
   - Provides final output and return values
   - Maximum execution speed

## Usage

```bash
Strict <strict-file>
```

### Example

```bash
Strict Examples/SimpleCalculator.strict
```

## Output

The runner provides detailed progress through each pipeline stage:

```
╔═════════════════════════════════════════════════╗
║           Strict Programming Language           ║
╚═════════════════════════════════════════════════╝

┌─ Step 1: Load Package and Parse Types (Strict.Language)
│  ✓ Loaded package: Strict.Base
│  ✓ Found type: Number
│  ✓ Members: 15
│  ✓ Methods: 42
└─ Step 1 Complete

┌─ Step 2: Parse Method Bodies (Strict.Expressions)
│  ✓ Parsed 42 methods
│  ✓ Total expressions: 156
└─ Step 2 Complete

...

╔═════════════════════════════════════════════════╗
║              Pipeline Complete ✓                ║
╚═════════════════════════════════════════════════╝
```

## Exit Codes

- **0**: Pipeline completed successfully
- **1**: Pipeline failed (error details printed to console)

## Use Cases

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
