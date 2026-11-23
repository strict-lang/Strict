# Bugs

- `Compiler/MlirParallel.strict` IsParallel test `Doubling.IsParallel and Logging.IsParallel and Randomizing.IsParallel and Timing.IsParallel is false` fails on the VM with `strict test Compiler/MlirParallel.strict unit` (got false, also on HEAD 2026-11-23); `strict check Compiler` does not run it (internal package tests are skipped by folder checks).
