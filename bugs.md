# Bugs

- `strict test Ladder.strict Advised native` in Ben (also `Consulted`) dies after 41 s with AccessViolation at `Advisor.Tiny`, `Advisor.None`, `Ladder.$Tests`: the constant loads `NeuralNetworks/tinyqwen3.gguf` (`Qwen3.Loaded(path, 32)`), `strict test Advisor.strict native` passes, reached from another type's test executable it crashes.
