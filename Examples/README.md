# Strict.Examples
Small programs showing what Strict can do today, also a playground for katas and new language features.

Start with `HelloLogger.strict`, `Sum.strict` (`strict Examples/Sum.strict 5 10 20` prints 35) and `FizzBuzz.strict`, then `Couple.strict` for generic types. Every method starts with its tests: `strict test Examples/FizzBuzz.strict` runs them on the Strict VM, `strict build Examples/Sum.strict` builds the native executable without running tests. Add `fast` for unoptimized host code, or `small` to optimize for size.
