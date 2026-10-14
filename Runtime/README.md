# Strict.Runtime
The Strict VM: `strict <file>` and `strict test` run the instructions of `Bytecode` here without building an executable. `Execute` loads the types, `Resolver` numbers the locals and registers of every method into slots, resolves call targets and jumps and decodes constants, `Machine` calls methods and binds members and arguments, `Frame` runs one call in its own list of slots (a call of a method that only returns a member, like `Vector2.X`, reads it through `Accessor` without a call), `Looping`, `Operations` and `Storage` carry out the instructions, `Natives` the methods of the base types the VM implements itself.

Speed: 1M loop iterations take 105 ms (a native executable 40 ms), training `NeuralNetworks/Iris.strict` 2.4 s.
