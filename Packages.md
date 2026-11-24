# Package Split Proposal

Proposal only — no files moved yet. Splits all 6 folders over 50 `.strict` files into feature sub-packages of ~24 files (all well under the 50-file limit), defines allowed dependencies as a DAG, and breaks the main cycles found by `AcyclicDependencies`.

---

## 1. Breaking the Biggest Cycles First

Today `Strict` has 30 folders, 67 production edges, and **44 cycle edges** (`AcyclicDependencies`):
1. **Root folder (`.`, 45 files) mixes base types and `Strict.strict` CLI**: every package uses `Number`, `Text`, `List`, `File`, etc. in `.`, while `Strict.strict` in `.` depends on `Compiler`, `Runtime`, `Bytecode`, `Expressions`, `Validators`, and `Networking`.
   - **Fix**: move `Strict.strict` (CLI entry) to `Cli/Strict.strict` (or `App/Strict.strict`). Root `.` then has 44 base/stdlib types with **0 outgoing package dependencies**.
2. **`Runtime` -> `NeuralNetworks` / `Compiler`**: test fixtures or hosted helpers in `Runtime` reach up into `NeuralNetworks` / `Compiler`.
   - **Fix**: keep `Runtime` depending only on `.`, `Math`, `Json`, `Networking`, `Bytecode`, `Expressions` (never `Compiler`, `NeuralNetworks`, or `Cli`).
3. **`Expressions` <-> `Bytecode` <-> `Validators` <-> `Compiler`**:
   - **Fix**: strict layering:
     `.` -> `Math`/`Json`/`Xml`/`Compression`/`Networking`/`ImageProcessing`/`Regex` -> `Expressions` -> `Bytecode` -> `Validators` -> `Runtime` -> `Compiler` -> `Cli` / `NeuralNetworks` / `Examples`.
   - Move any `Compiler`/`Validators` references out of `Expressions` and `Bytecode` (e.g., test fixtures using `Compiler/output/...` or `Validators` helpers).

---

## 2. Sub-Package Splits (~24 files each)

### `Expressions` (54 -> 3 packages)
- **`Expressions` (20)**: `BodyParser`, `Constraint`, `Declaration`, `HeaderEntries`, `HeaderTokens`, `IndexedValues`, `Indexing`, `LoopEdge`, `LoopNames`, `NamedIterators`, `Postfix`, `RangeArguments`, `Scope`, `ScopedStatement`, `StatementParser`, `SyntaxNode`, `SyntaxParser`, `Tokenizer`, `TypeChoice`, `TypeShape`.
  - *Allowed deps*: `.`
- **`Expressions/Typing` (22)**: `ArgumentTypes`, `BranchTypes`, `Callees`, `Changes`, `Composition`, `ConstantCalls`, `CrossChanges`, `Defaults`, `Definitions`, `FusedAverage`, `Fusion`, `Inferring`, `Instantiation`, `KnownTypes`, `ListFits`, `MethodBodies`, `Overloads`, `Return`, `ReturnTypes`, `ShapeTyping`, `Steps`, `TypeInference`.
  - *Allowed deps*: `.`, `Expressions`
- **`Expressions/Checks` (12)**: `Corners`, `CountedIterators`, `DefaultCheck`, `ErrorNarrowing`, `FileCheck`, `HiddenType`, `Memo`, `Monotone`, `MutatingCheck`, `ResolveCheck`, `TextJoins`, `Unswitching`.
  - *Allowed deps*: `.`, `Expressions`, `Expressions/Typing`

### `Runtime` (71 -> 3 packages)
- **`Runtime/Vm` (24)**: `Accessor`, `Allocations`, `Baseline`, `CheckFrames`, `ConstantCache`, `ConstantLoads`, `Constants`, `ErrorCheck`, `Foreign`, `Forwarding`, `Frame`, `FrameSlots`, `Invocation`, `KeptMethod`, `Looping`, `Machine`, `MachineView`, `Natives`, `Operations`, `PayloadDecoder`, `ProgramRun`, `Rounding`, `SampledFrame`, `VmValue`.
  - *Allowed deps*: `.`, `Math`, `Expressions/*`, `Bytecode/*`
- **`Runtime/Hosted` (24)**: `Admission`, `Answer`, `CommandLimit`, `Coordinator`, `Hosted`, `HostedBudget`, `HostedJson`, `HostedMath`, `HostedProcess`, `HostedRegex`, `HostedText`, `HostedTokenizer`, `JobCancel`, `Need`, `Peers`, `Placement`, `Profile`, `Remote`, `RemoteRun`, `Resident`, `Resolver`, `Routes`, `ServerFamily`, `Shortcuts`.
  - *Allowed deps*: `.`, `Math`, `Json`, `Regex`, `Networking`, `Bytecode/*`, `Runtime/Vm`
- **`Runtime/Testing` (23)**: `CheckTests`, `Execute`, `Folders`, `Mutants`, `Mutation`, `MutationCache`, `MutationFlow`, `MutationKinds`, `Sampler`, `ScriptCache`, `Scripted`, `SlowTests`, `Storage`, `StreamedLines`, `TestBudget`, `TestRequest`, `TracedFrame`, `TraceEntry`, `TrialMachine`, `TrialRun`, `ValueCode`, `ValueSearch`, `ValueTrace`.
  - *Allowed deps*: `.`, `Expressions/*`, `Bytecode/*`, `Validators`, `Runtime/Vm`, `Runtime/Hosted`

### `Bytecode` (80 -> 4 packages)
- **`Bytecode/Binary` (16)**: `BinaryFile`, `BinaryLinks`, `BinaryNames`, `BinarySources`, `BuildOutput`, `ByteEncoder`, `Crc`, `DataFiles`, `EntryDecoder`, `InstructionEntry`, `InstructionType`, `MemberEntry`, `MethodEntry`, `TypeEntry`, `ZipEntry`, `ZipReader`, `ZipWriter` (17).
  - *Allowed deps*: `.`, `Compression`
- **`Bytecode/Linking` (17)**: `Automatic`, `DefaultInstance`, `FileCompiler`, `InferenceCache`, `KeptInference`, `LinkCompile`, `Linker`, `LinkFresh`, `LinkHomes`, `LinkRound`, `Mentions`, `OptionMember`, `OptionsCodegen`, `Repositories`, `Signature`, `SourceFiles`, `TypeFolders`, `TypeSource` (18).
  - *Allowed deps*: `.`, `Expressions/*`, `Bytecode/Binary`
- **`Bytecode/Codegen` (24)**: `Borrowing`, `BranchValues`, `CallTarget`, `ChangedArguments`, `ChangedInstance`, `CodeBlock`, `ConversionCodegen`, `ErrorCodegen`, `ErrorScope`, `IndexCodegen`, `KindTests`, `ListCodegen`, `LoopCodegen`, `MethodCodegen`, `NamedArguments`, `OperatorCodegen`, `SelectorCodegen`, `TestCodegen`, `TestComparison`, `TestLines`, `TestReports`, `TypeCodegen`, `TypeSums`, `ValueCodegen`.
  - *Allowed deps*: `.`, `Expressions/*`, `Bytecode/Binary`, `Bytecode/Linking`
- **`Bytecode/Optimize` (21)**: `Closeness`, `ConstantFolding`, `ConstantMethods`, `FractionLoop`, `Inliner`, `InlineRules`, `InlineSplice`, `LaneCalls`, `LaneFlow`, `LaneHopes`, `LaneNesting`, `LaneSkips`, `LoadReuse`, `LoopHoisting`, `LoopPeeling`, `MemberAverage`, `Repeating`, `Specialization`, `StepRounding`, `StructLanes`, `TailCalls`.
  - *Allowed deps*: `.`, `Expressions/*`, `Bytecode/Binary`, `Bytecode/Codegen`

### `Compiler` (210 -> 9 packages)
- **`Compiler/Build` (24)**: `AndroidBuild`, `BuildProfile`, `CSharpBinding`, `Intermediates`, `IOSBuild`, `LibraryBuild`, `LibraryLists`, `LinuxBuild`, `MacOSBundle`, `MacOSTools`, `NativeBuild`, `NativeCache`, `NativeChunks`, `NativeCompiler`, `NativeFolder`, `NativeLibrary`, `NativeSources`, `Platform`, `Release`, `Running`, `Seed`, `SelfBuild`, `Toolchain`, `ToolRunner`, `WebBuild`, `WindowsTools` (26).
  - *Allowed deps*: `.`, `Expressions/*`, `Bytecode/*`, `Validators`, `Runtime/*`, `Compiler/Mlir`, `Compiler/Gpu`
- **`Compiler/Checks` (16)**: `CheckAudit`, `CheckBuild`, `CheckEmission`, `CheckGpuAttentionBf16`, `CheckGpuCache`, `CheckGpuConvolution`, `CheckGpuHadamard`, `CheckGpuJit`, `CheckGpuKept`, `CheckGpuResidency`, `CheckMemory`, `CheckParallel`, `CheckParallelMlir`, `CheckProfile`, `CheckThrottling`, `CodegenStamp`.
  - *Allowed deps*: `.`, `Compiler/Build`, `Runtime/*`
- **`Compiler/Gpu` (28)**: `CubinBuild`, `DeviceBuild`, `GpuBuilt`, `GpuChoiceSamples`, `GpuFunction`, `GpuLayerSamples`, `GpuModule`, `GpuNumberSamples`, `GpuParts`, `MetalBranch`, `MetalBuild`, `MetalCall`, `MetalCast`, `MetalFunction`, `MetalInstruction`, `MetalShader`, `MetalValue`, `MlirGpu`, `MlirGpuCheck`, `MlirGpuContract`, `PtxCache`, `PtxFunction`, `PtxGlobals`, `PtxLibraries`, `PtxMerge`, `PtxModule`, `WgslFunction`, `WgslShader`, `WgslStatement` (29 -> or move `Wgsl*` to `Compiler/Web` with `WebBuild`).
  - *Allowed deps*: `.`, `Bytecode/*`, `Compiler/Mlir`
- **`Compiler/Mlir` (26)**: `CallGraph`, `EmissionCache`, `MlirAlone`, `MlirAppended`, `MlirArguments`, `MlirAutomatic`, `MlirCached`, `MlirCalls`, `MlirCarried`, `MlirChanged`, `MlirChecks`, `MlirCodes`, `MlirConstants`, `MlirCost`, `MlirCounter`, `MlirDefaults`, `MlirDeferred`, `MlirEntry`, `MlirEquality`, `MlirFresh`, `MlirFunction`, `MlirGated`, `MlirGlobals`, `MlirLines`, `MlirModule`, `MlirPurity`.
  - *Allowed deps*: `.`, `Expressions/*`, `Bytecode/*`
- **`Compiler/MlirValues` (23)**: `MlirBits`, `MlirBytes`, `MlirCouples`, `MlirDestined`, `MlirDots`, `MlirDoubles`, `MlirDying`, `MlirElements`, `MlirEmbedded`, `MlirExact`, `MlirFlats`, `MlirForeign`, `MlirFormats`, `MlirHanded`, `MlirInstances`, `MlirLayout`, `MlirLibrary`, `MlirListDestined`, `MlirListOperators`, `MlirLists`, `MlirMemory`, `MlirNarrowing`, `MlirNatives`.
  - *Allowed deps*: `.`, `Bytecode/*`, `Compiler/Mlir`
- **`Compiler/MlirStructs` (22)**: `MlirOwned`, `MlirPacked`, `MlirPacking`, `MlirPairs`, `MlirParameters`, `MlirParts`, `MlirPassed`, `MlirRational`, `MlirReadText`, `MlirRecursion`, `MlirRounded`, `MlirSafety`, `MlirSingle`, `MlirSpent`, `MlirSplit`, `MlirStatements`, `MlirStructs`, `MlirTails`, `MlirTextLists`, `MlirTexts`, `MlirTyped`, `MlirUnseen`, `MlirValidated`, `MlirValued`, `MlirWideMembers`, `MlirWidths` (26).
  - *Allowed deps*: `.`, `Bytecode/*`, `Compiler/Mlir`, `Compiler/MlirValues`
- **`Compiler/MlirLoops` (23)**: `MlirFastLoop`, `MlirFeeding`, `MlirFiltering`, `MlirFlatCall`, `MlirGroupBatched`, `MlirGrouped`, `MlirGroupIndex`, `MlirGroups`, `MlirGuards`, `MlirKernel`, `MlirKernelBody`, `MlirKernelLaunch`, `MlirKinds`, `MlirKindTests`, `MlirLanes`, `MlirLoops`, `MlirMembered`, `MlirParallel`, `MlirRangeLoop`, `MlirSums`, `MlirTensors`, `MlirTokenLaunch`, `MlirTokens`, `MlirTokenSamples` (24).
  - *Allowed deps*: `.`, `Bytecode/*`, `Compiler/Mlir`, `Compiler/MlirValues`, `Compiler/MlirStructs`
- **`Compiler/MlirDevice` (11)**: `MlirDevice`, `MlirDeviceCalls`, `MlirDeviceChoice`, `MlirDeviceDot`, `MlirDeviceLoops`, `MlirDeviceMembers`, `MlirDeviceRecords`, `MlirDeviceRows`, `MlirDeviceStatements`, `MlirDeviceSums`, `MlirCalleeGuards`.
  - *Allowed deps*: `.`, `Compiler/Mlir`, `Compiler/MlirLoops`
- **`Compiler/MlirWhole` (20)**: `MlirWhole`, `MlirWholeBody`, `MlirWholeCall`, `MlirWholeChild`, `MlirWholeCounted`, `MlirWholeDescribed`, `MlirWholeDivision`, `MlirWholeFields`, `MlirWholeFlat`, `MlirWholeFunction`, `MlirWholeGuard`, `MlirWholeInstance`, `MlirWholeKernel`, `MlirWholeLanes`, `MlirWholeLoop`, `MlirWholeNarrow`, `MlirWholeNative`, `MlirWholeRows`, `MlirWholeSpan`, `MlirWholeStatements`, `MlirWholeText`, `MlirWholeTypes`, `MlirWholeValues` (23).
  - *Allowed deps*: `.`, `Compiler/Mlir`, `Compiler/MlirLoops`, `Compiler/MlirDevice`

### `NeuralNetworks` (127 -> 5 packages)
- **`NeuralNetworks/Core` (25)**: `Activation`, `ActivationComparison`, `Attention`, `AttentionGradients`, `Basis`, `Batch`, `BFloat16`, `Convolution`, `Convolved`, `Convrot`, `Float32`, `Float8`, `Gpu`, `GpuMemory`, `GroupNorm`, `Grouping`, `HeadRotated`, `HeadRows`, `Layer`, `MaskedAttention`, `Moments`, `Normalization`, `Products`, `Softmax`, `Sylvester`.
  - *Allowed deps*: `.`, `Math`
- **`NeuralNetworks/Kernels` (24)**: `DeltaBatch`, `DeltaBlock`, `DeltaChunks`, `Experts`, `Hybrid`, `HyperStreams`, `LayerDropping`, `MappedExperts`, `MixturePruning`, `NeuronPruning`, `NgramEmbedding`, `Ngrams`, `Recurrent`, `ReturnedMemory`, `RopeRotated`, `Routing`, `RowModulated`, `RowsAhead`, `SequenceLength`, `ShortConvolution`, `Shortened`, `SituRotated`, `TokenModulated`, `TokenSplitter`.
  - *Allowed deps*: `.`, `Math`, `NeuralNetworks/Core`
- **`NeuralNetworks/Models` (27)**: `FlashDrafter`, `FlashLayer`, `FlashNext`, `FlashNextText`, `FlashSpeculation`, `Gemma4`, `Gemma4Benchmark`, `Gemma4Block`, `GemmaTokens`, `Generation`, `Gguf`, `GgufFile`, `Gpt2`, `Gpt2Benchmark`, `Gpt2Block`, `Gpt2Pass`, `Gpt2Text`, `Gpt2Tokens`, `Greedy`, `Lfm2Text`, `Lfm3`, `Lfm3Block`, `Lfm3Text`, `LookupDecoding`, `MachineUse`, `NemotronText`, `Perplexity`, `Qwen3`, `Qwen3Benchmark`, `Qwen3Block`, `Qwen3CoderText`, `Qwen3Images`, `Qwen3Text`, `QwenSafetensors`, `QwenVision`, `QwenVisionBlock`, `QwenVisionMerger`, `Safetensors`, `Sampling`, `Speculation`, `TensorFile` (41 -> split into **`NeuralNetworks/Formats`** (14: `Gguf*`, `Safetensors`, `QwenSafetensors`, `TensorFile`, `Generation`, `Greedy`, `Sampling`, `MachineUse`, `Speculation`, `LookupDecoding`, `*Tokens`) and **`NeuralNetworks/Models`** (27)).
  - *Allowed deps*: `.`, `Math`, `Json`, `ImageProcessing`, `Networking`, `Bytecode/Binary`, `NeuralNetworks/Core`, `NeuralNetworks/Kernels`
- **`NeuralNetworks/Training` (24)**: `Activity`, `Arm`, `ArmEvolution`, `BinaryAddition`, `Csv`, `Curve`, `Dataset`, `Depth`, `Descent`, `Diabetes`, `Drafter`, `Draw`, `Evaluation`, `FeatureMap`, `Finetuning`, `Gpt2Training`, `Gru`, `Iris`, `LinearRegression`, `LogicGates`, `Network`, `Neuroevolution`, `Optimizer`, `OptimizerComparison`, `Perceptron`, `Reber`, `ReberPrediction`, `Smallest`, `SpectralFit`, `SpectralHistory`, `SpectralSolve`, `Weasel`, `Xor` (33 -> split `Spectral*` + `Weasel` + `Xor` + `Arm*` + `LogicGates` + `Reber*` into **`NeuralNetworks/Evolution`** (13) and **`NeuralNetworks/Training`** (20)).
  - *Allowed deps*: `.`, `Math`, `NeuralNetworks/Core`
- **`NeuralNetworks/Probes` (4)**: `CheckNativeThreads`, `MtpProbe`, `QuickProbe`, `Reach` (or fold into `NeuralNetworks/Models` / `Training`).

### `Examples` (88 -> 4 packages)
- **`Examples/Basics` (22)**: `Announcer`, `AreaCalculator`, `ArithmeticFunction`, `AssertionPrecision`, `Beeramid`, `ComputedRectangle`, `ConvertingNumbers`, `Countdown`, `Couple`, `EvenFibonacci`, `Fibonacci`, `FizzBuzz`, `FourthSum`, `FruitPrices`, `GcdCalculator`, `Grade`, `Greeter`, `HelloLogger`, `Prices`, `SimpleCalculator`, `Squares`, `Sum`, `TemperatureConverter` (23).
- **`Examples/Collections` (22)**: `AutofilledMutable`, `Boxed`, `BoxUser`, `DoublingList`, `FindFirstNonConsecutiveNumber`, `FindNumberCount`, `GenericParameter`, `Growing`, `Halvings`, `IndexedCall`, `Labeled`, `LargeCollections`, `LinkedListAnalyzer`, `ListFiltering`, `MatchingLoopFinder`, `NumberStats`, `NumberSummer`, `NumberTree`, `ReduceButGrow`, `RemoveDuplicateWords`, `RemoveExclamation`, `RemoveParentheses`, `ReverseList`, `TextLists` (24).
- **`Examples/Runtime` (21)**: `DeepRecursion`, `DirProbe`, `ErrorHandling`, `ErrorStatements`, `ExecuteOperation`, `FrameLimit`, `Instruction`, `LineEndings`, `LockedCycles`, `LoopAllocations`, `LoopConditionMember`, `LoopSpeed`, `MemoryPressure`, `Node`, `Opcode`, `ParallelGarbage`, `ParallelKept`, `ParallelSum`, `ProcessProbe`, `RegisterBank`, `Resolution`, `Swarm`, `TracedValues`, `YoungCollections` (24).
- **`Examples/Native` (17)**: `Leibniz`, `NativeArithmetic`, `NativeConditions`, `NativeLoop`, `NumbersPlusFloats`, `Pixel`, `Processor`, `PureAdder`, `RegexSpeed`, `Rendering`, `Shade`, `Shelf`, `ShorterPath`, `Spans`, `Turning`, `Unicode`, `Walker`.
  - *Allowed deps for all `Examples/*`*: `.`, `Math`, `Bytecode/*`, `Runtime/*` (examples/benchmarks only; nothing depends on `Examples/*`).

---

## 3. Path & Fixture Risks Before Moving Any File

Moving files into subfolders changes their Strict package name (`Strict/Folder/Subfolder/Type`) and breaks hardcoded relative paths:
1. **`Strict.strict` & `Compiler/SelfBuild.strict`**: `CorePackages` and `ExtraPackages` list top-level packages; `SelfBuild.IsCoreTarget` and `Intermediates.SourceFolders` must include the new sub-packages (or scan recursively).
2. **`Compiler/NativeRuntime.mlir` & `Compiler/Bootstrap.cmd`**: `Bootstrap.mlir`, `Seed.strict`, and `NativeBuild` reference `Compiler/NativeRuntime*.mlir` and `Strict.strict`.
3. **Bytecode & Compiler unit-test fixtures**:
   - `Bytecode/BinaryFile.strict`, `BinaryLinks.strict`, `BinarySources.strict`, `BuildOutput.strict`, `CallTarget.strict`, `SourceFiles.strict`: hardcode `./Examples/Sum.strict`, `./Math/Vector2.strict`, `NeuralNetworks/images`, `Compiler/NativeRuntime.mlir`, etc.
   - `Validators/ValidateCheck.strict`, `Validators/ProjectRules.strict`, `Validators/FolderFileLimit.strict`: reference `Examples/ProjectRule`, `Examples/LinkChain`, `Validators/...`.
4. **Sibling repos (`DeltaEngine`, `Ben`, `Solver`, `StrictComfy`)**: any `NeuralNetworks/Qwen3Benchmark` or `Compiler/...` path references in scripts/tests must be updated when those files move.
