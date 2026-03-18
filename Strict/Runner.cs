using Strict.Bytecode;
using Strict.Compiler;
using Strict.Compiler.Assembly;
using Strict.Expressions;
using Strict.Language;
using Strict.Optimizers;
using Strict.TestRunner;
using Strict.Validators;
using Type = Strict.Language.Type;

namespace Strict;

/// <summary>
/// Allows running or build a .strict source file, running the Run method or supplying an
/// expression to be executed based on what is available in the given file. Caches .strictbinary
/// bytecode instructions that are used in later runs, only updated when source files are newer.
/// Note that only .strict files contain the full actual code, everything after that is
/// stripped, optimized, and just includes what is actually executed.
/// </summary>
public sealed class Runner
{
	public Runner(string strictFilePath, Package? skipPackageSearchAndUseThisTestPackage = null,
		string expressionToRun = Method.Run, bool enableTestsAndDetailedOutput = false)
	{
		this.strictFilePath = strictFilePath;
		this.skipPackageSearchAndUseThisTestPackage = skipPackageSearchAndUseThisTestPackage;
		this.expressionToRun = expressionToRun;
		this.enableTestsAndDetailedOutput = enableTestsAndDetailedOutput;
		parser = new MethodExpressionParser();
		repositories = new Repositories(parser);
		Log("Strict.Runner: " + strictFilePath);
	}

	private readonly string strictFilePath;
	private readonly Package? skipPackageSearchAndUseThisTestPackage;
	private readonly string expressionToRun;
	private readonly bool enableTestsAndDetailedOutput;
	private readonly MethodExpressionParser parser;
	private readonly Repositories repositories;
	private readonly List<long> stepTimes = new();

	private void Log(string message)
	{
		if (enableTestsAndDetailedOutput)
			Console.WriteLine(message);
	}

	/// <summary>
	/// Generates a platform-specific executable from the compiled instructions. Uses MLIR when
	/// -mlir is specified, LLVM IR when -llvm is specified, otherwise NASM + gcc/clang pipeline.
	/// LLVM and MLIR are opt-in until feature parity is reached.
	/// Throws <see cref="ToolNotFoundException"/> if required tools are missing.
	/// </summary>
	public async Task Build(Platform platform, CompilerBackend backend = CompilerBackend.MlirDefault)
	{
		var binary = await GetBinary();
		InstructionsCompiler compiler = backend switch
		{
			CompilerBackend.Llvm => new InstructionsToLlvmIr(),
			CompilerBackend.Nasm => new InstructionsToAssembly(),
			_ => new InstructionsToMlir()
		};
		Linker linker = backend switch
		{
			CompilerBackend.Llvm => new LlvmLinker(),
			CompilerBackend.Nasm => new NativeExecutableLinker(),
			_ => new MlirLinker()
		};
		var irFilePath = Path.ChangeExtension(strictFilePath, compiler.Extension);
		await File.WriteAllTextAsync(irFilePath, await compiler.Compile(binary, platform));
		var exeFilePath = await linker.CreateExecutable(irFilePath, platform, binary.UsesConsolePrint);
		PrintCompilationSummary(backend, platform, exeFilePath);
	}

	/// <summary>
	/// Tries to load a .strictbinary directly if it exists and is up to date, otherwise will load
	/// from source and generate a fresh .strictbinary to be used in later runs as well.
	/// </summary>
	private async Task<BinaryExecutable> GetBinary()
	{
		var basePackage = skipPackageSearchAndUseThisTestPackage ?? await GetPackage(nameof(Strict));
		if (Path.GetExtension(strictFilePath) == BinaryExecutable.Extension)
			return LogTiming("Loading existing " + strictFilePath,
				() => new BinaryExecutable(strictFilePath, basePackage));
		var cachedBinaryPath = Path.ChangeExtension(strictFilePath, BinaryExecutable.Extension);
		if (File.Exists(cachedBinaryPath))
		{
			var binary = new BinaryExecutable(cachedBinaryPath, basePackage);
			var binaryLastModified = new FileInfo(cachedBinaryPath).LastWriteTimeUtc;
			var sourceLastModified = new FileInfo(strictFilePath).LastWriteTimeUtc;
			foreach (var typeFullName in binary.MethodsPerType.Keys)
			{
				var fileLastModified =
					new FileInfo(typeFullName + Type.Extension).LastWriteTimeUtc;
				if (fileLastModified > sourceLastModified)
					sourceLastModified = fileLastModified;
			}
			if (binaryLastModified >= sourceLastModified)
			{
				Log("Cached " + cachedBinaryPath + " from " + binaryLastModified +
					" is still good, using it. Latest source file change: " + sourceLastModified);
				return binary;
			}
			Log("Cached " + cachedBinaryPath + " is outdated from " + binaryLastModified +
				", source modified at " + sourceLastModified);
		}
		return await LoadFromSourceAndSaveBinary(basePackage);
	}

	private async Task<Package> GetPackage(string name)
	{
		if (skipPackageSearchAndUseThisTestPackage != null)
			return skipPackageSearchAndUseThisTestPackage;
		return await LogTiming(nameof(GetPackage) + " " + name,
			async () => name.StartsWith(nameof(Strict), StringComparison.Ordinal)
				? await repositories.LoadStrictPackage(name)
				: throw new NotSupportedException("No github package search ability was implemented " +
					"yet, only Strict packages work for now: " + name));
	}

	private T LogTiming<T>(string message, Func<T> callToTime)
	{
		var startTicks = DateTime.UtcNow.Ticks;
		try
		{
			return callToTime();
		}
		finally
		{
			var endTicks = DateTime.UtcNow.Ticks;
			Log(message + " Time: " + TimeSpan.FromTicks(endTicks - startTicks).TotalMilliseconds +
				" ms");
			stepTimes.Add(endTicks - startTicks);
		}
	}

	private async Task<BinaryExecutable> LoadFromSourceAndSaveBinary(Package basePackage)
	{
		var typeName = Path.GetFileNameWithoutExtension(strictFilePath);
		var typeLines = new TypeLines(typeName, await File.ReadAllLinesAsync(strictFilePath));
		using var mainType = new Type(basePackage, typeLines).ParseMembersAndMethods(parser);
		if (enableTestsAndDetailedOutput)
		{
			Parse(mainType);
			Validate(mainType);
			RunTests(basePackage, mainType);
		}
		var expression = parser.ParseExpression(
			new Body(new Method(mainType, 0, parser, [nameof(LoadFromSourceAndSaveBinary)])),
			expressionToRun);
		var executable = GenerateBinaryExecutable(expression);
		Log("Generated bytecode instructions: " + executable.TotalInstructionsCount);
		OptimizeBytecode(executable);
		return CacheStrictExecutable(executable);
	}

	private void Parse(Type mainType) =>
		Log(LogTiming(nameof(Parse) + " " + strictFilePath, () =>
		{
			var parsedMethods = 0;
			var totalExpressions = 0;
			foreach (var method in mainType.Methods)
				if (!method.IsTrait)
				{
					var body = method.GetBodyAndParseIfNeeded();
					parsedMethods++;
					if (body is Body bodyExpr)
						totalExpressions += bodyExpr.Expressions.Count;
					else
						totalExpressions++;
				}
			return "Parsed methods: " + parsedMethods + ", total expressions: " + totalExpressions;
		}));

	private void Validate(Type mainType) =>
		Log(LogTiming(nameof(Validate) + " " + strictFilePath, () =>
		{
			new TypeValidator().Visit(mainType);
			var constants = new ConstantCollapser();
			constants.Visit(mainType);
			return "All type validations passed. Constant expressions collapsed: " +
				constants.CollapsedCount;
		}));

	private void RunTests(Package basePackage, Type mainType) =>
		Log(LogTiming(nameof(Parse) + " " + strictFilePath, () =>
		{
			var testExecutor = new TestInterpreter(basePackage);
			testExecutor.RunAllTestsInType(mainType);
			return "Methods tested: " + testExecutor.Statistics.MethodsTested + ", Types tested: " +
				testExecutor.Statistics.TypesTested + "\n" + testExecutor.Statistics;
		}));

	private BinaryExecutable GenerateBinaryExecutable(Expression entryPoint) =>
		LogTiming(nameof(GenerateBinaryExecutable),
			() => new BinaryGenerator(entryPoint).Generate());

	private void OptimizeBytecode(BinaryExecutable executable) =>
		Log(LogTiming(nameof(OptimizeBytecode), () =>
		{
			var optimizers = new InstructionOptimizer[]
			{
				new TestCodeRemover(), new ConstantFoldingOptimizer(), new DeadStoreEliminator(),
				new UnreachableCodeEliminator(), new RedundantLoadEliminator()
			};
			var beforeInstructionsCount = executable.TotalInstructionsCount;
			var optimizersResults = new List<string>();
			foreach (var optimizer in optimizers)
			{
				var beforeOptimizerInstructions = executable.TotalInstructionsCount;
				optimizer.Optimize(executable);
				var removed = executable.TotalInstructionsCount - beforeOptimizerInstructions;
				optimizersResults.Add(optimizer.GetType().Name + ": removed " + removed + " instructions");
			}
			var afterInstructionsCount = executable.TotalInstructionsCount;
			var removedInstructions = beforeInstructionsCount - afterInstructionsCount;
			return "Removed instruction: " + removedInstructions + " (" +
				removedInstructions * 100 / beforeInstructionsCount + "%)" +
				" with " + optimizers.Length + " optimizers:\n\t" + string.Join("\n\t", optimizersResults);
		}));

	private BinaryExecutable CacheStrictExecutable(BinaryExecutable binary)
	{
		var outputFilePath = Path.ChangeExtension(strictFilePath, BinaryExecutable.Extension);
		binary.Serialize(outputFilePath);
		Log("Saving " + new FileInfo(outputFilePath).Length + " bytes of bytecode to: " +
			outputFilePath);
		return binary;
	}

	private void PrintCompilationSummary(CompilerBackend backend, Platform platform, string exeFilePath) =>
		Console.WriteLine("Compiled " + strictFilePath + " via " + backend + " in " +
			TimeSpan.FromTicks(stepTimes.Sum()).ToString(@"s\.ffffff") + "s to " + platform +
			" executable of " + new FileInfo(exeFilePath).Length + " bytes to: " + exeFilePath);

/*obs, integrate below! TODO: cleanup
	var startTicks = DateTime.UtcNow.Ticks;
	currentFolder = Path.GetDirectoryName(Path.GetFullPath(strictFilePath))!;
		var typeName = Path.GetFileNameWithoutExtension(strictFilePath);
		if (skipPackageSearchAndUseThisTestPackage != null)
		{
			var basePackage = skipPackageSearchAndUseThisTestPackage;
			if (Directory.Exists(strictFilePath))
				(package, mainType) = LoadPackageFromDirectory(basePackage, strictFilePath);
			else if (Path.GetExtension(strictFilePath) == Binary.Extension)
			{
				binary = new Binary(strictFilePath, basePackage);
	//package = new Package(basePackage, typeName);
	mainType = new Type(basePackage, new TypeLines(typeName, Method.Run))
					.ParseMembersAndMethods(new MethodExpressionParser());
			}
			else
			{
				var packageName = Path.GetFileNameWithoutExtension(strictFilePath);
//package = new Package(basePackage, packageName);
var typeLines = new TypeLines(typeName, File.ReadAllLines(strictFilePath));
mainType = new Type(basePackage, typeLines)
					.ParseMembersAndMethods(new MethodExpressionParser());
			}
		}
		else
{
	package = null!;
	mainType = null!;
}
var endTicks = DateTime.UtcNow.Ticks;
stepTimes.Add(endTicks - startTicks);
Log("└─ Step 1 ⏱ Time: " +
	TimeSpan.FromTicks(endTicks - startTicks).TotalMilliseconds + " ms");
private Binary? binary;
	private readonly string currentFolder;
	private readonly Package package;
	private readonly Type mainType;
	*
	private async Task SaveLlvmExecutable(IReadOnlyList<Instruction> optimizedInstructions, Platform platform,
		IReadOnlyDictionary<string, List<Instruction>>? precompiledMethods)
	{
		var llvmCompiler = new InstructionsToLlvmIr();
		var llvmIr = llvmCompiler.CompileForPlatform(mainType.Name, optimizedInstructions, platform,
			precompiledMethods);
		var llvmPath = Path.Combine(currentFolder, mainType.Name + ".ll");
		await File.WriteAllTextAsync(llvmPath, llvmIr);
		Console.WriteLine("Saved " + platform + " LLVM IR to: " + llvmPath);
		var exeFilePath = new LlvmLinker().CreateExecutable(llvmPath, platform,
			llvmCompiler.IsPlatformUsingStdLibAndHasPrintInstructions(platform, optimizedInstructions,
				precompiledMethods));
		PrintCompilationSummary("LLVM", platform, exeFilePath);
	}

	private async Task SaveMlirExecutable(IReadOnlyList<Instruction> optimizedInstructions, Platform platform,
		IReadOnlyDictionary<string, List<Instruction>>? precompiledMethods)
	{
		var mlirCompiler = new InstructionsToMlir();
		var mlirText = mlirCompiler.CompileForPlatform(mainType.Name, optimizedInstructions, platform,
			precompiledMethods);
		var mlirPath = Path.Combine(currentFolder, mainType.Name + ".mlir");
		await File.WriteAllTextAsync(mlirPath, mlirText);
		Console.WriteLine("Saved " + platform + " MLIR to: " + mlirPath);
		var exeFilePath = new MlirLinker().CreateExecutable(mlirPath, platform,
			mlirCompiler.IsPlatformUsingStdLibAndHasPrintInstructions(platform, optimizedInstructions,
				precompiledMethods));
		PrintCompilationSummary("MLIR", platform, exeFilePath);
	}

	private async Task SaveNasmExecutable(IReadOnlyList<Instruction> optimizedInstructions, Platform platform,
		IReadOnlyDictionary<string, List<Instruction>>? precompiledMethods)
	{
		var compiler = new InstructionsToAssembly();
		var assemblyText = compiler.CompileForPlatform(mainType.Name, optimizedInstructions, platform,
			precompiledMethods);
		var asmPath = Path.Combine(currentFolder, mainType.Name + ".asm");
		await File.WriteAllTextAsync(asmPath, assemblyText);
		Console.WriteLine("Saved " + platform + " NASM assembly to: " + asmPath);
		var hasPrint = compiler.HasPrintInstructions(optimizedInstructions);
		var exeFilePath = new NativeExecutableLinker().CreateExecutable(asmPath, platform, hasPrint);
		PrintCompilationSummary("NASM", platform, exeFilePath);
	}

	private void ExecuteBytecode(IReadOnlyList<Instruction> instructions,
		Dictionary<string, List<Instruction>>? precompiledMethods = null,
		IReadOnlyDictionary<string, ValueInstance>? initialVariables = null)
	{
		Log("┌─ Step 8: Execute");
		Log("│  ✓ Executing " + mainType.Name + ".Run method:");
		var startTicks = DateTime.UtcNow.Ticks;
		new VirtualMachine(package, precompiledMethods).Execute(instructions, initialVariables);
		var endTicks = DateTime.UtcNow.Ticks;
		Log("│  ✓ Run method executed successfully, instructions: " + instructions.Count);
		stepTimes.Add(endTicks - startTicks);
		Log("└─ Step 8 ⏱ Time: " + TimeSpan.FromTicks(endTicks - startTicks).TotalMilliseconds +
			" ms");
	}

	private IReadOnlyDictionary<string, ValueInstance>? BuildProgramArguments(string[] programArgs)
	{
		if (programArgs is not { Length: > 0 })
			return null;
		var runMethod = mainType.Methods.FirstOrDefault(method =>
				method.Name == Method.Run && method.Parameters.Count == programArgs.Length) ??
			mainType.Methods.FirstOrDefault(method => method is
				{ Name: Method.Run, Parameters: [{ Type.IsList: true }] });
		if (runMethod == null)
			throw new NotSupportedException( //ncrunch: no coverage
				"No Run method with " + programArgs.Length + " arguments " +
				"found: " + ParsingFailed.GetClickableStacktraceLine(mainType, 0, Method.Run));
		var numberType = package.GetType(Type.Number);
		var numbersType = package.GetListImplementationType(numberType);
		var numbers = programArgs.Select(argument =>
			new ValueInstance(numberType,
				double.Parse(argument, CultureInfo.InvariantCulture))).ToArray();
		var numbersValue = new ValueInstance(numbersType, numbers);
		return new Dictionary<string, ValueInstance> { [runMethod.Parameters[0].Name] = numbersValue };
	}

//TODO: wrong!
	/// <summary>
	/// Evaluates a Strict expression like "TypeName(args).Method" or "TypeName(args)" (calls Run).
	/// The result is printed to Console if the method returns a value.
	/// Example: runner.RunExpression("FibonacciRunner(5).Compute") prints "5".
	/// </summary>
	public async Task RunExpression(string expressionString)
	{
		var typeName = Path.GetFileNameWithoutExtension(strictFilePath);
		var basePackage = skipPackageSearchAndUseThisTestPackage ?? await GetPackage(nameof(Strict));
		var sourceLines = await File.ReadAllLinesAsync(strictFilePath);
		var targetType = new Type(basePackage, new TypeLines(typeName, sourceLines)).ParseMembersAndMethods(parser);
		try
		{
			var expression = parser.ParseExpression(
				new Body(new Method(targetType, 0, parser, new[] { nameof(RunExpression) })),
				expressionString);
			var binary = GenerateBinaryExecutable(expression);
			OptimizeBytecode(binary);
			var vm = new VirtualMachine(binary);
			vm.Execute();
			if (vm.Returns.HasValue)
				Console.WriteLine(vm.Returns.Value.ToExpressionCodeString());
		}
		finally
		{
			targetType.Dispose();
		}
	}
	 //TODO: wrong, this is already above in expression string[]? programArgs = null)
*/
	public async Task Run()
	{
		var binary = await GetBinary();
		LogTiming(nameof(Run), () => new VirtualMachine(binary).Execute());
		Console.WriteLine("Executed " + strictFilePath + " via " + nameof(VirtualMachine) + " in " +
			TimeSpan.FromTicks(stepTimes.Sum()).ToString(@"s\.ffffff") + "s");
	}
}