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

public sealed class Runner
{
	/// <summary>
	/// Allows running or build a .strict source file, running the Run method or supplying an
	/// expression to be executed based on what is available in the given file. Caches .strictbinary
	/// bytecode instructions that are used in later runs, only updated when source files are newer.
	/// Note that only .strict files contain the full actual code, everything after that is
	/// stripped, optimized, and just includes what is actually executed.
	/// </summary>
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
				var fileLastModified = new FileInfo(typeFullName+Type.Extension).LastWriteTimeUtc;
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
						totalExpressions++; //ncrunch: no coverage
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
		LogTiming(nameof(GenerateBinaryExecutable), () => new BinaryGenerator(entryPoint).Generate());

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

	/*obs, integrate below!
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

*/
	public async Task Run()
	{
		var binary = await GetBinary();
		new VirtualMachine(binary).Execute();
	}
		/*
		binary != null
			? RunFromPreloadedBytecode(programArgs)
			: RunFromSource(programArgs);

	private async Task RunFromPreloadedBytecode(string[] programArgs)
	{
		Log("╔═══════════════════════════════════════════╗");
		Log("║  Running from pre-compiled .strictbinary  ║");
		Log("╚═══════════════════════════════════════════╝");
		var runInstructions = binary!.FindInstructions(mainType.FullName, Method.Run, 0) ??
			throw new InvalidOperationException("No Run method found in " + mainType.Name);
		var precompiledMethods = BuildPrecompiledMethodsFromBytecodeTypes();
		ExecuteBytecode(runInstructions, precompiledMethods, BuildProgramArguments(programArgs));
		Log("Successfully executed pre-compiled " + mainType.Name + " in " +
			TimeSpan.FromTicks(stepTimes.Sum()).ToString(@"s\.ffffff") + "s");
	}

	private Dictionary<string, List<Instruction>>? BuildPrecompiledMethodsFromBytecodeTypes()
	{
		if (binary == null)
			return null;
		var methods = new Dictionary<string, List<Instruction>>(StringComparer.Ordinal);
		foreach (var typeData in binary.MethodsPerType.Values)
			foreach (var (methodKey, methods) in typeData.InstructionsPerMethodGroup)
				foreach (var (method, instructions) in methods)
					methods[methodKey] = instructions;
		return methods.Count > 0
			? methods
			: null;
	}

	private async Task RunFromSource(string[] programArgs)
	{
		var optimizedInstructions = BuildFromSource(true);
		ExecuteBytecode(optimizedInstructions, null, BuildProgramArguments(programArgs));
		Log("Successfully parsed, optimized and executed " + mainType.Name + " in " +
			TimeSpan.FromTicks(stepTimes.Sum()).ToString(@"s\.ffffff") + "s");
		return this;
	}
*

	private IReadOnlyList<TypeBytecodeData> BuildTypeBytecodeData(
		List<Instruction> optimizedRunInstructions)
	{
		var methodsByType = new Dictionary<Type, Dictionary<Method, IList<Instruction>>>();
		var methodsToCompile = new Queue<Method>();
		var compiledMethodKeys = new HashSet<string>(StringComparer.Ordinal);
		EnqueueCalledMethods(optimizedRunInstructions, methodsToCompile, compiledMethodKeys);
		while (methodsToCompile.Count > 0)
		{
			var method = methodsToCompile.Dequeue();
			if (method.IsTrait || !IsTypeInsideCurrentPackage(method.Type))
				continue;
			var methodExpressions = GetMethodExpressions(method);
			var methodInstructions = new BinaryGenerator(
				new InvokedMethod(methodExpressions, EmptyArguments, method.ReturnType),
				new Registry()).Generate();
			if (!methodsByType.TryGetValue(method.Type, out var typeMethods))
			{
				typeMethods = new Dictionary<Method, IList<Instruction>>();
				methodsByType[method.Type] = typeMethods;
			}
			typeMethods[method] = methodInstructions;
			EnqueueCalledMethods(methodInstructions, methodsToCompile, compiledMethodKeys);
		}
		var requiredTypes = CollectRequiredTypes(methodsByType, optimizedRunInstructions);
		var usedMethodKeys = compiledMethodKeys;
		var calledTypeNames = methodsByType.Where(typeEntry => typeEntry.Value.Count > 0).
			Select(typeEntry => typeEntry.Key.Name).ToHashSet(StringComparer.Ordinal);
		var memberTypeNames = new HashSet<string>(mainType.Members.Select(member =>
			member.Type.Name), StringComparer.Ordinal);
		foreach (var currentPackageType in methodsByType.Keys.Where(IsTypeInsideCurrentPackage))
		foreach (var member in currentPackageType.Members)
			memberTypeNames.Add(member.Type.Name);
		var typeData = new List<TypeBytecodeData>();
		foreach (var requiredType in requiredTypes)
		{
			if (!IsTypeInsideCurrentPackage(requiredType) &&
				!calledTypeNames.Contains(requiredType.Name) &&
				!memberTypeNames.Contains(requiredType.Name))
				continue;
			methodsByType.TryGetValue(requiredType, out var compiledMethods);
			var methodDefinitions = requiredType.Methods.Where(method =>
				usedMethodKeys.Contains(BuildMethodKey(method)) ||
				requiredType == mainType && method.Name == Method.Run).Select(method =>
				new MethodBytecodeData(method.Name,
					method.Parameters.Select(parameter =>
						new MethodParameterBytecodeData(parameter.Name, parameter.Type.Name)).ToList(),
					method.ReturnType.Name)).ToList();
			var members = requiredType.Members.Where(member => !member.IsConstant).Select(member =>
				new MemberBytecodeData(member.Name, member.Type.Name)).ToList();
			var methodInstructions = new Dictionary<MethodBytecodeData, IList<Instruction>>();
			if (compiledMethods != null)
				foreach (var compiledMethod in compiledMethods)
					methodInstructions[new MethodBytecodeData(compiledMethod.Key.Name,
						compiledMethod.Key.Parameters.Select(parameter =>
							new MethodParameterBytecodeData(parameter.Name, parameter.Type.Name)).ToList(), //ncrunch: no coverage
						compiledMethod.Key.ReturnType.Name)] = compiledMethod.Value;
			typeData.Add(new TypeBytecodeData(requiredType.Name, BuildTypeEntryPath(requiredType),
				members, methodDefinitions,
				requiredType == mainType
					? optimizedRunInstructions
					: EmptyInstructions,
				methodInstructions));
		}
		return typeData;
	}

	private HashSet<Type> CollectRequiredTypes(
		Dictionary<Type, Dictionary<Method, IList<Instruction>>> methodsByType,
		IReadOnlyList<Instruction> optimizedRunInstructions)
	{
		var requiredTypes = new HashSet<Type> { mainType };
		foreach (var member in mainType.Members)
			requiredTypes.Add(member.Type);
		foreach (var instruction in optimizedRunInstructions)
			CollectInstructionTypes(instruction, requiredTypes);
		foreach (var methodEntry in methodsByType)
		{
			requiredTypes.Add(methodEntry.Key);
			if (IsTypeInsideCurrentPackage(methodEntry.Key))
				foreach (var member in methodEntry.Key.Members)
					requiredTypes.Add(member.Type);
			foreach (var compiledMethod in methodEntry.Value)
			{
				AddMethodSignatureTypes(compiledMethod.Key, requiredTypes);
				foreach (var instruction in compiledMethod.Value)
					CollectInstructionTypes(instruction, requiredTypes);
			}
		}
		return requiredTypes;
	}

	private static void AddMethodSignatureTypes(Method method, ISet<Type> requiredTypes)
	{
		requiredTypes.Add(method.Type);
		requiredTypes.Add(method.ReturnType);
		foreach (var parameter in method.Parameters)
			requiredTypes.Add(parameter.Type);
	}

	private static void CollectInstructionTypes(Instruction instruction, ISet<Type> requiredTypes)
	{
		if (instruction is not Invoke { Method: not null } invoke)
			return;
		AddMethodSignatureTypes(invoke.Method.Method, requiredTypes);
		requiredTypes.Add(invoke.Method.ReturnType);
	}

	private static readonly IList<Instruction> EmptyInstructions = Array.Empty<Instruction>();
	private static readonly IReadOnlyDictionary<Method, IList<Instruction>> EmptyMethodInstructions =
		new Dictionary<Method, IList<Instruction>>();

	private string BuildTypeEntryPath(Type type) =>
		IsTypeInsideCurrentPackage(type)
			? package.Name + "/" + type.Name
			: "Strict/" + type.Name;

	private bool IsTypeInsideCurrentPackage(Type type) =>
		type.Package == package ||
		type.Package.FullName.StartsWith(package.FullName + Context.ParentSeparator,
			StringComparison.Ordinal);

	private static IReadOnlyList<Expression> GetMethodExpressions(Method method)
	{
		var methodBody = method.GetBodyAndParseIfNeeded();
		return methodBody is Body body
			? body.Expressions
			: [methodBody];
	}

	// ReSharper disable once CollectionNeverUpdated.Local
	private static readonly Dictionary<string, ValueInstance> EmptyArguments =
		new(StringComparer.Ordinal);

	private static void EnqueueCalledMethods(IReadOnlyList<Instruction> instructions,
		Queue<Method> methodsToCompile, HashSet<string> compiledMethodKeys)
	{
		foreach (var invokeInstruction in instructions.OfType<Invoke>())
			if (invokeInstruction.Method != null)
			{
				EnqueueCalledMethod(invokeInstruction.Method.Method, methodsToCompile,
					compiledMethodKeys);
				if (invokeInstruction.Method.Instance != null)
					EnqueueMethodsFromExpression(invokeInstruction.Method.Instance, methodsToCompile,
						compiledMethodKeys);
				foreach (var argument in invokeInstruction.Method.Arguments)
					EnqueueMethodsFromExpression(argument, methodsToCompile, compiledMethodKeys);
			}
	}

	private static void EnqueueMethodsFromExpression(Expression expression,
		Queue<Method> methodsToCompile, HashSet<string> compiledMethodKeys)
	{
		switch (expression)
		{
		case MethodCall methodCall:
			//ncrunch: no coverage start
			EnqueueCalledMethod(methodCall.Method, methodsToCompile, compiledMethodKeys);
			if (methodCall.Instance != null)
				EnqueueMethodsFromExpression(methodCall.Instance, methodsToCompile,
					compiledMethodKeys);
			foreach (var argument in methodCall.Arguments)
				EnqueueMethodsFromExpression(argument, methodsToCompile, compiledMethodKeys);
			break; //ncrunch: no coverage end
		case MemberCall { Instance: not null } memberCall:
			// ReSharper disable once TailRecursiveCall
			//ncrunch: no coverage start
			EnqueueMethodsFromExpression(memberCall.Instance, methodsToCompile, compiledMethodKeys);
			break;
		case List listExpression:
			foreach (var value in listExpression.Values)
				EnqueueMethodsFromExpression(value, methodsToCompile, compiledMethodKeys);
			break; //ncrunch: no coverage end
		}
	}

	private static void EnqueueCalledMethod(Method method, Queue<Method> methodsToCompile,
		HashSet<string> compiledMethodKeys)
	{
		if (method.Name == Method.From)
			return;
		var methodKey = BuildMethodKey(method);
		if (compiledMethodKeys.Add(methodKey))
			methodsToCompile.Enqueue(method);
	}

	private static string BuildMethodKey(Method method) =>
		BinaryExecutable.BuildMethodHeader(method.Name,
			method.Parameters.Select(parameter =>
				new BinaryMember(parameter.Name, parameter.Type.Name, null)).ToList(),
			method.ReturnType);

	/// <summary>
	/// Loads a package, which is all .strict files in a folder, then finds the Run entry point and
	/// uses that as our mainType. Otherwise the same as
	/// </summary>
	private static (Package Package, Type MainType) LoadPackageFromDirectory(Package basePackage,
		string dirPath)
	{
		var packageName = Path.GetFileName(dirPath);
		var childPackage = new Package(basePackage, packageName);
		var parser = new MethodExpressionParser();
		var files = Directory.GetFiles(dirPath, "*" + Type.Extension, SearchOption.TopDirectoryOnly);
		// ReSharper disable once RedundantSuppressNullableWarningExpression
		var typeLinesByName = files.ToDictionary(s => Path.GetFileNameWithoutExtension(s)!,
			filePath => new TypeLines(Path.GetFileNameWithoutExtension(filePath),
				File.ReadAllLines(filePath)), StringComparer.Ordinal);
		foreach (var sortedTypeLines in SortTypesByDependency(typeLinesByName))
			new Type(childPackage, sortedTypeLines).ParseMembersAndMethods(parser);
		if (!childPackage.Types.TryGetValue(packageName, out var mainType))
			// Fallback: use the first type with a Run method if no type matches the directory name
			mainType = childPackage.Types.Values.FirstOrDefault(type => //ncrunch: no coverage
					type.Methods.Any(method => method.Name == Method.Run)) ?? //ncrunch: no coverage
				throw new InvalidOperationException("Package directory '" + dirPath + "' does not contain " +
					"a type named '" + packageName + "' or any type with a Run method.");
		return (childPackage, mainType);
	}

	private static IEnumerable<TypeLines> SortTypesByDependency(
		Dictionary<string, TypeLines> typeLinesByName)
	{
		var withInternalDeps = new Dictionary<string, TypeLines>(StringComparer.Ordinal);
		foreach (var typeLines in typeLinesByName.Values)
			if (typeLines.DependentTypes.Any(typeLinesByName.ContainsKey))
				withInternalDeps[typeLines.Name] = typeLines; //ncrunch: no coverage
			else
				yield return typeLines;
		while (withInternalDeps.Count > 0)
		{ //ncrunch: no coverage start
			var resolved = withInternalDeps.Values.
				Where(t => !t.DependentTypes.Any(dep => withInternalDeps.ContainsKey(dep))).
				Select(t => t.Name).ToList();
			if (resolved.Count == 0)
			{
				// Circular or unresolvable dependencies — yield remaining types as-is
				foreach (var remaining in withInternalDeps.Values)
					yield return remaining;
				yield break;
			}
			foreach (var name in resolved)
			{
				yield return withInternalDeps[name];
				withInternalDeps.Remove(name);
			}
		} //ncrunch: no coverage end
	}

	/// <summary>
	/// Evaluates a Strict expression like "TypeName(args).Method" or "TypeName(args)" (calls Run).
	/// The result is printed to Console if the method returns a value.
	/// Example: runner.RunExpression("FibonacciRunner(5).Compute") prints "5".
	/// </summary>
	public async Task RunExpression(string expression)
	{
		var (typeName, constructorArgs, methodName) = ParseExpressionArg(expression);
		var targetType = typeName == mainType.Name
			? mainType
			: package.GetType(typeName);
		var method = methodName != null
			? targetType.Methods.FirstOrDefault(m => m.Name == methodName && m.Parameters.Count == 0) ??
			throw new InvalidOperationException("Method " + methodName + " not found in " + targetType.Name)
			: targetType.Methods.FirstOrDefault(
				m => m.Name == Method.Run && m.Parameters.Count == 0) ?? //ncrunch: no coverage
			throw new InvalidOperationException("No Run method found in " + targetType.Name);
		var body = method.GetBodyAndParseIfNeeded();
		var expressions = body is Body bodyExpr
			? bodyExpr.Expressions
			: [body];
		var instance = new ValueInstance(targetType, BuildInstanceValueArray(targetType, constructorArgs));
		var instructions = new BinaryGenerator(
			new InstanceInvokedMethod(expressions, EmptyArguments, instance, method.ReturnType),
			new Registry()).Generate();
		var vm = new VirtualMachine(package);
		vm.Execute(OptimizeBytecode(instructions));
		if (vm.Returns.HasValue)
			Console.WriteLine(vm.Returns.Value.ToExpressionCodeString());
	}

	private static (string TypeName, double[] ConstructorArgs, string? MethodName) ParseExpressionArg(
		string expression)
	{
		var dotIndex = expression.LastIndexOf('.');
		string? methodName = null;
		var typeAndArgs = expression;
		if (dotIndex >= 0 && expression.IndexOf('(') < dotIndex)
		{
			methodName = expression[(dotIndex + 1)..];
			typeAndArgs = expression[..dotIndex];
		}
		var parenIndex = typeAndArgs.IndexOf('(');
		var typeName = typeAndArgs[..parenIndex];
		var argsContent = typeAndArgs[(parenIndex + 1)..^1].Trim();
		var constructorArgs = argsContent.Length == 0
			? Array.Empty<double>()
			: argsContent.Split(',').Select(argStr =>
				double.Parse(argStr.Trim(), CultureInfo.InvariantCulture)).ToArray();
		return (typeName, constructorArgs, methodName);
	}

	private ValueInstance[] BuildInstanceValueArray(Type type, double[] constructorArgs)
	{
		var numberType = package.GetType(Type.Number);
		var members = type.Members;
		var values = new ValueInstance[members.Count];
		var argIndex = 0;
		for (var memberIndex = 0; memberIndex < members.Count; memberIndex++)
			values[memberIndex] = members[memberIndex].Type.IsTrait
				? new ValueInstance(members[memberIndex].Type, 0)
				: argIndex < constructorArgs.Length
					? new ValueInstance(numberType, constructorArgs[argIndex++])
					: new ValueInstance(members[memberIndex].Type, 0);
		return values;
	}
*/
}