// When things are in flux, force generating a new .strictbinary every time by disabling the cache
//#define DISABLE_BINARY_CACHE
using Strict.Bytecode;
using Strict.Compiler;
using Strict.Compiler.Assembly;
using System.Globalization;
using Strict.Expressions;
using Strict.Language;
using Strict.Optimizers;
using Strict.TestRunner;
using Strict.Validators;
using Type = Strict.Language.Type;
using Strict.Bytecode.Serialization;

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
	private bool IsExpressionInvocation =>
		expressionToRun != Method.Run && expressionToRun.Contains('(');
	private string[] ProgramArguments =>
		expressionToRun == Method.Run || IsExpressionInvocation
			? []
			: expressionToRun.Split(' ',
				StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);

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
		if (IsExpressionInvocation)
			throw new CannotBuildExecutableWithCustomExpression();
		var binary = await GetBinary();
		//TODO: convoluted! fix and remove this mess
		if (binary.GetRunMethods().Any(method => method.parameters.Count > 0))
		{
			var launcherPath = CreateManagedLauncher(platform);
			PrintLauncherSummary(platform, launcherPath);
			return;
		}
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

	public class CannotBuildExecutableWithCustomExpression : Exception;

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
			//TODO: convoluted! fix and remove this mess
			BinaryExecutable binary;
			try
			{
				binary = new BinaryExecutable(cachedBinaryPath, basePackage);
			}
			catch (Exception ex) when (ex is BinaryType.InvalidVersion or BinaryExecutable.InvalidFile)
			{
				Log("Cached " + cachedBinaryPath + " is no longer compatible: " + ex.Message);
				return await LoadFromSourceAndSaveBinary(basePackage);
			}
			var binaryLastModified = new FileInfo(cachedBinaryPath).LastWriteTimeUtc;
			var sourceLastModified = new FileInfo(strictFilePath).LastWriteTimeUtc;
			foreach (var typeFullName in binary.MethodsPerType.Keys)
			{
				var fileLastModified =
					new FileInfo(typeFullName + Type.Extension).LastWriteTimeUtc;
				if (fileLastModified > sourceLastModified)
					sourceLastModified = fileLastModified;
			}
#if !DISABLE_BINARY_CACHE
			if (binaryLastModified >= sourceLastModified)
			{
				Log("Cached " + cachedBinaryPath + " from " + binaryLastModified +
					" is still good, using it. Latest source file change: " + sourceLastModified);
				return binary;
			}
#endif
			Log("Cached " + cachedBinaryPath + " is outdated from " + binaryLastModified +
				", source modified at " + sourceLastModified);
		}
		return await LoadFromSourceAndSaveBinary(basePackage);
	}

	private async Task<Package> GetPackage(string name)
	{
		if (skipPackageSearchAndUseThisTestPackage != null)
			return skipPackageSearchAndUseThisTestPackage;
		return await LogTiming(nameof(GetPackage) + " " + name, async () =>
		{
			if (!name.StartsWith(nameof(Strict), StringComparison.Ordinal))
				throw new NotSupportedException("No github package search ability was implemented yet, only Strict packages work for now: " + name);
			return await repositories.LoadStrictPackage(name);
		});
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
		var existingType = basePackage.FindDirectType(typeName);
		Type mainType;
		if (existingType == null)
		{
			var typeLines = new TypeLines(typeName, await File.ReadAllLinesAsync(strictFilePath));
			mainType = new Type(basePackage, typeLines).ParseMembersAndMethods(parser);
		}
		else if (existingType.Methods.Any(method => !method.IsTrait))
			mainType = existingType;
		else
		{
			basePackage.Remove(existingType);
			var typeLines = new TypeLines(typeName, await File.ReadAllLinesAsync(strictFilePath));
			mainType = new Type(basePackage, typeLines).ParseMembersAndMethods(parser);
		}
		if (enableTestsAndDetailedOutput)
		{
			Parse(mainType);
			Validate(mainType);
			RunTests(basePackage, mainType);
		}
		var executable = GenerateBinaryExecutable(mainType);
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

	private BinaryExecutable GenerateBinaryExecutable(Type mainType) =>
		LogTiming(nameof(GenerateBinaryExecutable), () =>
		{
			var runMethods = mainType.Methods.Where(method => method.Name == Method.Run).ToArray();
			if (runMethods.Length == 0)
				throw new NotSupportedException("No Run method found in " + mainType.Name);
			var preferredEntryMethod = runMethods.FirstOrDefault(method => method.Parameters.Count == 0) ??
				runMethods[0];
			return BinaryGenerator.GenerateFromRunMethods(preferredEntryMethod, runMethods);
		});

	private BinaryExecutable GenerateExpressionBinaryExecutable(Expression entryPoint) =>
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

	/// <summary>
	/// Evaluates a Strict expression like "TypeName(args).Method" or "TypeName(args)" (calls Run).
	/// The result is printed to Console if the method returns a value.
	/// Example: runner.RunExpression("FibonacciRunner(5).Compute") prints "5".
	/// </summary>
	public async Task RunExpression(string expressionString)
	{
		//TODO: still duplicated code, should be the same as Run!
		var typeName = Path.GetFileNameWithoutExtension(strictFilePath);
		var basePackage = skipPackageSearchAndUseThisTestPackage ?? await GetPackage(nameof(Strict));
		var sourceLines = await File.ReadAllLinesAsync(strictFilePath);
		var targetType = new Type(basePackage, new TypeLines(typeName, sourceLines)).ParseMembersAndMethods(parser);
		try
		{
			var expression = parser.ParseExpression(
				new Body(new Method(targetType, 0, parser, [nameof(RunExpression)])),
				expressionString);
			var binary = GenerateExpressionBinaryExecutable(expression);
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

	//TODO: why do we care? just use BinaryExecutable.EntryPoint!
	private BinaryMethod FindRunMethodForArguments(BinaryExecutable binary)
	{
		var runMethods = binary.GetRunMethods();
		var exactMatch = runMethods.FirstOrDefault(method =>
			method.parameters.Count == ProgramArguments.Length);
		if (exactMatch != null)
			return exactMatch;
		var listMatch = runMethods.FirstOrDefault(method => method.parameters.Count == 1 &&
			ResolveType(binary, method.parameters[0].FullTypeName).IsList);
		if (listMatch != null)
			return listMatch;
		throw new NotSupportedException("No Run method accepts " + ProgramArguments.Length +
			" arguments.");
	}

	//TODO: overcomplicated
	private IReadOnlyDictionary<string, ValueInstance>? BuildProgramArguments(
		BinaryExecutable binary, BinaryMethod runMethod)
	{
		if (runMethod.parameters.Count == 0)
			return null;
		if (runMethod.parameters.Count == 1)
		{
			var listType = ResolveType(binary, runMethod.parameters[0].FullTypeName);
			if (listType.IsList)
			{
				var elementType = ((GenericTypeImplementation)listType).ImplementationTypes[0];
				var listItems = ProgramArguments.Select(argument =>
					CreateValueInstance(elementType, argument)).ToArray();
				return new Dictionary<string, ValueInstance>
				{
					[runMethod.parameters[0].Name] = new(listType, listItems)
				};
			}
		}
		if (runMethod.parameters.Count != ProgramArguments.Length)
			throw new NotSupportedException("Run expects " + runMethod.parameters.Count +
				" arguments, but got " + ProgramArguments.Length + ".");
		var values = new Dictionary<string, ValueInstance>(runMethod.parameters.Count);
		for (var index = 0; index < runMethod.parameters.Count; index++)
		{
			var parameter = runMethod.parameters[index];
			values[parameter.Name] = CreateValueInstance(ResolveType(binary, parameter.FullTypeName), ProgramArguments[index]);
		}
		return values;
	}

	//TODO: overcomplicated
	private static Type ResolveType(BinaryExecutable binary, string fullTypeName) =>
		(fullTypeName.Contains(Context.ParentSeparator)
			? binary.basePackage.FindFullType(fullTypeName)
			: null) ??
		binary.basePackage.FindType(fullTypeName) ??
		(fullTypeName.StartsWith(nameof(Strict) + Context.ParentSeparator, StringComparison.Ordinal)
			? binary.basePackage.FindType(fullTypeName[(nameof(Strict).Length + 1)..])
			: null) ??
		binary.basePackage.GetType(fullTypeName);

	private static ValueInstance CreateValueInstance(Type targetType, string argument)
	{
		if (targetType.IsNumber)
			return new ValueInstance(targetType, double.Parse(argument, CultureInfo.InvariantCulture));
		if (targetType.IsText)
			return new ValueInstance(argument);
		return targetType.IsBoolean
			? new ValueInstance(targetType, bool.Parse(argument))
			: throw new NotSupportedException("Only Number, Text, Boolean and List arguments are supported.");
	}

	private static string GetRunMethodTypeFullName(BinaryExecutable binary, BinaryMethod runMethod) =>
		binary.MethodsPerType.First(typeData => typeData.Value.MethodGroups.TryGetValue(Method.Run,
			out var overloads) && overloads.Contains(runMethod)).Key;

	//TODO: why do we have to do this here? shouldn't this be happening in generation?
	private string CreateManagedLauncher(Platform platform)
	{
		if (platform == Platform.Windows && !OperatingSystem.IsWindows() ||
			platform == Platform.Linux && !OperatingSystem.IsLinux() ||
			platform == Platform.MacOS && !OperatingSystem.IsMacOS())
			throw new NotSupportedException(
				"Runtime launcher builds require building on the target platform.");
		var runtimeDirectory = Path.GetDirectoryName(typeof(Program).Assembly.Location) ??
			throw new DirectoryNotFoundException("Strict runtime output directory not found.");
		var outputDirectory = Path.GetDirectoryName(Path.GetFullPath(strictFilePath)) ??
			throw new DirectoryNotFoundException("Output directory not found.");
		var runtimeExecutableName = OperatingSystem.IsWindows()
			? "Strict.exe"
			: "Strict";
		var outputExecutablePath = Path.Combine(outputDirectory, OperatingSystem.IsWindows()
			? Path.GetFileNameWithoutExtension(strictFilePath) + ".exe"
			: Path.GetFileNameWithoutExtension(strictFilePath));
		File.Copy(Path.Combine(runtimeDirectory, runtimeExecutableName), outputExecutablePath, true);
		foreach (var filePath in Directory.GetFiles(runtimeDirectory, "*.dll"))
			File.Copy(filePath, Path.Combine(outputDirectory, Path.GetFileName(filePath)), true);
		foreach (var filePath in Directory.GetFiles(runtimeDirectory, "*.json"))
			File.Copy(filePath, Path.Combine(outputDirectory, Path.GetFileName(filePath)), true);
		if (!OperatingSystem.IsWindows())
			File.SetUnixFileMode(outputExecutablePath,
				UnixFileMode.UserRead | UnixFileMode.UserWrite | UnixFileMode.UserExecute |
				UnixFileMode.GroupRead | UnixFileMode.GroupExecute | UnixFileMode.OtherRead |
				UnixFileMode.OtherExecute);
		return outputExecutablePath;
	}

	private static void PrintLauncherSummary(Platform platform, string exeFilePath) =>
		Console.WriteLine("Created " + platform + " executable launcher of " +
			new FileInfo(exeFilePath).Length + " bytes to: " + exeFilePath);

	public async Task Run()
	{
		if (IsExpressionInvocation)
		{
			await RunExpression(expressionToRun);
			return;
		}
		var binary = await GetBinary();
		if (ProgramArguments.Length > 0)
		{
			//TODO: why do we have to do this here? shouldn't this be happening in generation?
			var runMethod = FindRunMethodForArguments(binary);
			binary.SetEntryPoint(GetRunMethodTypeFullName(binary, runMethod), Method.Run,
				runMethod.parameters.Count, runMethod.ReturnTypeName);
			var programArguments = BuildProgramArguments(binary, runMethod);
			LogTiming(nameof(Run),
				() => new VirtualMachine(binary).Execute(initialVariables: programArguments));
		}
		else
			LogTiming(nameof(Run), () => new VirtualMachine(binary).Execute());
		Console.WriteLine("Executed " + strictFilePath + " via " + nameof(VirtualMachine) + " in " +
			TimeSpan.FromTicks(stepTimes.Sum()).ToString(@"s\.ffffff") + "s");
	}
}
//TODO: whole class is too long and complicated, this should all be doable in 300-400 lines max!