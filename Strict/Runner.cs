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
/// Runs or builds a .strict source file via its Run method or a supplied expression.
/// Caches .strictbinary bytecode for later runs, regenerating when source is newer.
/// Loading a .strictbinary is fully self-contained and needs no source packages at all.
/// </summary>
public sealed class Runner
{
	public Runner(string strictFilePath, string expressionToRun = Method.Run,
		bool enableDetailedOutput = false)
	{
		this.strictFilePath = strictFilePath;
		this.expressionToRun = expressionToRun;
		this.enableDetailedOutput = enableDetailedOutput;
		parser = new MethodExpressionParser();
		repositories = new Repositories(parser);
		Log("Strict.Runner: " + strictFilePath);
	}

	private readonly string strictFilePath;
	private readonly string expressionToRun;
	private readonly bool enableDetailedOutput;
	private readonly MethodExpressionParser parser;
	private readonly Repositories repositories;
	private readonly List<long> stepTimes = [];
	private bool IsExpressionInvocation =>
		expressionToRun != Method.Run && expressionToRun.Contains('(');
	private string[] ProgramArguments =>
		expressionToRun == Method.Run || IsExpressionInvocation
			? []
			: expressionToRun.Split(' ',
				StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);

	private void Log(string message)
	{
		if (enableDetailedOutput)
			Console.WriteLine(message);
	}

	/// <summary>
	/// Generates a platform-specific executable from the compiled instructions. Uses MLIR when
	/// -mlir is specified, LLVM IR when -llvm is specified, otherwise NASM + gcc/clang pipeline.
	/// Throws <see cref="ToolNotFoundException"/> if required tools are missing.
	/// </summary>
	public async Task Build(Platform platform, CompilerBackend backend = CompilerBackend.MlirDefault)
	{
		if (IsExpressionInvocation)
			throw new CannotBuildExecutableWithCustomExpression();
		var binary = await GetBinary();
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
	/// Returns a BinaryExecutable. For .strictbinary input or a valid cache, loads directly
	/// without any package — the binary is fully self-contained. Only loads source packages
	/// when we need to compile from .strict source files.
	/// </summary>
	private async Task<BinaryExecutable> GetBinary()
	{
		if (Path.GetExtension(strictFilePath) == BinaryExecutable.Extension)
			return LogTiming("Loading " + strictFilePath, () => new BinaryExecutable(strictFilePath));
#if !DISABLE_BINARY_CACHE
		var cachedBinaryFilePath = Path.ChangeExtension(strictFilePath, BinaryExecutable.Extension);
		if (File.Exists(cachedBinaryFilePath))
		{
			var binaryTime = new FileInfo(cachedBinaryFilePath).LastWriteTimeUtc;
			var sourceTime = new FileInfo(strictFilePath).LastWriteTimeUtc;
			if (binaryTime >= sourceTime)
			{
				try
				{
					var binary = LogTiming("Loading cached " + cachedBinaryFilePath,
						() => new BinaryExecutable(cachedBinaryFilePath));
					Log("Using cached " + cachedBinaryFilePath + " from " + binaryTime);
					return binary;
				}
				catch (Exception ex) when (ex is BinaryType.InvalidVersion
					or BinaryExecutable.InvalidFile or BinaryExecutable.TypeNotFoundForBytecode
					or ParsingFailed or Type.TypeAlreadyExistsInPackage
					or Context.NameMustBeAWordWithoutAnySpecialCharactersOrNumbers
					or Context.TypeNotFound or EndOfStreamException)
				{
					Log("Cached binary incompatible: " + ex.Message + ", regenerating ..");
				}
			}
			else
				Log("Cached binary outdated (" + binaryTime + " < " + sourceTime + "), regenerating ..");
		}
#endif
		var package = await LoadBasePackage();
		return await LoadFromSourceAndSaveBinary(package);
	}

	private async Task<Package> LoadBasePackage()
	{
		var basePackage = await repositories.LoadStrictPackage();
		var sourceDir = Path.GetDirectoryName(Path.GetFullPath(strictFilePath))!;
		var strictRoot = Path.GetFullPath(basePackage.FolderPath);
		if (!sourceDir.StartsWith(strictRoot, StringComparison.OrdinalIgnoreCase) ||
			string.Equals(sourceDir, strictRoot, StringComparison.OrdinalIgnoreCase) ||
			IsExamplesDir(sourceDir))
			return basePackage;
		var relative = Path.GetRelativePath(strictRoot, sourceDir).
			Replace(Path.DirectorySeparatorChar, Context.ParentSeparator).
			Replace(Path.AltDirectorySeparatorChar, Context.ParentSeparator);
		return await repositories.LoadStrictPackage(
			nameof(Strict) + Context.ParentSeparator + relative);
	}

	private static bool IsExamplesDir(string dir) =>
		dir.Split(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar).Any(part =>
			part.Equals("Examples", StringComparison.OrdinalIgnoreCase));

	private async Task<BinaryExecutable> LoadFromSourceAndSaveBinary(Package package)
	{
		var typeName = Path.GetFileNameWithoutExtension(strictFilePath);
		var existingType = package.FindDirectType(typeName);
		Type mainType;
		if (existingType == null)
		{
			var typeLines = new TypeLines(typeName, await File.ReadAllLinesAsync(strictFilePath));
			mainType = new Type(package, typeLines).ParseMembersAndMethods(parser);
		}
		else if (existingType.Methods.Any(method => !method.IsTrait))
			mainType = existingType;
		else
		{
			//TODO: this seems a bit strange
			package.Remove(existingType);
			var typeLines = new TypeLines(typeName, await File.ReadAllLinesAsync(strictFilePath));
			mainType = new Type(package, typeLines).ParseMembersAndMethods(parser);
		}
		if (enableDetailedOutput)
		{
			Parse(mainType);
			Validate(mainType);
			RunTests(package, mainType);
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

	private void RunTests(Package package, Type mainType) =>
		Log(LogTiming(nameof(RunTests) + " " + strictFilePath, () =>
		{
			var testExecutor = new TestInterpreter(package);
			testExecutor.RunAllTestsInType(mainType);
			return testExecutor.Statistics.ToString();
		}));

	private BinaryExecutable GenerateBinaryExecutable(Type mainType) =>
		LogTiming(nameof(GenerateBinaryExecutable), () =>
		{
			var runMethods = mainType.Methods.Where(method => method.Name == Method.Run).ToArray();
			if (runMethods.Length == 0)
				throw new NotSupportedException("No Run method found in " + mainType.Name);
			var preferredEntryMethod =
				runMethods.FirstOrDefault(method => method.Parameters.Count == 0) ?? runMethods[0];
			return BinaryGenerator.GenerateFromRunMethods(preferredEntryMethod, runMethods);
		});

	private void OptimizeBytecode(BinaryExecutable executable) =>
		Log(LogTiming(nameof(OptimizeBytecode), () =>
		{
			var beforeCount = executable.TotalInstructionsCount;
			var allOptimizers = new AllInstructionOptimizers();
			allOptimizers.Optimize(executable);
			var removed = beforeCount - executable.TotalInstructionsCount;
			return "Removed instructions: " + removed + " (" + removed * 100 / beforeCount +
				"%) with " + allOptimizers.NumberOfOptimizers + " optimizers.";
		}));

	private BinaryExecutable CacheStrictExecutable(BinaryExecutable binary)
	{
		var outputFilePath = Path.ChangeExtension(strictFilePath, BinaryExecutable.Extension);
		try
		{
			binary.Serialize(outputFilePath);
			Log("Saving " + new FileInfo(outputFilePath).Length + " bytes of bytecode to: " +
				outputFilePath);
		}
		catch (NotSupportedException ex)
		{
			Log("Bytecode serialization not yet supported for this program: " + ex.Message);
		}
		return binary;
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
			var runMethod = FindRunMethodForArguments(binary);
			binary.SetEntryPoint(
				binary.MethodsPerType.First(typeData =>
					typeData.Value.MethodGroups.TryGetValue(Method.Run, out var overloads) &&
					overloads.Contains(runMethod)).Key,
				Method.Run, runMethod.parameters.Count, runMethod.ReturnTypeName);
			var arguments = BuildProgramArguments(binary, runMethod);
			LogTiming(nameof(Run),
				() => new VirtualMachine(binary).Execute(initialVariables: arguments));
		}
		else
			LogTiming(nameof(Run), () => new VirtualMachine(binary).Execute());
		Console.WriteLine("Executed " + strictFilePath + " via " + nameof(VirtualMachine) + " in " +
			TimeSpan.FromTicks(stepTimes.Sum()).ToString(@"s\.ffffff") + "s");
		stepTimes.Clear();
	}

	public async Task RunExpression(string expressionString)
	{
		var typeName = Path.GetFileNameWithoutExtension(strictFilePath);
		var package = await LoadBasePackage();
		var sourceLines = await File.ReadAllLinesAsync(strictFilePath);
		var targetType =
			new Type(package, new TypeLines(typeName, sourceLines)).ParseMembersAndMethods(parser);
		try
		{
			var method = new Method(targetType, 0, parser,
				[nameof(RunExpression), "\t" + expressionString]);
			var call = new MethodCall(method);
			var binary = LogTiming(nameof(GenerateBinaryExecutable),
				() => new BinaryGenerator(call).Generate());
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

	private BinaryMethod FindRunMethodForArguments(BinaryExecutable binary)
	{
		var runMethods = binary.GetRunMethods();
		return runMethods.FirstOrDefault(method =>
				method.parameters.Count == ProgramArguments.Length) ??
			runMethods.FirstOrDefault(method => method.parameters.Count == 1 &&
				ResolveType(binary, method.parameters[0].FullTypeName).IsList) ??
			throw new NotSupportedException("No Run method accepts " + ProgramArguments.Length +
				" arguments.");
	}

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
			values[parameter.Name] =
				CreateValueInstance(ResolveType(binary, parameter.FullTypeName),
					ProgramArguments[index]);
		}
		return values;
	}

	private static Type ResolveType(BinaryExecutable binary, string fullTypeName)
	{
		if (fullTypeName.Contains(Context.ParentSeparator))
		{
			var found = binary.basePackage.FindFullType(fullTypeName);
			if (found != null)
				return found;
		}
		return binary.basePackage.FindType(fullTypeName) ??
			binary.basePackage.FindType(
				fullTypeName[(fullTypeName.LastIndexOf(Context.ParentSeparator) + 1)..]) ??
			binary.basePackage.GetType(fullTypeName);
	}

	private static ValueInstance CreateValueInstance(Type targetType, string argument)
	{
		if (targetType.IsNumber)
			return new ValueInstance(targetType, double.Parse(argument, CultureInfo.InvariantCulture));
		if (targetType.IsText)
			return new ValueInstance(argument);
		if (targetType.IsBoolean)
			return new ValueInstance(targetType, bool.Parse(argument));
		if (targetType.Name == "Path")
			return new ValueInstance(targetType, [new ValueInstance(argument)]);
		throw new NotSupportedException(
			"Only Number, Text, Boolean, Path and List arguments are supported.");
	}

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

	private void PrintCompilationSummary(CompilerBackend backend, Platform platform,
		string exeFilePath) =>
		Console.WriteLine("Compiled " + strictFilePath + " via " + backend + " in " +
			TimeSpan.FromTicks(stepTimes.Sum()).ToString(@"s\.ffffff") + "s to " + platform +
			" executable of " + new FileInfo(exeFilePath).Length + " bytes to: " + exeFilePath);
}