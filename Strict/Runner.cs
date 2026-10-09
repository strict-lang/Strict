// When things are in flux, force generating a new .strictbinary every time by disabling the cache
//#define DISABLE_BINARY_CACHE
using System.Globalization;
using Strict.Bytecode;
using Strict.Bytecode.Serialization;
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
/// Runs or builds a .strict source file via its Run method or a supplied expression.
/// Caches .strictbinary bytecode for later runs, regenerating when source is newer.
/// Loading a .strictbinary is fully self-contained and needs no source packages at all.
/// </summary>
public sealed partial class Runner
{
	public Runner(string strictFilePath, string expressionToRun = Method.Run,
		bool enableDetailedOutput = false)
	{
		packageDirectory = Directory.Exists(strictFilePath)
			? Path.TrimEndingDirectorySeparator(Path.GetFullPath(strictFilePath))
			: null;
		this.strictFilePath = packageDirectory == null
			? strictFilePath
			: Path.Combine(packageDirectory, Path.GetFileName(packageDirectory) + Type.Extension);
		if (packageDirectory == null && !File.Exists(strictFilePath))
			throw new StrictFileNotFound(strictFilePath);
		this.expressionToRun = expressionToRun;
		this.enableDetailedOutput = enableDetailedOutput;
		parser = new MethodExpressionParser();
		repositories = new Repositories(parser);
		Log("Strict.Runner: " + strictFilePath);
	}

	private readonly string strictFilePath;

	private readonly string? packageDirectory;

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
			if (binaryTime >= sourceTime && binaryTime >= RuntimeBuildTime &&
				!DirectoryHasNewerStrictFile(Path.GetDirectoryName(Path.GetFullPath(strictFilePath)),
					binaryTime))
				try
				{
					var binary = LogTiming("Loading cached " + cachedBinaryFilePath,
						() => new BinaryExecutable(cachedBinaryFilePath));
					if (!UsedPackageHasNewerStrictFile(binary, binaryTime))
					{
						Log("Using cached " + cachedBinaryFilePath + " from " + binaryTime);
						return binary;
					}
					Log("Cached binary outdated, a used package changed, regenerating ..");
				}
				catch (Exception ex) when (ex is BinaryType.InvalidVersion or BinaryExecutable.InvalidFile
					or BinaryExecutable.TypeNotFoundForBytecode or ParsingFailed
					or Type.TypeAlreadyExistsInPackage
					or Context.NameMustBeAWordWithoutAnySpecialCharactersOrNumbers or Context.TypeNotFound
					or EndOfStreamException)
				{
					Log("Cached binary incompatible: " + ex.Message + ", regenerating ..");
				}
			else
				Log("Cached binary outdated (" + binaryTime + " < " + sourceTime + "), regenerating ..");
		}
#endif
		var package = await LogTimingAsync("Load packages", LoadBasePackage);
		return await LoadFromSourceAndSaveBinary(package);
	}

	private async Task<Package> LoadBasePackage()
	{
		var basePackage = await repositories.LoadStrictPackage();
		if (packageDirectory != null)
			return await repositories.LoadFromPath(
				nameof(Strict) + Context.ParentSeparator + Path.GetFileName(packageDirectory),
				packageDirectory);
		var sourceDir = Path.GetDirectoryName(Path.GetFullPath(strictFilePath))!;
		var strictRoot = Path.GetFullPath(basePackage.FolderPath);
		if (!sourceDir.StartsWith(strictRoot, StringComparison.OrdinalIgnoreCase) ||
			string.Equals(sourceDir, strictRoot, StringComparison.OrdinalIgnoreCase))
			return basePackage;
		var relative = Path.GetRelativePath(strictRoot, sourceDir).
			Replace(Path.DirectorySeparatorChar, Context.ParentSeparator).
			Replace(Path.AltDirectorySeparatorChar, Context.ParentSeparator);
		return await repositories.LoadStrictPackage(nameof(Strict) + Context.ParentSeparator +
			relative);
	}

	/// <summary>
	/// A cache compiled by an older parser, generator or optimizer is outdated as well.
	/// </summary>
	private static readonly DateTime RuntimeBuildTime = Directory.
		EnumerateFiles(AppContext.BaseDirectory, nameof(Strict) + "*.dll").
		Select(File.GetLastWriteTimeUtc).DefaultIfEmpty(DateTime.MinValue).Max();

	private async Task<BinaryExecutable> LoadFromSourceAndSaveBinary(Package package)
	{
		var typeName = Path.GetFileNameWithoutExtension(strictFilePath);
		var existingType = package.FindDirectType(typeName);
		Type mainType;
		if (existingType == null)
		{
			var typeLines = new TypeLines(typeName, TypeLines.FromFile(strictFilePath));
			mainType = new Type(package, typeLines).ParseMembersAndMethods(parser);
			mainType.ValidateMembersAndVariablesAreUsed();
		}
		else if (existingType.Methods.Any(method => !method.IsTrait))
		{
			mainType = existingType;
		}
		else
		{
			//TODO: this seems a bit strange
			package.Remove(existingType);
			var typeLines = new TypeLines(typeName, TypeLines.FromFile(strictFilePath));
			mainType = new Type(package, typeLines).ParseMembersAndMethods(parser);
			mainType.ValidateMembersAndVariablesAreUsed();
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
				throw new NoRunMethodFound(mainType.Name);
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
			return "Removed instructions: " + removed + " (" + removed * 100 / beforeCount + "%) with " +
				allOptimizers.NumberOfOptimizers + " optimizers.";
		}));

	private T LogTiming<T>(string message, Func<T> callToTime)
	{
		var startTicks = DateTime.UtcNow.Ticks;
		var startAllocations = GC.GetAllocatedBytesForCurrentThread();
		try
		{
			return callToTime();
		}
		finally
		{
			LogElapsed(message, startTicks,
				GC.GetAllocatedBytesForCurrentThread() - startAllocations);
		}
	}

	/// <summary>
	/// Package loading continues on other threads, so all allocations are counted.
	/// </summary>
	private async Task<T> LogTimingAsync<T>(string message, Func<Task<T>> callToTime)
	{
		var startTicks = DateTime.UtcNow.Ticks;
		var startAllocations = GC.GetTotalAllocatedBytes(true);
		try
		{
			return await callToTime();
		}
		finally
		{
			LogElapsed(message, startTicks, GC.GetTotalAllocatedBytes(true) - startAllocations);
		}
	}

	private void LogElapsed(string message, long startTicks, long allocatedBytes)
	{
		var endTicks = DateTime.UtcNow.Ticks;
		Log(message + " Time: " + TimeSpan.FromTicks(endTicks - startTicks).TotalMilliseconds +
			" ms, allocated: " + allocatedBytes / 1024 + " KB");
		stepTimes.Add(endTicks - startTicks);
	}

	public async Task Run()
	{
		if (IsExpressionInvocation)
		{
			await RunExpression(expressionToRun);
			return;
		}
		var binary = await GetBinary();
		if (ProgramArguments.Length > 0 ||
			binary.GetRunMethods().All(method => method.parameters.Count > 0))
		{
			var runMethod = FindRunMethodForArguments(binary);
			binary.SetEntryPoint(
				binary.MethodsPerType.First(typeData =>
					typeData.Value.MethodGroups.TryGetValue(Method.Run, out var overloads) &&
					overloads.Contains(runMethod)).Key, Method.Run, runMethod.parameters.Count,
				runMethod.ReturnTypeName);
			var arguments = BuildProgramArguments(binary, runMethod);
			LogTiming(nameof(Run), () => new VirtualMachine(binary).Execute(initialVariables: arguments));
		}
		else
		{
			LogTiming(nameof(Run), () => new VirtualMachine(binary).Execute());
		}
		Console.WriteLine("Executed " + strictFilePath + " via " + nameof(VirtualMachine) + " in " +
			TimeSpan.FromTicks(stepTimes.Sum()).ToString(@"s\.ffffff") + "s");
		stepTimes.Clear();
	}

	public async Task RunExpression(string expressionString)
	{
		var typeName = Path.GetFileNameWithoutExtension(strictFilePath);
		var package = await LoadBasePackage();
		var existingType = package.FindDirectType(typeName);
		var targetType = existingType ?? new Type(package,
			new TypeLines(typeName, TypeLines.FromFile(strictFilePath))).ParseMembersAndMethods(parser);
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
			if (existingType == null)
				targetType.Dispose();
		}
	}

	private BinaryMethod FindRunMethodForArguments(BinaryExecutable binary)
	{
		var runMethods = binary.GetRunMethods();
		return
			runMethods.FirstOrDefault(method => method.parameters.Count == ProgramArguments.Length) ??
			runMethods.FirstOrDefault(method => method.parameters.Count == 1 &&
				ResolveType(binary, method.parameters[0].FullTypeName).IsList) ??
			throw new NoRunMethodAcceptsArguments(ProgramArguments.Length);
	}

	private IReadOnlyDictionary<string, ValueInstance>? BuildProgramArguments(BinaryExecutable binary,
		BinaryMethod runMethod)
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
			throw new RunArgumentCountMismatch(runMethod.parameters.Count, ProgramArguments.Length);
		var values = new Dictionary<string, ValueInstance>(runMethod.parameters.Count);
		for (var index = 0; index < runMethod.parameters.Count; index++)
		{
			var parameter = runMethod.parameters[index];
			values[parameter.Name] = CreateValueInstance(ResolveType(binary, parameter.FullTypeName),
				ProgramArguments[index]);
		}
		return values;
	}

	private static Type ResolveType(BinaryExecutable binary, string fullTypeName)
	{
		if (fullTypeName.Contains(Context.ParentSeparator))
		{
			var found = binary.TypeResolver.FindFullType(fullTypeName);
			if (found != null)
				return found;
		}
		return binary.TypeResolver.FindType(fullTypeName) ??
			binary.TypeResolver.FindType(
				fullTypeName[(fullTypeName.LastIndexOf(Context.ParentSeparator) + 1)..]) ??
			binary.TypeResolver.GetType(fullTypeName);
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
		throw new UnsupportedRunArgumentType(targetType.Name);
	}

	private void
		PrintCompilationSummary(CompilerBackend backend, Platform platform, string exeFilePath) =>
		Console.WriteLine("Compiled " + strictFilePath + " via " + backend + " in " +
			TimeSpan.FromTicks(stepTimes.Sum()).ToString(@"s\.ffffff") + "s to " + platform +
			" executable of " + new FileInfo(exeFilePath).Length + " bytes to: " + exeFilePath);

	public sealed class StrictFileNotFound(string filePath)
		: Exception(Path.GetFullPath(filePath) + " does not exist");

	public sealed class NoRunMethodFound(string typeName)
		: Exception("No Run method found in " + typeName);

	public sealed class NoRunMethodAcceptsArguments(int count)
		: Exception("No Run method accepts " + count + " arguments.");

	public sealed class RunArgumentCountMismatch(int expected, int given)
		: Exception("Run expects " + expected + " arguments, but got " + given + ".");

	public sealed class UnsupportedRunArgumentType(string typeName) : Exception(typeName +
		" is not supported, only Number, Text, Boolean, Path and List arguments are supported.");
}
