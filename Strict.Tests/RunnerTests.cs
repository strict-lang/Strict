using System.IO.Compression;
using System.Runtime.InteropServices;
using Strict.Bytecode;
using Strict.Bytecode.Serialization;
using Strict.Compiler;
using Strict.Expressions;
using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict.Tests;

public sealed class RunnerTests
{
	[SetUp]
	public void CreateTextWriter()
	{
		consoleWriter = new StringWriter();
		rememberConsole = Console.Out;
		Console.SetOut(consoleWriter);
	}

	private StringWriter consoleWriter = null!;
	private TextWriter rememberConsole = null!;

	[TestCase("")]
	[TestCase("/")]
	public async Task RunBaseTypesTestPackageFromDirectory(string suffix)
	{
		await new Runner(Path.Combine(FindRepoRoot(), "Examples", "BaseTypesTest") + suffix).Run();
		var expected = string.Join(Environment.NewLine, "Hello, World!", "Hello, Strict!", "3 + 4 = 7",
			"10 * 3 = 30", "(1, 2, 3).Sum = 6", "");
		Assert.That(consoleWriter.ToString(), Does.StartWith(expected));
		var binaryPath = Path.ChangeExtension(GetExamplesFilePath("BaseTypesTest/BaseTypesTest"),
			BinaryExecutable.Extension);
		var standalone = NativeProcessRunner.Run("dotnet",
			"\"" + StrictAssemblyForFreshProcess() + "\" \"" + binaryPath + "\"", 120000);
		Assert.That(standalone.ExitCode, Is.Zero, standalone.Output + standalone.Error);
		Assert.That(standalone.Output, Does.Contain(expected));
	}

	[Test]
	public void MissingStrictFileGivesClearError() =>
		Assert.That(async () => await new Runner("ImageProcessing/Missing.strict").Run(),
			Throws.InstanceOf<Runner.StrictFileNotFound>().With.Message.Contains("Missing.strict"));

	[Test]
	public void RelativeExecutableIsResolvedAgainstCurrentDirectory() =>
		Assert.That(NativeProcessRunner.ResolveExecutable("Compiler/output/add.exe"),
			Is.EqualTo(Path.GetFullPath("Compiler/output/add.exe")));

	[TestCase("NativeArithmetic", 20)]
	[TestCase("NativeConditions", 30)]
	[TestCase("NativeLoop", 45)]
	[Category("Slow")]
	public void StrictSourceCompilerBuildsAndRunsNativeExecutable(string example, int expected)
	{
		var root = FindRepoRoot();
		var result = NativeProcessRunner.Run("dotnet", "\"" + StrictAssemblyForFreshProcess() + "\" \"" +
			Path.Combine(root, "Compiler", "SourceCompiler.strict") + "\" \"" +
			Path.Combine(root, "Examples", example + Type.Extension) + "\"", 120000);
		Assert.That(result.Output, Does.Contain("Run returned " + expected), result.Output + result.Error);
	}

	[TestCaseSource(nameof(StrictProgramPaths))]
	[Category("Slow")]
	public void RunStrictProgramFromSourceAndCachedBinaryInFreshProcess(string relativePath)
	{
		var root = FindRepoRoot();
		var sourcePath = Path.Combine(root, relativePath);
		var hasRun = HasRunMethod(sourcePath);
		var arguments = ProgramArguments.TryGetValue(relativePath, out var argument)
			? " \"" + Path.Combine(root, argument) + "\""
			: "";
		File.Delete(Path.ChangeExtension(sourcePath, BinaryExecutable.Extension));
		var testDirectory = Directory.GetCurrentDirectory();
		Directory.SetCurrentDirectory(root);
		try
		{
			foreach (var inputPath in hasRun
				? [sourcePath, Path.ChangeExtension(sourcePath, BinaryExecutable.Extension)]
				: new[] { sourcePath })
			{
				var result = NativeProcessRunner.Run("dotnet",
					"\"" + StrictAssemblyForFreshProcess() + "\" \"" + inputPath + "\"" + arguments, 120000);
				if (hasRun)
					Assert.That(result.ExitCode, Is.Zero,
						inputPath + Environment.NewLine + result.Output + result.Error);
				else
					Assert.That(result.Output, Does.Contain("NoRunMethodFound: No Run method found in " +
						Path.GetFileNameWithoutExtension(relativePath)), result.Output + result.Error);
			}
		}
		finally
		{
			Directory.SetCurrentDirectory(testDirectory);
		}
	}

	private static bool HasRunMethod(string filePath) =>
		File.ReadLines(filePath).Any(line => line == Method.Run || line.StartsWith(Method.Run + "(",
			StringComparison.Ordinal) || line.StartsWith(Method.Run + " ", StringComparison.Ordinal));

	private static readonly Dictionary<string, string> ProgramArguments = new()
	{
		["Language/Parser.strict"] = "Examples/HelloLogger.strict",
		["Language/PackageTests.strict"] = "Examples/BaseTypesTest",
		["Compiler/SourceCompiler.strict"] = "Examples/NativeArithmetic.strict",
		["ImageProcessing/ProcessImage.strict"] = "ImageProcessing/test_image.jpg",
		["Process.strict"] = "Examples/HelloLogger.strict",
		["Expressions/RoundTrip.strict"] = "Expressions"
	};

	private static IEnumerable<string> StrictProgramPaths()
	{
		var root = FindRepoRoot();
		return StrictFolders().SelectMany(folder =>
				Directory.GetFiles(Path.Combine(root, folder), "*" + Type.Extension)).
			Select(file => Path.GetRelativePath(root, file).Replace('\\', '/')).Order();
	}

	private static IEnumerable<string> StrictFolders()
	{
		var root = FindRepoRoot();
		string[] projects = [".", "Math", "ImageProcessing", "Language", "Expressions", "Validators",
			"TestRunner", "HighLevelRuntime", "Bytecode", "Optimizers", "Runtime", "Compiler", "Examples"];
		return projects.Concat(Directory.GetDirectories(Path.Combine(root, "Examples")).
			Select(folder => Path.GetRelativePath(root, folder).Replace('\\', '/')).
			Where(folder => !folder.EndsWith("/bin") && !folder.EndsWith("/obj")));
	}

	[TestCaseSource(nameof(StrictFolders))]
	[Category("Slow")]
	public async Task StrictParserRoundTripsEveryLine(string folder)
	{
		var root = FindRepoRoot();
		await new Runner(Path.Combine(root, "Expressions", "RoundTrip" + Type.Extension),
			Path.Combine(root, folder)).Run();
		Assert.That(consoleWriter.ToString(), Does.Contain("Round trip mismatches: 0"));
	}

	[Test]
	public void RunStrictPackageLoaderPreservesTypeNamesAndLines()
	{
		var root = FindRepoRoot();
		var packagePath = Path.Combine(root, "Examples", "BaseTypesTest");
		var sourcePath = Path.Combine(root, "Language", "PackageTests.strict");
		foreach (var inputPath in new[] { sourcePath, Path.ChangeExtension(sourcePath, BinaryExecutable.Extension) })
		{
			using var process = new System.Diagnostics.Process();
			process.StartInfo = new System.Diagnostics.ProcessStartInfo("dotnet")
			{
				WorkingDirectory = root,
				UseShellExecute = false,
				RedirectStandardOutput = true,
				CreateNoWindow = true,
				ArgumentList = { StrictAssemblyForFreshProcess(), inputPath, packagePath }
			};
			process.Start();
			var output = process.StandardOutput.ReadToEnd();
			Assert.That(process.WaitForExit(30000), Is.True, output);
			Assert.That(process.ExitCode, Is.Zero, output);
			foreach (var file in Directory.GetFiles(packagePath, "*.strict"))
				Assert.That(output, Does.Contain(
					Environment.NewLine + Path.GetFileNameWithoutExtension(file) + ":" +
					File.ReadAllLines(file).Length + Environment.NewLine));
		}
	}
	private static readonly object FreshAssemblyGate = new();
	private static string? freshStrictAssembly;

	private static string StrictAssemblyForFreshProcess()
	{
		var location = typeof(Strict.Program).Assembly.Location;
		if (Environment.GetEnvironmentVariable("NCrunch") != "1")
			return location;
		lock (FreshAssemblyGate)
		{
			if (freshStrictAssembly != null)
				return freshStrictAssembly;
			var artifacts = Path.Combine(Path.GetTempPath(), "StrictFreshProcess");
			var projectFile = Path.Combine(FindRepoRoot(), "Strict", "Strict.csproj");
			var build = NativeProcessRunner.Run("dotnet",
				"build \"" + projectFile + "\" --artifacts-path \"" + artifacts +
				"\" --verbosity quiet --nologo", 120000);
			Assert.That(build.ExitCode, Is.Zero, build.Output + build.Error);
			var built = Directory.GetFiles(Path.Combine(artifacts, "bin"), "Strict.dll",
				SearchOption.AllDirectories);
			Assert.That(built, Has.Length.EqualTo(1), string.Join(Environment.NewLine, built));
			return freshStrictAssembly = built[0];
		}
	}

	[TearDown]
	public void RestoreConsole() => Console.SetOut(rememberConsole);

	[Test]
	public async Task RunDirectoryTestsUsesNativeDirectoryInTestsAndVirtualMachine()
	{
		await new Runner(GetExamplesFilePath("BaseTypesTest/DirectoryTests")).Run();
		Assert.That(consoleWriter.ToString(), Does.Contain("Directory exists: true"));
	}

	[Test]
	public async Task RunSimpleCalculator()
	{
		var asmFilePath = Path.ChangeExtension(SimpleCalculatorFilePath, ".asm");
		if (File.Exists(asmFilePath))
			File.Delete(asmFilePath); //ncrunch: no coverage
		try
		{
			await new Runner(SimpleCalculatorFilePath).Run();
		}
		//ncrunch: no coverage start
		catch (IOException)
		{
			Thread.Sleep(100);
			await new Runner(SimpleCalculatorFilePath).Run();
		} //ncrunch: no coverage end
		Assert.That(consoleWriter.ToString(),
			Does.StartWith("2 + 3 = 5" + Environment.NewLine + "2 * 3 = 6" + Environment.NewLine));
		Assert.That(File.Exists(asmFilePath), Is.False);
	}

	[Test]
	public async Task RunFromBytecodeFileProducesSameOutput()
	{
		var binaryFilePath = await GetExamplesBinaryFile("SimpleCalculator");
		await new Runner(binaryFilePath).Run();
		Assert.That(consoleWriter.ToString(),
			Does.StartWith("2 + 3 = 5" + Environment.NewLine + "2 * 3 = 6"));
	}

	[Test]
	public async Task CachedBinaryOlderThanRuntimeIsRegenerated()
	{
		var tempDirectory = Path.Combine(Path.GetTempPath(), "Strict" + Guid.NewGuid().ToString("N"));
		Directory.CreateDirectory(tempDirectory);
		var sourcePath = Path.Combine(tempDirectory, Path.GetFileName(SimpleCalculatorFilePath));
		var binaryPath = Path.ChangeExtension(sourcePath, BinaryExecutable.Extension);
		try
		{
			File.Copy(SimpleCalculatorFilePath, sourcePath);
			await new Runner(sourcePath).Run();
			var runtimeTime = File.GetLastWriteTimeUtc(typeof(Runner).Assembly.Location);
			File.SetLastWriteTimeUtc(sourcePath, runtimeTime.AddMinutes(-2));
			File.SetLastWriteTimeUtc(binaryPath, runtimeTime.AddMinutes(-1));
			await new Runner(sourcePath).Run();
			Assert.That(File.GetLastWriteTimeUtc(binaryPath), Is.GreaterThan(runtimeTime));
		}
		finally
		{
			Directory.Delete(tempDirectory, true);
		}
	}

	[Test]
	public async Task RunFromBytecodeFileWithoutStrictSourceFile()
	{
		var tempDirectory = Path.Combine(Path.GetTempPath(), "Strict" + Guid.NewGuid().ToString("N"));
		Directory.CreateDirectory(tempDirectory);
		var copiedSourceFilePath =
			Path.Combine(tempDirectory, Path.GetFileName(SimpleCalculatorFilePath));
		var copiedBinaryFilePath =
			Path.ChangeExtension(copiedSourceFilePath, BinaryExecutable.Extension);
		try
		{
			File.Copy(SimpleCalculatorFilePath, copiedSourceFilePath);
			await new Runner(copiedSourceFilePath).Run();
			Assert.That(File.Exists(copiedBinaryFilePath), Is.True);
			consoleWriter.GetStringBuilder().Clear();
			File.Delete(copiedSourceFilePath);
			await new Runner(copiedBinaryFilePath).Run();
			Assert.That(consoleWriter.ToString(),
				Does.StartWith("2 + 3 = 5" + Environment.NewLine + "2 * 3 = 6"));
		}
		finally
		{
			if (Directory.Exists(tempDirectory))
				Directory.Delete(tempDirectory, true);
		}
	}

	[Test]
	public void BuildWithExpressionEntryPointThrows()
	{
		var runner = new Runner(SimpleCalculatorFilePath, "(1, 2, 3).Length");
		Assert.That(async () => await runner.Build(Platform.Windows),
			Throws.TypeOf<Runner.CannotBuildExecutableWithCustomExpression>());
	}

	[Test]
	public async Task RunExpressionOnTypeInsidePackageDirectory()
	{
		await new Runner(GetExamplesFilePath("FizzBuzz"), "FizzBuzz(15).Classify").Run();
		Assert.That(consoleWriter.ToString(), Does.StartWith("FizzBuzz"));
	}

	[Test]
	public async Task ImplicitCallInsideLoopUsesMethodInstanceInVirtualMachine()
	{
		await new Runner(Path.Combine(FindRepoRoot(), "Compiler", "InstrToAsm.strict"),
			"InstrToAsm(BytecodeInstruction.ReturnOp(0), 0).FloatText(-3)").Run();
		Assert.That(consoleWriter.ToString(), Does.StartWith("-3.0"));
	}

	[Test]
	public async Task RunSucceedsWhileCachedBinaryIsOpenedByAnotherReader()
	{
		var source = Path.Combine(FindRepoRoot(), "Examples", "HelloLogger.strict");
		var binaryPath = Path.ChangeExtension(source, BinaryExecutable.Extension);
		await new Runner(source).Run();
		File.SetLastWriteTimeUtc(source, DateTime.UtcNow.AddMinutes(1));
		try
		{
			using var reader = new FileStream(binaryPath, FileMode.Open, FileAccess.Read, FileShare.Read);
			consoleWriter.GetStringBuilder().Clear();
			await new Runner(source).Run();
			Assert.That(consoleWriter.ToString(), Does.Contain("Hi"));
		}
		finally
		{
			File.SetLastWriteTimeUtc(source, DateTime.UtcNow);
		}
	}

	[Test]
	public async Task CachedBinaryIsOutdatedWhenUsedPackageChanged()
	{
		var entry = Path.Combine(FindRepoRoot(), "Compiler", "EmitTests.strict");
		await new Runner(entry).Run();
		var dependency = Path.Combine(FindRepoRoot(), "Bytecode", "BytecodeInstruction.strict");
		var originalTime = File.GetLastWriteTimeUtc(dependency);
		File.SetLastWriteTimeUtc(dependency, DateTime.UtcNow.AddMinutes(1));
		try
		{
			consoleWriter.GetStringBuilder().Clear();
			await new Runner(entry, Method.Run, true).Run();
			Assert.That(consoleWriter.ToString(), Does.Not.Contain("Using cached"));
		}
		finally
		{
			File.SetLastWriteTimeUtc(dependency, originalTime);
		}
	}

	[Test]
	public async Task AppendAfterLoopKeepsElementInVirtualMachine()
	{
		await new Runner(Path.Combine(FindRepoRoot(), "Bytecode", "InstructionList.strict"),
			"InstructionList.Empty.Append(BytecodeInstruction.ReturnOp(1)).Count").Run();
		Assert.That(consoleWriter.ToString(), Does.StartWith("1"));
	}

	[Test]
	public async Task InstructionListAtOutOfRangeInVirtualMachine()
	{
		await new Runner(Path.Combine(FindRepoRoot(), "Bytecode", "InstructionList.strict"),
			"InstructionList.Empty.At(0)").Run();
		Assert.That(consoleWriter.ToString(), Does.StartWith("(Return, 0, 0, )"));
	}

	[Test]
	public async Task AsmFileIsNotCreatedWhenRunningFromPrecompiledBytecode()
	{
		var asmPath = Path.ChangeExtension(SimpleCalculatorFilePath, ".asm");
		if (File.Exists(asmPath))
			File.Delete(asmPath); //ncrunch: no coverage
		var binaryPath = await GetExamplesBinaryFile("SimpleCalculator");
		await new Runner(binaryPath).Run();
		Assert.That(File.Exists(asmPath), Is.False);
	}

	[Test]
	public async Task SaveStrictBinaryWithTypeBytecodeEntriesOnly()
	{
		var binaryPath = await GetExamplesBinaryFile("SimpleCalculator");
		await using var archive = await ZipFile.OpenReadAsync(binaryPath);
		var entries = archive.Entries.Select(entry => entry.FullName.Replace('\\', '/')).ToList();
		Assert.That(
			entries.All(entry =>
				entry.EndsWith(BinaryType.BytecodeEntryExtension, StringComparison.OrdinalIgnoreCase)),
			Is.True);
		Assert.That(entries.Any(entry => entry.Contains("#", StringComparison.Ordinal)), Is.False);
		Assert.That(entries, Does.Contain("SimpleCalculator.bytecode"));
		Assert.That(entries, Does.Contain("Strict/Number.bytecode"));
		Assert.That(entries, Does.Contain("Strict/Logger.bytecode"));
		Assert.That(entries, Does.Contain("Strict/Text.bytecode"));
		Assert.That(entries, Does.Contain("Strict/Character.bytecode"));
		Assert.That(entries, Does.Contain("Strict/TextWriter.bytecode"));
	}

	[Test]
	public async Task RunSumWithProgramArguments()
	{
		await new Runner(SumFilePath, "5 10 20").Run();
		Assert.That(consoleWriter.ToString(), Does.Contain("35"));
	}

	[Test]
	public async Task RunSumWithDifferentProgramArgumentsDoesNotReuseCachedEntryPoint()
	{
		await new Runner(SumFilePath, "5 10 20").Run();
		consoleWriter.GetStringBuilder().Clear();
		await new Runner(SumFilePath, "1 2").Run();
		Assert.That(consoleWriter.ToString(), Does.Contain("3"));
	}

	[Test]
	public async Task RunSumWithNoArgumentsUsesEmptyList()
	{
		await new Runner(SumFilePath).Run();
		Assert.That(consoleWriter.ToString(), Does.Contain("0"));
	}

	[Test]
	public async Task RunAutofilledMutable() =>
		await new Runner(GetExamplesFilePath("AutofilledMutable")).Run();

	[Test]
	public async Task RunParseHelloLogger()
	{
		await new Runner(GetExamplesFilePath("Parsing/ParseHelloLogger")).Run();
		var output = consoleWriter.ToString();
		Assert.That(output, Does.Contain("Member(has): has logger"));
		Assert.That(output, Does.Contain("Member(mutable): mutable count = 0"));
		Assert.That(output, Does.Contain("Member(constant): constant Max = 100"));
		Assert.That(output, Does.Contain("Method: Run"));
		Assert.That(output, Does.Contain("Method: Add(other) Number"));
		Assert.That(output, Does.Contain("Body:"));
	}

	[Test]
	public async Task RunParseExpressions()
	{
		await new Runner(GetExamplesFilePath("Parsing/ParseExpressions")).Run();
		var output = consoleWriter.ToString();
		Assert.That(output, Does.Contain("Parsing HelloLogger.strict expressions"));
		Assert.That(output, Does.Contain("has member: logger"));
		Assert.That(output, Does.Contain("mutable member: count = 0"));
		Assert.That(output, Does.Contain("Method body expressions:"));
		Assert.That(output, Does.Contain("MethodCall: logger.Log"));
		Assert.That(output, Does.Contain("Return: total"));
		Assert.That(output, Does.Contain("If: condition=count > 0"));
		Assert.That(output, Does.Contain("For: iterator=items"));
		Assert.That(output, Does.Contain("Declaration: count = 5"));
	}

	[Test]
	public async Task RunParseMethodHeaders()
	{
		await new Runner(GetExamplesFilePath("Parsing/ParseMethodHeaders")).Run();
		var output = consoleWriter.ToString();
		Assert.That(output, Does.Contain("Parsing method headers from type definitions"));
		Assert.That(output, Does.Contain("Method: Run (no return type)"));
		Assert.That(output, Does.Contain("Method: Add returns Number"));
		Assert.That(output, Does.Contain("Method: GetName returns Text"));
		Assert.That(output, Does.Contain("Method: IsDone returns Boolean"));
		Assert.That(output, Does.Contain("Body expression types:"));
		Assert.That(output, Does.Contain("MethodCall: logger.Log"));
		Assert.That(output, Does.Contain("Return: count + 1"));
		Assert.That(output, Does.Contain("If: count > 0"));
		Assert.That(output, Does.Contain("For: items"));
		Assert.That(output, Does.Contain("Declaration: total = 0"));
		Assert.That(output, Does.Contain("Reassignment: total = total + value"));
	}

	[Test]
	public async Task RunFibonacci()
	{
		await new Runner(GetExamplesFilePath("Fibonacci")).Run();
		var output = consoleWriter.ToString();
		Assert.That(output, Does.Contain("Fibonacci(10) = 55"));
		Assert.That(output, Does.Contain("Fibonacci(5) = 5"));
	}

	[Test]
	public async Task RunSimpleCalculatorTwiceWithoutTestPackage()
	{
		await new Runner(SimpleCalculatorFilePath).Run();
		consoleWriter.GetStringBuilder().Clear();
		await new Runner(SimpleCalculatorFilePath).Run();
		Assert.That(consoleWriter.ToString(), Does.Contain("2 + 3 = 5"));
	}

	[Test]
	public async Task SaveStrictBinaryEntryNameTableSkipsPrefilledNames()
	{
		var tempDirectory = Path.Combine(Path.GetTempPath(), "Strict" + Guid.NewGuid().ToString("N"));
		Directory.CreateDirectory(tempDirectory);
		try
		{
			var sourceCopyPath = Path.Combine(tempDirectory, Path.GetFileName(SimpleCalculatorFilePath));
			File.Copy(SimpleCalculatorFilePath, sourceCopyPath);
			await new Runner(sourceCopyPath).Run();
			var binaryPath = Path.ChangeExtension(sourceCopyPath, BinaryExecutable.Extension);
			await using var archive = await ZipFile.OpenReadAsync(binaryPath);
			var entry = archive.Entries.First(file => file.FullName == "SimpleCalculator.bytecode");
			using var reader = new BinaryReader(await entry.OpenAsync());
			Assert.That(reader.ReadByte(), Is.EqualTo((byte)'S'));
			Assert.That(reader.ReadByte(), Is.EqualTo(BinaryType.Version));
			var customNamesCount = reader.Read7BitEncodedInt();
			var customNames = new List<string>(customNamesCount);
			for (var nameIndex = 0; nameIndex < customNamesCount; nameIndex++)
				customNames.Add(reader.ReadString());
			Assert.That(customNames, Does.Not.Contain("Strict/Number"));
			Assert.That(customNames, Does.Not.Contain("Strict/Text"));
			Assert.That(customNames, Does.Not.Contain("Strict/Boolean"));
			Assert.That(customNames, Does.Not.Contain("SimpleCalculator"));
		}
		finally
		{
			if (Directory.Exists(tempDirectory))
				Directory.Delete(tempDirectory, true);
		}
	}

	private static string SimpleCalculatorFilePath => GetExamplesFilePath("SimpleCalculator");
	private static string SumFilePath => GetExamplesFilePath("Sum");

	private async Task<string> GetExamplesBinaryFile(string filename)
	{
		var localPath = Path.ChangeExtension(GetExamplesFilePath(filename), BinaryExecutable.Extension);
		if (!File.Exists(localPath))
			await new Runner(GetExamplesFilePath(filename)).Run(); //ncrunch: no coverage
		consoleWriter.GetStringBuilder().Clear();
		return localPath;
	}

	public static string GetExamplesFilePath(string filename)
	{
		var localPath = Path.Combine(
			Repositories.GetLocalDevelopmentPath(Repositories.StrictOrg, nameof(Strict)), "Examples",
			filename + Type.Extension);
		return File.Exists(localPath)
			? localPath
			: Path.Combine(FindRepoRoot(), "Examples", filename + Type.Extension);
	}

	private static string FindRepoRoot()
	{
		var directory = Repositories.GetLocalDevelopmentPath(Repositories.StrictOrg, nameof(Strict));
		if (File.Exists(Path.Combine(directory, "Strict.sln")))
			return directory;
		directory = AppContext.BaseDirectory;
		while (directory != null)
		{
			if (File.Exists(Path.Combine(directory, "Strict.sln")))
				return directory;
			directory = Path.GetDirectoryName(directory);
		}
		throw new DirectoryNotFoundException("Cannot find repository root (Strict.sln not found)");
	}

	[Test]
	//[Category("Slow")]
	//TODO: works and helps finding issues, but is so annoyingly slow that NCrunch becomes stuck for 10-20s, no good! we first need to get things fast!
	public async Task RunAdjustBrightness()
	{
#if DEBUG
		try
		{
			PerformanceLog.IsEnabled = true;
#endif
			await new Runner(GetExamplesFilePath("../ImageProcessing/AdjustBrightness")).Run();
			var output = consoleWriter.ToString();
			Assert.That(output, Does.Contain("Brightness adjustment successful: (0.25, 0.25, 0.25)"));
#if DEBUG
		}
		finally
		{
			PerformanceLog.IsEnabled = false;
		}
#endif
	}

	[Test]
	[Category("Slow")] //TODO: still need to test this once optimizations are done, flatArrays!
	public async Task RunAdjustBrightnessAllocatesBelowHalfMegabytePerRun()
	{
		var runner = new Runner(GetExamplesFilePath("../ImageProcessing/AdjustBrightness"));
		ValueInstance.SetCreationLimit(int.MaxValue);
		try
		{
			await runner.Run();
			var allocatedBefore = GC.GetAllocatedBytesForCurrentThread();
			await runner.Run();
			var allocatedAfter = GC.GetAllocatedBytesForCurrentThread();
			const int Width = 128;
			const int Height = 72;
			// Allocation budget accounts for Color/Byte type overhead; the CompactType
			// optimizer will reduce this further once all ColorValue usages are converted
			Assert.That(allocatedAfter - allocatedBefore, Is.LessThan(Width * Height * 4 * 4 * 4));
		}
		finally
		{
			ValueInstance.SetCreationLimit(int.MaxValue);
		}
	}

	[Test]
	public void NativeImageRoundTripIsPixelIdenticalForPng()
	{
		var repoRoot = FindRepoRoot();
		var testImagePath = Path.Combine(repoRoot, "ImageProcessing", "4x4.png");
		var searchDirectory = AppContext.BaseDirectory;
		CopyNativePluginsToDirectory(repoRoot, searchDirectory);
		var originalBytes = NativePluginLoader.TryLoadNativeLifecycle("ImageLoader", testImagePath,
			searchDirectory, out var width, out var height);
		Assert.That(originalBytes, Is.Not.Null, "Plugin is missing at " + searchDirectory);
		Assert.That(width, Is.EqualTo(4));
		Assert.That(height, Is.EqualTo(4));
		Assert.That(originalBytes!.Length, Is.EqualTo(4 * 4 * 4));
		var outputPath = Path.Combine(repoRoot, "ImageProcessing", "4x4_output.png");
		NativePluginLoader.TrySaveNativeImage("ImageSaver", outputPath, originalBytes, width, height,
			searchDirectory);
		Assert.That(File.Exists(outputPath), Is.True);
		var reloadedBytes = NativePluginLoader.TryLoadNativeLifecycle("ImageLoader", outputPath,
			searchDirectory, out var reloadedWidth, out var reloadedHeight);
		Assert.That(reloadedBytes, Is.Not.Null);
		Assert.That(reloadedWidth, Is.EqualTo(width));
		Assert.That(reloadedHeight, Is.EqualTo(height));
		Assert.That(reloadedBytes, Is.EqualTo(originalBytes));
	}

	private static void CopyNativePluginsToDirectory(string repoRoot, string targetDirectory)
	{
		var extension = RuntimeInformation.IsOSPlatform(OSPlatform.Windows)
			? ".dll"
			: RuntimeInformation.IsOSPlatform(OSPlatform.OSX)
				? ".dylib"
				: ".so";
		var loaderSource =
			Path.Combine(repoRoot, "NativePlugins", "ImageLoader", "ImageLoader" + extension);
		var saverSource =
			Path.Combine(repoRoot, "NativePlugins", "ImageSaver", "ImageSaver" + extension);
		CopyIfNewerOrMissing(loaderSource, Path.Combine(targetDirectory, "ImageLoader" + extension));
		CopyIfNewerOrMissing(saverSource, Path.Combine(targetDirectory, "ImageSaver" + extension));
		if (extension != ".so")
		{
			CopyIfNewerOrMissing(loaderSource.Replace(extension, ".so"),
				Path.Combine(targetDirectory, "ImageLoader.so"));
			CopyIfNewerOrMissing(saverSource.Replace(extension, ".so"),
				Path.Combine(targetDirectory, "ImageSaver.so"));
		}
	}

	private static void CopyIfNewerOrMissing(string source, string target)
	{
		if (File.Exists(source) && (!File.Exists(target) ||
			File.GetLastWriteTimeUtc(source) > File.GetLastWriteTimeUtc(target)))
			File.Copy(source, target, true);
	}

	[Test]
	public async Task NativeImageLoadProcessSavePipeline()
	{
		var repoRoot = FindRepoRoot();
		var testImagePath = Path.Combine(repoRoot, "ImageProcessing", "test_image.jpg");
		CopyNativePluginsToDirectory(repoRoot, AppContext.BaseDirectory);
		var processImagePath =
			Path.Combine(repoRoot, "ImageProcessing", "ProcessImage" + Type.Extension);
		await new Runner(processImagePath, testImagePath).Run();
		await new Runner(processImagePath, testImagePath).Run();
		var outputImagePath = testImagePath.Replace(".jpg", "_output.jpg");
		Assert.That(File.Exists(outputImagePath), Is.True, outputImagePath);
		var output = consoleWriter.ToString();
		Assert.That(output, Does.Contain("Processed image saved to:"));
	}

	//ncrunch: no coverage start
	[Test]
	[Category("Slow")]
	public async Task RunAdjustBrightnessRegeneratesCachedBinaryWhenColorChanges()
	{
		var adjustBrightnessPath = GetExamplesFilePath("../ImageProcessing/AdjustBrightness");
		var colorPath = Path.Combine(Path.GetDirectoryName(adjustBrightnessPath)!, "Color.strict");
		var binaryPath = Path.ChangeExtension(adjustBrightnessPath, BinaryExecutable.Extension);
		var originalColorCode = await File.ReadAllTextAsync(colorPath);
		var originalColorTimestamp = File.GetLastWriteTimeUtc(colorPath);
		var hadBinary = File.Exists(binaryPath);
		var originalBinaryBytes = hadBinary
			? await File.ReadAllBytesAsync(binaryPath)
			: [];
		var originalBinaryTimestamp = hadBinary
			? File.GetLastWriteTimeUtc(binaryPath)
			: DateTime.MinValue;
		try
		{
			if (File.Exists(binaryPath))
				File.Delete(binaryPath);
			await new Runner(adjustBrightnessPath).Run();
			var firstBinaryTimestamp = File.GetLastWriteTimeUtc(binaryPath);
			consoleWriter.GetStringBuilder().Clear();
			await File.WriteAllTextAsync(colorPath,
				originalColorCode.Replace("has Alpha = 1", "has Alpha = 0.5"));
			File.SetLastWriteTimeUtc(colorPath, DateTime.UtcNow.AddSeconds(2));
			await new Runner(adjustBrightnessPath).Run();
			Assert.That(File.GetLastWriteTimeUtc(binaryPath), Is.GreaterThan(firstBinaryTimestamp));
		}
		finally
		{
			await File.WriteAllTextAsync(colorPath, originalColorCode);
			File.SetLastWriteTimeUtc(colorPath, originalColorTimestamp);
			if (hadBinary)
			{
				await File.WriteAllBytesAsync(binaryPath, originalBinaryBytes);
				File.SetLastWriteTimeUtc(binaryPath, originalBinaryTimestamp);
			}
			else if (File.Exists(binaryPath))
			{
				File.Delete(binaryPath);
			}
		}
	}
}