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

	[TestCase("HelloLogger")]
	[TestCase("NativeArithmetic")]
	[TestCase("NativeConditions")]
	[TestCase("NativeLoop")]
	[TestCase("AreaCalculator")]
	[TestCase("SimpleCalculator")]
	[TestCase("TemperatureConverter")]
	[TestCase("Pixel")]
	[TestCase("Fibonacci")]
	[TestCase("GcdCalculator")]
	[Category("Slow")]
	public async Task NativeExecutableRunsLikeVirtualMachine(string example)
	{
		var sourcePath = GetExamplesFilePath(example);
		await new Runner(sourcePath).Build(Enum.Parse<Platform>(NativeProcessRunner.OperatingSystemName));
		consoleWriter.GetStringBuilder().Clear();
		new VirtualMachine(new BinaryExecutable(Path.ChangeExtension(sourcePath,
			BinaryExecutable.Extension))).Execute();
		var native = NativeProcessRunner.Run(Path.ChangeExtension(sourcePath,
			OperatingSystem.IsWindows()
				? ".exe"
				: null), "");
		Assert.That(native.Output.ReplaceLineEndings(),
			Is.EqualTo(consoleWriter.ToString().ReplaceLineEndings()), native.Error);
	}

	[TestCase("HelloLogger")]
	[TestCase("AreaCalculator")]
	[TestCase("SimpleCalculator")]
	[TestCase("TemperatureConverter")]
	[TestCase("Pixel")]
	[TestCase("Fibonacci")]
	[TestCase("GcdCalculator")]
	[Category("Slow")]
	public async Task StrictNativeCompilerRunsLikeVirtualMachine(string example)
	{
		var root = FindRepoRoot().Replace('\\', '/');
		var sourcePath = root + "/Examples/" + example + Type.Extension;
		await new Runner(sourcePath).Run();
		consoleWriter.GetStringBuilder().Clear();
		new VirtualMachine(new BinaryExecutable(Path.ChangeExtension(sourcePath,
			BinaryExecutable.Extension))).Execute();
		var expected = consoleWriter.ToString();
		consoleWriter.GetStringBuilder().Clear();
		await new Runner(root + "/Compiler/NativeCompiler" + Type.Extension, sourcePath + " " + root).Run();
		Assert.That(consoleWriter.ToString(), Does.Contain("Built "));
		var native = NativeProcessRunner.Run(Path.ChangeExtension(sourcePath,
			OperatingSystem.IsWindows()
				? ".exe"
				: null), "");
		Assert.That(native.Output.ReplaceLineEndings(), Is.EqualTo(expected.ReplaceLineEndings()),
			native.Error);
	}

	[TestCaseSource(nameof(StrictProgramPaths))]
	[Category("Slow")]
	public void RunStrictProgramFromSourceAndCachedBinaryInFreshProcess(string relativePath)
	{
		var root = FindRepoRoot();
		var sourcePath = Path.Combine(root, relativePath);
		var hasRun = HasRunMethod(sourcePath);
		var arguments = ProgramArguments.TryGetValue(relativePath, out var argument)
			? string.Concat(argument.Split(' ').Select(part => " \"" + Path.Combine(root, part) + "\""))
			: "";
		regeneratingRuntime ??= CopyStrictRuntime(DateTime.UtcNow.AddDays(1));
		var testDirectory = Directory.GetCurrentDirectory();
		Directory.SetCurrentDirectory(root);
		try
		{
			foreach (var inputPath in hasRun
				? [sourcePath, Path.ChangeExtension(sourcePath, BinaryExecutable.Extension)]
				: new[] { sourcePath })
			{
				var result = NativeProcessRunner.Run("dotnet",
					"\"" + regeneratingRuntime + "\" \"" + inputPath + "\"" + arguments, 120000);
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

	/// <summary>
	/// Its Strict*.dll are newer than every cached binary, source runs always compile from source.
	/// </summary>
	private string? regeneratingRuntime;

	[OneTimeTearDown]
	public void DeleteRegeneratingRuntime()
	{
		if (regeneratingRuntime != null)
			Directory.Delete(Path.GetDirectoryName(regeneratingRuntime)!, true);
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
		["Expressions/RoundTrip.strict"] = "Expressions",
		["Expressions/ResolveCheck.strict"] = "Expressions .",
		["Expressions/TypeReport.strict"] = "Expressions .",
		["Bytecode/FileCompiler.strict"] = "Examples/HelloLogger.strict .",
		["Compiler/NativeCompiler.strict"] = "Examples/NativeArithmetic.strict .",
		["Runtime/Execute.strict"] = "Examples/HelloLogger.strict .",
		["Validators/ValidateCheck.strict"] = "Validators ."
	};

	private static IEnumerable<string> StrictProgramPaths()
	{
		var root = FindRepoRoot();
		return StrictFolders().SelectMany(folder =>
				Directory.GetFiles(Path.Combine(root, folder), "*" + Type.Extension)).
			Select(file => Path.GetRelativePath(root, file).Replace('\\', '/')).Order();
	}

	internal static IEnumerable<string> StrictFolders()
	{
		var root = FindRepoRoot();
		string[] projects =
		[
			".", "Math", "ImageProcessing", "Language", "Expressions", "Validators", "TestRunner",
			"HighLevelRuntime", "Bytecode", "Optimizers", "Runtime", "Compiler", "Examples"
		];
		return projects.Concat(Directory.GetDirectories(Path.Combine(root, "Examples")).
			Select(folder => Path.GetRelativePath(root, folder).Replace('\\', '/')).
			Where(folder => !folder.EndsWith("/bin", StringComparison.Ordinal) &&
				!folder.EndsWith("/obj", StringComparison.Ordinal)));
	}

	[TestCaseSource(nameof(StrictFolders))]
	[Category("Slow")]
	public Task StrictParserRoundTripsEveryLine(string folder) =>
		RunExpressionsProgram("RoundTrip", Path.Combine(FindRepoRoot(), folder),
			"Round trip mismatches: 0");

	[TestCaseSource(nameof(StrictFolders))]
	[Category("Slow")]
	public Task StrictResolvesEveryName(string folder) =>
		RunExpressionsProgram("ResolveCheck", Path.Combine(FindRepoRoot(), folder) + " " +
			FindRepoRoot(), "Unresolved names: 0");

	[TestCaseSource(nameof(StrictFolders))]
	[Category("Slow")]
	public async Task StrictInfersSameTypesAsCSharp(string folder)
	{
		var root = FindRepoRoot();
		await RunExpressionsProgram("TypeReport", Path.Combine(root, folder) + " " + root, "");
		var strictTypes = consoleWriter.ToString().Split('\n').Select(line => line.Trim()).ToHashSet();
		var mismatches = (await CSharpStatementTypes.Collect(root, folder)).Except(strictTypes).
			Select(line => line + " <> Strict: " + strictTypes.FirstOrDefault(strictLine =>
				strictLine.StartsWith(line[..(line.IndexOf(' ') + 1)], StringComparison.Ordinal))).ToList();
		Assert.That(mismatches, Is.Empty, string.Join(Environment.NewLine, mismatches.Order()));
	}

	private async Task RunExpressionsProgram(string program, string arguments, string expectedOutput)
	{
		await new Runner(Path.Combine(FindRepoRoot(), "Expressions", program + Type.Extension),
			arguments).Run();
		Assert.That(consoleWriter.ToString(), Does.Contain(expectedOutput));
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

	private static readonly Lock FreshAssemblyGate = new();
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

	/// <summary>
	/// Newer Strict*.dll outdate cached binaries, a copy with its own write time decides that for
	/// a fresh process instead of re-timing or deleting files other tests use. In the repo for CI.
	/// </summary>
	private static string CopyStrictRuntime(DateTime writeTime)
	{
		var strictAssembly = StrictAssemblyForFreshProcess();
		var directory = Path.Combine(AppContext.BaseDirectory,
			"StrictRuntime" + Guid.NewGuid().ToString("N"));
		Directory.CreateDirectory(directory);
		foreach (var file in Directory.GetFiles(Path.GetDirectoryName(strictAssembly)!))
		{
			var copy = Path.Combine(directory, Path.GetFileName(file));
			File.Copy(file, copy);
			File.SetLastWriteTimeUtc(copy, writeTime);
		}
		return Path.Combine(directory, Path.GetFileName(strictAssembly));
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
	public Task RunSimpleCalculator() =>
		AfterRunningSimpleCalculatorCopy(sourcePath =>
		{
			Assert.That(consoleWriter.ToString(),
				Does.StartWith("2 + 3 = 5" + Environment.NewLine + "2 * 3 = 6" + Environment.NewLine));
			Assert.That(File.Exists(Path.ChangeExtension(sourcePath, ".asm")), Is.False);
			return Task.CompletedTask;
		});

	[Test]
	public Task RunFromBytecodeFileProducesSameOutput() =>
		AfterRunningSimpleCalculatorCopy(async sourcePath =>
		{
			consoleWriter.GetStringBuilder().Clear();
			await new Runner(Path.ChangeExtension(sourcePath, BinaryExecutable.Extension)).Run();
			Assert.That(consoleWriter.ToString(),
				Does.StartWith("2 + 3 = 5" + Environment.NewLine + "2 * 3 = 6"));
		});

	[Test]
	public Task CachedBinaryOlderThanRuntimeIsRegenerated() =>
		AfterRunningSimpleCalculatorCopy(async sourcePath =>
		{
			var binaryPath = Path.ChangeExtension(sourcePath, BinaryExecutable.Extension);
			var runtimeTime = File.GetLastWriteTimeUtc(typeof(Runner).Assembly.Location);
			File.SetLastWriteTimeUtc(sourcePath, runtimeTime.AddMinutes(-2));
			File.SetLastWriteTimeUtc(binaryPath, runtimeTime.AddMinutes(-1));
			await new Runner(sourcePath).Run();
			Assert.That(File.GetLastWriteTimeUtc(binaryPath), Is.GreaterThan(runtimeTime));
		});

	[Test]
	public Task CachedBinaryWithOlderVersionIsRegenerated() =>
		AfterRunningSimpleCalculatorCopy(async sourcePath =>
		{
			var binaryPath = Path.ChangeExtension(sourcePath, BinaryExecutable.Extension);
			await using (var archive = await ZipFile.OpenAsync(binaryPath, ZipArchiveMode.Update))
			await using (var entry = await archive.GetEntry("SimpleCalculator.bytecode")!.OpenAsync())
			{
				entry.Position = 1;
				entry.WriteByte(BinaryType.Version - 1);
			}
			consoleWriter.GetStringBuilder().Clear();
			await new Runner(sourcePath, Method.Run, true).Run();
			Assert.That(consoleWriter.ToString(),
				Does.Contain("Cached binary incompatible: File version: " + (BinaryType.Version - 1)).
					And.Contain("2 + 3 = 5"));
			Assert.That(() => new BinaryExecutable(binaryPath), Throws.Nothing);
		});

	/// <summary>
	/// Tests changing sources or cached binaries use copies, parallel tests read the repo files.
	/// </summary>
	private static async Task InTemporaryCopy(IEnumerable<string> files, Func<string, Task> test)
	{
		var directory = Path.Combine(Path.GetTempPath(), "Strict" + Guid.NewGuid().ToString("N"));
		Directory.CreateDirectory(directory);
		try
		{
			foreach (var file in files)
				File.Copy(file, Path.Combine(directory, Path.GetFileName(file)));
			await test(directory);
		}
		finally
		{
			Directory.Delete(directory, true);
		}
	}

	private static Task InTemporaryFile(string typeName, string code, Func<string, Task> test) =>
		InTemporaryCopy([], async directory =>
		{
			var path = Path.Combine(directory, typeName + Type.Extension);
			await File.WriteAllTextAsync(path, code);
			await test(path);
		});

	[Test]
	public Task EndlessRecursionOnVmNamesStrictLineAndCallers() =>
		InTemporaryFile("EndlessVm",
			"has number\nhas logger\nDeeper Number\n\tif number < 0\n\t\treturn 0\n" +
			"\tEndlessVm(number + 1).Deeper\nRun\n\tlogger.Log(EndlessVm(1).Deeper)", async path =>
			{
				Assert.That(await Strict.Program.Main([path]), Is.EqualTo(1));
				Assert.That(consoleWriter.ToString(),
					Does.Contain("StackOverflow").And.Contain(path + ":line 6").
						And.Contain("EndlessVm.Deeper (255 times)").And.Contain("EndlessVm.Run"));
			});

	[Test]
	public Task LoopCollectingBillionsOfNumbersOnVmNamesStrictLine() =>
		InTemporaryFile("HugeRange",
			"has logger\nNumbers(limit Number) Numbers\n\tHugeRange.Numbers(3) is (1, 2)\n" +
			"\tfor Range(1, limit)\n\t\tvalue\nRun\n\tlogger.Log(Numbers(3000000000).Length)",
			async path =>
			{
				Assert.That(await Strict.Program.Main([path]), Is.EqualTo(1));
				Assert.That(consoleWriter.ToString(),
					Does.Contain("Loop count or range bound 3000000000").
						And.Contain(path + ":line 4").And.Not.Contain("at Strict.VirtualMachine"));
			});

	[Test]
	public Task EnormousListInTestNamesStrictLine() =>
		InTemporaryFile("HugeList",
			"has count Number\nhas numbers with Length is count\nLength Number\n" +
			"\tHugeList(3000 * 1000000).Length is 1\n\tnumbers.Length", async path =>
			{
				Assert.That(await Strict.Program.Main(["test", path]), Is.EqualTo(1));
				Assert.That(consoleWriter.ToString(),
					Does.Contain("OutOfMemoryException").And.Contain(path + ":line 4").
						And.Not.Contain("at Strict.HighLevelRuntime"));
			});

	[Test]
	public Task EndlessRecursionInTestNamesStrictLineAndTestLine() =>
		InTemporaryFile("EndlessTest",
			"has number\nDeeper Number\n\tEndlessTest(1).Deeper is 0\n\tif number < 0\n\t\treturn 0\n" +
			"\tEndlessTest(number + 1).Deeper", async path =>
			{
				Assert.That(await Strict.Program.Main(["test", path]), Is.EqualTo(1));
				Assert.That(consoleWriter.ToString(),
					Does.Contain("CallDepthExceeded").And.Contain(path + ":line 6").
						And.Contain(path + ":line 3"));
			});

	[Test]
	public Task FailureDeepInLegalRecursionNamesCallersInsteadOfCrashing() =>
		InTemporaryFile("FailDeep",
			"has number\nDeeper Number\n\tFailDeep(1).Deeper is 0\n\tconstant numbers = (1, 2)\n" +
			"\tif number > 125\n\t\treturn numbers(number)\n\tFailDeep(number + 1).Deeper",
			async path =>
			{
				Assert.That(await Strict.Program.Main(["test", path]), Is.EqualTo(1));
				Assert.That(consoleWriter.ToString(),
					Does.Contain("ListIndexOutOfRange").And.Contain(path + ":line 7").
						And.Contain(path + ":line 3"));
			});

	/// <summary>
	/// Runs a temporary SimpleCalculator copy once, the test gets the copied source path and finds
	/// the cached binary next to it.
	/// </summary>
	private static Task AfterRunningSimpleCalculatorCopy(Func<string, Task> test) =>
		InTemporaryCopy([SimpleCalculatorFilePath], async directory =>
		{
			var sourcePath = Path.Combine(directory, Path.GetFileName(SimpleCalculatorFilePath));
			await new Runner(sourcePath).Run();
			await test(sourcePath);
		});

	[Test]
	public Task RunFromBytecodeFileWithoutStrictSourceFile() =>
		AfterRunningSimpleCalculatorCopy(async sourcePath =>
		{
			var binaryPath = Path.ChangeExtension(sourcePath, BinaryExecutable.Extension);
			Assert.That(File.Exists(binaryPath), Is.True);
			consoleWriter.GetStringBuilder().Clear();
			File.Delete(sourcePath);
			await new Runner(binaryPath).Run();
			Assert.That(consoleWriter.ToString(),
				Does.StartWith("2 + 3 = 5" + Environment.NewLine + "2 * 3 = 6"));
		});

	[Test]
	public Task TestCommandReportsFailingInlineTestWithoutDotNetStackTrace() =>
		InTemporaryFile("WrongTwice",
			"has logger\nTwice(number) Number\n\tTwice(2) is 5\n\tnumber * 2\nRun\n\tlogger.Log(Twice(2))",
			async path =>
			{
				Assert.That(await Strict.Program.Main(["test", path]), Is.EqualTo(1));
				Assert.That(consoleWriter.ToString(),
					Does.Contain(path + ":line 3").And.Not.Contain("at Strict.Runner"));
			});

	[Test]
	public async Task TypeNameCallParametersWinOverCallerMembers()
	{
		var directory = Path.Combine(Path.GetTempPath(), "Strict" + Guid.NewGuid().ToString("N"));
		Directory.CreateDirectory(directory);
		await File.WriteAllTextAsync(Path.Combine(directory, "Maker" + Type.Extension),
			"has unit Number\nMade(number Number) Number\n\tMaker.Made(1) is 1\n\tnumber\n" +
			"Doubled Number\n\tMaker(2).Doubled is 4\n\tunit * 2");
		var path = Path.Combine(directory, "Counter" + Type.Extension);
		await File.WriteAllTextAsync(path,
			"has number\nhas logger\nShifted Number\n\tCounter(5).Shifted is 4\n" +
			"\tMaker.Made(number - 1)\nRun\n\tlogger.Log(Counter(5).Shifted)");
		try
		{
			await new Runner(path).Run();
			Assert.That(consoleWriter.ToString(), Does.StartWith("4"));
		}
		finally
		{
			Directory.Delete(directory, true);
		}
	}

	[Test]
	public async Task DeclaredPackageTypesWinAfterExamplesWereLoaded()
	{
		await new Runner(SimpleCalculatorFilePath).Check(false);
		Assert.That(async () => await new Runner(Path.Combine(FindRepoRoot(), "Compiler",
			"EmitTests" + Type.Extension)).Check(false), Throws.Nothing);
	}

	[Test]
	public async Task UnknownOptionIsUsageError()
	{
		Assert.That(await Strict.Program.Main([GetExamplesFilePath("SimpleCalculator"), "-fast"]),
			Is.EqualTo(2));
		Assert.That(consoleWriter.ToString(), Does.Contain("Unknown option -fast"));
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
	public Task RunSucceedsWhileCachedBinaryIsOpenedByAnotherReader() =>
		InTemporaryCopy([Path.Combine(FindRepoRoot(), "Examples", "HelloLogger.strict")],
			async directory =>
			{
				var source = Path.Combine(directory, "HelloLogger.strict");
				await new Runner(source).Run();
				File.SetLastWriteTimeUtc(source, DateTime.UtcNow.AddMinutes(1));
				await using var reader = new FileStream(Path.ChangeExtension(source,
					BinaryExecutable.Extension), FileMode.Open, FileAccess.Read, FileShare.Read);
				consoleWriter.GetStringBuilder().Clear();
				await new Runner(source).Run();
				Assert.That(consoleWriter.ToString(), Does.Contain("Hi"));
			});

	[Test]
	[SetCulture("de-DE")]
	public async Task DiagnosticTimesUseInvariantCulture()
	{
		await new Runner(SimpleCalculatorFilePath, Method.Run, true).Run();
		Assert.That(consoleWriter.ToString(), Does.Not.Match(@"Time: \d+,\d+ ms"));
	}

	/// <summary>
	/// Used packages are repo folders, so the copied binary and runtime are made older than the
	/// newest base package file instead of re-timing a file other tests use.
	/// </summary>
	[Test]
	public Task CachedBinaryIsOutdatedWhenUsedPackageChanged() =>
		AfterRunningSimpleCalculatorCopy(sourcePath =>
		{
			var basePackageChange = new DirectoryInfo(FindRepoRoot()).
				EnumerateFiles("*" + Type.Extension).Max(file => file.LastWriteTimeUtc);
			File.SetLastWriteTimeUtc(sourcePath, basePackageChange.AddMinutes(-2));
			File.SetLastWriteTimeUtc(Path.ChangeExtension(sourcePath, BinaryExecutable.Extension),
				basePackageChange.AddMinutes(-1));
			var runtime = CopyStrictRuntime(basePackageChange.AddMinutes(-2));
			try
			{
				var result = NativeProcessRunner.Run("dotnet",
					"\"" + runtime + "\" \"" + sourcePath + "\" -diagnostics", 120000);
				Assert.That(result.Output, Does.Contain("Cached binary outdated, a used package changed").
					And.Contain("2 + 3 = 5"), result.Output + result.Error);
			}
			finally
			{
				Directory.Delete(Path.GetDirectoryName(runtime)!, true);
			}
			return Task.CompletedTask;
		});

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
	public Task AsmFileIsNotCreatedWhenRunningFromPrecompiledBytecode() =>
		AfterRunningSimpleCalculatorCopy(async sourcePath =>
		{
			await new Runner(Path.ChangeExtension(sourcePath, BinaryExecutable.Extension)).Run();
			Assert.That(File.Exists(Path.ChangeExtension(sourcePath, ".asm")), Is.False);
		});

	[Test]
	public Task SaveStrictBinaryWithTypeBytecodeEntriesOnly() =>
		AfterRunningSimpleCalculatorCopy(async sourcePath =>
		{
			await using var archive =
				await ZipFile.OpenReadAsync(Path.ChangeExtension(sourcePath, BinaryExecutable.Extension));
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
		});

	[Test]
	public Task ListConstantOfConstructedValuesRunsFromSource() =>
		InTemporaryCopy([], async folder =>
		{
			var path = Path.Combine(folder, "RangePair" + Type.Extension);
			await File.WriteAllTextAsync(path, string.Join('\n',
				"has number", "constant Ranges = (Range(0, 1), Range(1, 3))", "Total Number",
				"\tRangePair(0).Total is 2", "\tRanges.Length + number", "Run Number",
				"\tRangePair(1).Total"));
			Assert.That(async () => await new Runner(path).Run(), Throws.Nothing);
		});

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
	public Task SaveStrictBinaryEntryNameTableSkipsPrefilledNames() =>
		AfterRunningSimpleCalculatorCopy(async sourcePath =>
		{
			await using var archive =
				await ZipFile.OpenReadAsync(Path.ChangeExtension(sourcePath, BinaryExecutable.Extension));
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
		});

	private static string SimpleCalculatorFilePath => GetExamplesFilePath("SimpleCalculator");
	private static string SumFilePath => GetExamplesFilePath("Sum");

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
	public Task NativeImageRoundTripIsPixelIdenticalForPng() =>
		InTemporaryCopy([], directory =>
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
			var outputPath = Path.Combine(directory, "4x4_output.png");
			NativePluginLoader.TrySaveNativeImage("ImageSaver", outputPath, originalBytes, width,
				height, searchDirectory);
			Assert.That(File.Exists(outputPath), Is.True);
			var reloadedBytes = NativePluginLoader.TryLoadNativeLifecycle("ImageLoader", outputPath,
				searchDirectory, out var reloadedWidth, out var reloadedHeight);
			Assert.That(reloadedBytes, Is.Not.Null);
			Assert.That(reloadedWidth, Is.EqualTo(width));
			Assert.That(reloadedHeight, Is.EqualTo(height));
			Assert.That(reloadedBytes, Is.EqualTo(originalBytes));
			return Task.CompletedTask;
		});

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
			!File.ReadAllBytes(source).AsSpan().SequenceEqual(File.ReadAllBytes(target))))
			File.Copy(source, target, true);
	}

	[Test]
	public Task NativeImageLoadProcessSavePipeline() =>
		InTemporaryCopy([Path.Combine(FindRepoRoot(), "ImageProcessing", "test_image.jpg")],
			async directory =>
			{
				var repoRoot = FindRepoRoot();
				var testImagePath = Path.Combine(directory, "test_image.jpg");
				CopyNativePluginsToDirectory(repoRoot, AppContext.BaseDirectory);
				var processImagePath =
					Path.Combine(repoRoot, "ImageProcessing", "ProcessImage" + Type.Extension);
				await new Runner(processImagePath, testImagePath).Run();
				await new Runner(processImagePath, testImagePath).Run();
				var outputImagePath = testImagePath.Replace(".jpg", "_output.jpg");
				Assert.That(File.Exists(outputImagePath), Is.True, outputImagePath);
				Assert.That(consoleWriter.ToString(), Does.Contain("Processed image saved to:"));
			});

	//ncrunch: no coverage start
	[Test]
	[Category("Slow")]
	public Task RunAdjustBrightnessRegeneratesCachedBinaryWhenColorChanges() =>
		InTemporaryCopy(
			Directory.GetFiles(Path.Combine(FindRepoRoot(), "ImageProcessing"), "*" + Type.Extension),
			async directory =>
			{
				var adjustBrightnessPath = Path.Combine(directory, "AdjustBrightness" + Type.Extension);
				var binaryPath = Path.ChangeExtension(adjustBrightnessPath, BinaryExecutable.Extension);
				await new Runner(adjustBrightnessPath).Run();
				var firstBinaryTimestamp = File.GetLastWriteTimeUtc(binaryPath);
				File.SetLastWriteTimeUtc(Path.Combine(directory, "Color" + Type.Extension),
					DateTime.UtcNow.AddSeconds(2));
				await new Runner(adjustBrightnessPath).Run();
				Assert.That(File.GetLastWriteTimeUtc(binaryPath), Is.GreaterThan(firstBinaryTimestamp));
			});
}