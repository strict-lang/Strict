using Strict.Bytecode;
using Strict.Bytecode.Serialization;
using Strict.Compiler;
using Strict.Expressions;
using Strict.Language;
using Strict.Language.Tests;
using System.IO.Compression;

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

	[TearDown]
	public void RestoreConsole() => Console.SetOut(rememberConsole);

	[Test]
	public async Task RunSimpleCalculator()
	{
		var asmFilePath = Path.ChangeExtension(SimpleCalculatorFilePath, ".asm");
		if (File.Exists(asmFilePath))
			File.Delete(asmFilePath); //ncrunch: no coverage
		try
		{
			await new Runner(SimpleCalculatorFilePath, TestPackage.Instance).Run();
		}
		//ncrunch: no coverage start
		catch (IOException)
		{
			// Try again if the file was used in another test
			Thread.Sleep(100);
			await new Runner(SimpleCalculatorFilePath, TestPackage.Instance).Run();
		} //ncrunch: no coverage end
		Assert.That(consoleWriter.ToString(),
			Does.StartWith("2 + 3 = 5" + Environment.NewLine + "2 * 3 = 6" + Environment.NewLine));
		Assert.That(File.Exists(asmFilePath), Is.False);
	}

	[Test]
	public async Task RunFromBytecodeFileProducesSameOutput()
	{
		var binaryFilePath = await GetExamplesBinaryFile("SimpleCalculator");
		await new Runner(binaryFilePath, TestPackage.Instance).Run();
		Assert.That(consoleWriter.ToString(),
			Does.StartWith("2 + 3 = 5" + Environment.NewLine + "2 * 3 = 6"));
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
			await new Runner(copiedSourceFilePath, TestPackage.Instance).Run();
			Assert.That(File.Exists(copiedBinaryFilePath), Is.True);
			consoleWriter.GetStringBuilder().Clear();
			File.Delete(copiedSourceFilePath);
			await new Runner(copiedBinaryFilePath, TestPackage.Instance).Run();
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
		var runner = new Runner(SimpleCalculatorFilePath, TestPackage.Instance, "(1, 2, 3).Length");
		Assert.That(async () => await runner.Build(Platform.Windows),
			Throws.TypeOf<Runner.CannotBuildExecutableWithCustomExpression>());
	}

	[Test]
	public async Task AsmFileIsNotCreatedWhenRunningFromPrecompiledBytecode()
	{
		var asmPath = Path.ChangeExtension(SimpleCalculatorFilePath, ".asm");
		if (File.Exists(asmPath))
			File.Delete(asmPath); //ncrunch: no coverage
		var binaryPath = await GetExamplesBinaryFile("SimpleCalculator");
		await new Runner(binaryPath, TestPackage.Instance).Run();
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
		await new Runner(SumFilePath, TestPackage.Instance, "5 10 20").Run();
		Assert.That(consoleWriter.ToString(), Does.Contain("35"));
	}

	[Test]
	public async Task RunSumWithDifferentProgramArgumentsDoesNotReuseCachedEntryPoint()
	{
		await new Runner(SumFilePath, TestPackage.Instance, "5 10 20").Run();
		consoleWriter.GetStringBuilder().Clear();
		await new Runner(SumFilePath, TestPackage.Instance, "1 2").Run();
		Assert.That(consoleWriter.ToString(), Does.Contain("3"));
	}

	[Test]
	public async Task RunSumWithNoArgumentsUsesEmptyList()
	{
		await new Runner(SumFilePath, TestPackage.Instance, "0").Run();
		Assert.That(consoleWriter.ToString(), Does.Contain("0"));
	}

	[Test]
	public async Task RunAutofilledMutable() =>
		await new Runner(GetExamplesFilePath("AutofilledMutable"), TestPackage.Instance).Run();

	[Test]
	public async Task RunParseHelloLogger()
	{
		await new Runner(GetExamplesFilePath("Parsing/ParseHelloLogger"), TestPackage.Instance).Run();
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
		await new Runner(GetExamplesFilePath("Parsing/ParseExpressions"), TestPackage.Instance).Run();
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
		await new Runner(GetExamplesFilePath("Parsing/ParseMethodHeaders"), TestPackage.Instance).Run();
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
		await new Runner(GetExamplesFilePath("Fibonacci"), TestPackage.Instance).Run();
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
			var sourceCopyPath =
				Path.Combine(tempDirectory, Path.GetFileName(SimpleCalculatorFilePath));
			File.Copy(SimpleCalculatorFilePath, sourceCopyPath);
			await new Runner(sourceCopyPath, TestPackage.Instance).Run();
			var binaryPath = Path.ChangeExtension(sourceCopyPath, BinaryExecutable.Extension);
			await using var archive = await ZipFile.OpenReadAsync(binaryPath);
			var entry = archive.Entries.First(file => file.FullName == "SimpleCalculator.bytecode");
			using var reader = new BinaryReader(entry.Open());
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
			await new Runner(GetExamplesFilePath(filename), TestPackage.Instance).Run(); //ncrunch: no coverage
		consoleWriter.GetStringBuilder().Clear();
		return localPath;
	}

	public static string GetExamplesFilePath(string filename)
	{
		var localPath = Path.Combine(
			Repositories.GetLocalDevelopmentPath(Repositories.StrictOrg, nameof(Strict)), "Examples",
			filename + Language.Type.Extension);
		return File.Exists(localPath)
			? localPath
			: Path.Combine(FindRepoRoot(), "Examples", filename + Language.Type.Extension);
	}

	[Test]
	public async Task RunAdjustBrightness()
	{
		//TODO: the for loop is wrong in AdjustBrightness.strict, it should be just: for image.Size
		await new Runner(GetExamplesFilePath("../ImageProcessing/AdjustBrightness"),
			await new Repositories(new MethodExpressionParser()).LoadStrictPackage()).Run();
		var output = consoleWriter.ToString();
		Assert.That(output, Does.Contain("Brightness adjustment successful: (0.25, 0.25, 0.25)"));
	}

	[Test]
	public async Task RunAdjustBrightnessAllocatesBelowHalfMegabytePerRun()
	{
		var strictBasePackage = await new Repositories(new MethodExpressionParser()).LoadStrictPackage();
		var runner = new Runner(GetExamplesFilePath("../ImageProcessing/AdjustBrightness"),
			strictBasePackage);
		ValueInstance.SetCreationLimit(11);
		try
		{
			var allocatedBefore = GC.GetAllocatedBytesForCurrentThread();
			await runner.Run();
			var allocatedAfter = GC.GetAllocatedBytesForCurrentThread();
			const int Width = 128;
			const int Height = 72;
			Assert.That(allocatedAfter - allocatedBefore, Is.LessThan(Width * Height * 4 * 4 + 50_000));
		}
		finally
		{
			ValueInstance.SetCreationLimit(int.MaxValue);
		}
	}

	[Test]
	public async Task RunAdjustBrightnessThrowsWhenValueInstanceCreationLimitIsExceeded()
	{
		var strictBasePackage = await new Repositories(new MethodExpressionParser()).LoadStrictPackage();
		var runner = new Runner(GetExamplesFilePath("../ImageProcessing/AdjustBrightness"),
			strictBasePackage);
		ValueInstance.SetCreationLimit(10);
		try
		{
			Assert.That(async () => await runner.Run(),
				Throws.TypeOf<ValueInstance.CreationLimitExceeded>());
		}
		finally
		{
			ValueInstance.SetCreationLimit(int.MaxValue);
		}
	}

	//ncrunch: no coverage start
	private static string FindRepoRoot()
	{
		var directory = AppContext.BaseDirectory;
		while (directory != null)
		{
			if (File.Exists(Path.Combine(directory, "Strict.sln")))
				return directory;
			directory = Path.GetDirectoryName(directory);
		}
		throw new DirectoryNotFoundException("Cannot find repository root (Strict.sln not found)");
	}
}