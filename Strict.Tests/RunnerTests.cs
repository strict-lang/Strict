using Strict.Bytecode.Serialization;
using Strict.Compiler;
using Strict.Compiler.Assembly;
using Strict.Language;
using Strict.Language.Tests;

namespace Strict.Tests;

public sealed class RunnerTests
{
	[SetUp]
	public void CreateTextWriter()
	{
		writer = new StringWriter();
		rememberConsole = Console.Out;
		Console.SetOut(writer);
	}

	private StringWriter writer = null!;
	private TextWriter rememberConsole = null!;

	[TearDown]
	public void RestoreConsole() => Console.SetOut(rememberConsole);

	[Test]
	public void RunSimpleCalculator()
	{
		using var _ = new Runner(TestPackage.Instance, SimpleCalculatorFilePath).Run();
		Assert.That(writer.ToString(),
			Does.StartWith("2 + 3 = 5" + Environment.NewLine + "2 * 3 = 6"));
	}

	[Test]
	public void RunBaseTypesTestPackageFromDirectory()
	{
		using var _ = new Runner(TestPackage.Instance, GetExamplesDirectoryPath("BaseTypesTest")).Run();
		var output = writer.ToString();
		Assert.That(output, Does.Contain("Hello, World!"));
		Assert.That(output, Does.Contain("Hello, Strict!"));
		Assert.That(output, Does.Contain("3 + 4 = 7"));
		Assert.That(output, Does.Contain("10 * 3 = 30"));
		Assert.That(output, Does.Contain("(1, 2, 3).Sum = 6"));
	}

	private string SimpleCalculatorFilePath => GetExamplesFilePath("SimpleCalculator");

	public static string GetExamplesFilePath(string filename)
	{
		var localPath = Path.Combine(
			Repositories.GetLocalDevelopmentPath(Repositories.StrictOrg, nameof(Strict)),
			"Examples", filename + Language.Type.Extension);
		return File.Exists(localPath)
			? localPath
			: Path.Combine(FindRepoRoot(), "Examples", filename + Language.Type.Extension);
	}

	public static string GetExamplesDirectoryPath(string directoryName)
	{
		var localPath = Path.Combine(
			Repositories.GetLocalDevelopmentPath(Repositories.StrictOrg, nameof(Strict)),
			"Examples", directoryName);
		return Directory.Exists(localPath)
			? localPath
			: Path.Combine(FindRepoRoot(), "Examples", directoryName);
	}

	public string GetExamplesBinaryFile(string filename)
	{
		var localPath = Path.ChangeExtension(GetExamplesFilePath(filename), BytecodeSerializer.Extension);
		if (File.Exists(localPath))
			return localPath;
		//ncrunch: no coverage start
		new Runner(TestPackage.Instance, GetExamplesFilePath(filename)).Run().Dispose();
		Assert.That(File.Exists(localPath), Is.True,
			BytecodeSerializer.Extension + " file should have been created");
		writer.GetStringBuilder().Clear();
		return localPath;
	} //ncrunch: no coverage end

	//ncrunch: no coverage start
	private static string FindRepoRoot()
	{
		var dir = AppContext.BaseDirectory;
		while (dir != null)
		{
			if (File.Exists(Path.Combine(dir, "Strict.sln")))
				return dir;
			dir = Path.GetDirectoryName(dir);
		}
		throw new DirectoryNotFoundException("Cannot find repository root (Strict.sln not found)");
	} //ncrunch: no coverage end

	[Test]
	public void RunWithFullDiagnostics()
	{
		using var _ = new Runner(TestPackage.Instance, SimpleCalculatorFilePath, true).Run();
		Assert.That(writer.ToString().Length, Is.GreaterThan(1000));
	}

	[Test]
	public void RunFromBytecodeFileProducesSameOutput()
	{
		var binaryFilePath = GetExamplesBinaryFile("SimpleCalculator");
		using var runner = new Runner(TestPackage.Instance, binaryFilePath).Run();
		Assert.That(writer.ToString(),
			Does.StartWith("2 + 3 = 5" + Environment.NewLine + "2 * 3 = 6"));
	}

	[Test]
	public void RunFromBytecodeFileWithoutStrictSourceFile()
	{
		var tempDirectory = Path.Combine(Path.GetTempPath(), "Strict" + Guid.NewGuid().ToString("N"));
		Directory.CreateDirectory(tempDirectory);
		var copiedSourceFilePath = Path.Combine(tempDirectory, Path.GetFileName(SimpleCalculatorFilePath));
		var copiedBinaryFilePath = Path.ChangeExtension(copiedSourceFilePath, BytecodeSerializer.Extension);
		try
		{
			File.Copy(SimpleCalculatorFilePath, copiedSourceFilePath);
			new Runner(TestPackage.Instance, copiedSourceFilePath).Run().Dispose();
			Assert.That(File.Exists(copiedBinaryFilePath), Is.True);
			writer.GetStringBuilder().Clear();
			File.Delete(copiedSourceFilePath);
			using var _ = new Runner(TestPackage.Instance, copiedBinaryFilePath).Run();
			Assert.That(writer.ToString(),
				Does.StartWith("2 + 3 = 5" + Environment.NewLine + "2 * 3 = 6"));
		}
		finally
		{
			if (Directory.Exists(tempDirectory))
				Directory.Delete(tempDirectory, true);
		}
	}

	[Test]
	public void RunFizzBuzz()
	{
		using var _ = new Runner(TestPackage.Instance, GetExamplesFilePath("FizzBuzz")).Run();
		var output = writer.ToString();
		Assert.That(output, Does.Contain("FizzBuzz(3) = Fizz"));
		Assert.That(output, Does.Contain("FizzBuzz(5) = Buzz"));
		Assert.That(output, Does.Contain("FizzBuzz(15) = FizzBuzz"));
		Assert.That(output, Does.Contain("FizzBuzz(7) = 7"));
	}

	[Test]
	public void RunAreaCalculator()
	{
		using var _ = new Runner(TestPackage.Instance, GetExamplesFilePath("AreaCalculator")).Run();
		var output = writer.ToString();
		Assert.That(output, Does.Contain("Area: 50"));
		Assert.That(output, Does.Contain("Perimeter: 30"));
	}

	[Test]
	public void RunGreeter()
	{
		using var _ = new Runner(TestPackage.Instance, GetExamplesFilePath("Greeter")).Run();
		var output = writer.ToString();
		Assert.That(output, Does.Contain("Hello, World!"));
		Assert.That(output, Does.Contain("Hello, Strict!"));
	}

	[Test]
	public void RunWithPlatformWindowsCreatesAsmFileWithWindowsEntryPoint()
	{
		var pureAdderPath = GetExamplesFilePath("PureAdder");
		var asmPath = Path.ChangeExtension(pureAdderPath, ".asm");
		using var runner = new Runner(TestPackage.Instance, pureAdderPath);
		try { runner.Run(Platform.Windows); }	catch (ToolNotFoundException) { }
		Assert.That(File.Exists(asmPath), Is.True, ".asm file should be created");
		Assert.That(writer.ToString(), Does.Contain("Saved Windows NASM assembly to:"));
		var asmContent = File.ReadAllText(asmPath);
		Assert.That(asmContent, Does.Contain("section .text"));
		Assert.That(asmContent, Does.Contain("global PureAdder"));
		Assert.That(asmContent, Does.Contain("global main"));
		Assert.That(asmContent, Does.Contain("extern ExitProcess"));
	}

	[Test]
	public void RunWithPlatformLinuxCreatesAsmFileWithStartEntryPoint()
	{
		var pureAdderPath = GetExamplesFilePath("PureAdder");
		var asmPath = Path.ChangeExtension(pureAdderPath, ".asm");
		using var runner = new Runner(TestPackage.Instance, pureAdderPath);
		try { runner.Run(Platform.Linux); } catch (ToolNotFoundException) { }
		Assert.That(File.Exists(asmPath), Is.True, ".asm file should be created");
		Assert.That(writer.ToString(), Does.Contain("Saved Linux NASM assembly to:"));
		var asmContent = File.ReadAllText(asmPath);
		Assert.That(asmContent, Does.Contain("global _start"));
		Assert.That(asmContent, Does.Contain("_start:"));
	}

	[Test]
	public void RunWithPlatformWindowsSupportsProgramsWithRuntimeMethodCalls()
	{
		var asmPath = Path.ChangeExtension(SimpleCalculatorFilePath, ".asm");
		using var runner = new Runner(TestPackage.Instance, SimpleCalculatorFilePath);
		try { runner.Run(Platform.Windows); }	catch (ToolNotFoundException) { }
		Assert.That(File.Exists(asmPath), Is.True, ".asm file should be created");
	}

	[Test]
	public void RunFromBytecodeWithPlatformWindowsSupportsRuntimeMethodCalls()
	{
		var binaryFilePath = GetExamplesBinaryFile("SimpleCalculator");
		var asmPath = Path.ChangeExtension(binaryFilePath, ".asm");
		if (File.Exists(asmPath))
			File.Delete(asmPath); //ncrunch: no coverage
		using var runner = new Runner(TestPackage.Instance, binaryFilePath);
		try { runner.Run(Platform.Windows); }	catch (ToolNotFoundException) { }
		Assert.That(File.Exists(asmPath), Is.True, ".asm file should be created for bytecode platform compilation");
	}

	[Test]
	public void RunWithPlatformDoesNotExecuteProgram()
	{
		var pureAdderPath = GetExamplesFilePath("PureAdder");
		using var runner = new Runner(TestPackage.Instance, pureAdderPath);
		try { runner.Run(Platform.Linux); } catch (ToolNotFoundException) { }
		Assert.That(writer.ToString(), Does.Not.Contain("executed"),
			"Platform compilation should not execute the program");
		Assert.That(writer.ToString(), Does.Contain("Saved Linux NASM assembly to:"),
			"Should report that assembly was saved");
	}

	[Test]
	public void RunWithNoPlatformDoesNotCreateAsmFile()
	{
		var asmFilePath = Path.ChangeExtension(SimpleCalculatorFilePath, ".asm");
		if (File.Exists(asmFilePath))
			File.Delete(asmFilePath);
		using var runner = new Runner(TestPackage.Instance, SimpleCalculatorFilePath).Run();
		Assert.That(File.Exists(asmFilePath), Is.False,
			".asm file should NOT be created without a platform flag");
		Assert.That(writer.ToString(), Does.Not.Contain("NASM assembly"));
	}

	[Test]
	public void RunWithPlatformWindowsThrowsToolNotFoundWhenNasmMissing()
	{
		if (NativeExecutableLinker.IsNasmAvailable)
			return;	//ncrunch: no coverage start
		using var runner = new Runner(TestPackage.Instance, GetExamplesFilePath("PureAdder"));
		Assert.Throws<ToolNotFoundException>(() => runner.Run(Platform.Windows));
	} //ncrunch: no coverage end

	[Test]
	public void AsmFileIsNotCreatedWhenRunningFromPrecompiledBytecode()
	{
		var asmPath = Path.ChangeExtension(SimpleCalculatorFilePath, ".asm");
		writer.GetStringBuilder().Clear();
		if (File.Exists(asmPath))
			File.Delete(asmPath); //ncrunch: no coverage
		using var runner = new Runner(TestPackage.Instance,
			Path.ChangeExtension(SimpleCalculatorFilePath, BytecodeSerializer.Extension));
		runner.Run();
		Assert.That(File.Exists(asmPath), Is.False,
			".asm file should not be created when loading precompiled bytecode");
	}

	[Test]
	public void SaveStrictBinaryWithTypeBytecodeEntriesOnly()
	{
		using var archive = System.IO.Compression.ZipFile.OpenRead(
			Path.ChangeExtension(SimpleCalculatorFilePath, BytecodeSerializer.Extension));
		var entries = archive.Entries.Select(entry => entry.FullName).ToList();
		Assert.That(
			entries.All(entry => entry.EndsWith(BytecodeSerializer.BytecodeEntryExtension,
				StringComparison.OrdinalIgnoreCase)), Is.True, string.Join(", ", entries.ToList()));
		Assert.That(entries.Any(entry => entry.Contains("#", StringComparison.Ordinal)), Is.False);
		Assert.That(entries, Does.Contain("SimpleCalculator/SimpleCalculator.bytecode"));
		Assert.That(entries, Does.Contain("Strict/Number.bytecode"));
		Assert.That(entries, Does.Contain("Strict/Logger.bytecode"));
		Assert.That(entries.Count, Is.LessThanOrEqualTo(4));
	}

	[Test]
	public void ExportOnlyUsedMethodsForBaseTypes()
	{
		var binaryFilePath = Path.ChangeExtension(SimpleCalculatorFilePath, BytecodeSerializer.Extension);
		new Runner(TestPackage.Instance, SimpleCalculatorFilePath).Run().Dispose();
		using var archive = System.IO.Compression.ZipFile.OpenRead(binaryFilePath);
		var numberMethodCount = ReadMethodHeaderCount(archive, "Strict/Number.bytecode");
		Assert.That(numberMethodCount, Is.LessThanOrEqualTo(3));
	}

	private static int ReadMethodHeaderCount(System.IO.Compression.ZipArchive archive,
		string entryName)
	{
		var entry = archive.GetEntry(entryName) ?? throw new InvalidOperationException(entryName);
		using var reader = new BinaryReader(entry.Open());
		_ = reader.ReadBytes(6);
		var version = reader.ReadByte();
		if (version != BytecodeSerializer.Version)
			throw new InvalidOperationException("Expected version " + BytecodeSerializer.Version); //ncrunch: no coverage
		var nameCount = reader.Read7BitEncodedInt();
		for (var nameIndex = 0; nameIndex < nameCount; nameIndex++)
			_ = reader.ReadString();
		var memberCount = reader.Read7BitEncodedInt();
		for (var memberIndex = 0; memberIndex < memberCount; memberIndex++)
		{ //ncrunch: no coverage start
			_ = reader.Read7BitEncodedInt();
			_ = reader.Read7BitEncodedInt();
			if (reader.ReadBoolean())
				throw new InvalidOperationException("Unexpected initial value in compact metadata");
		} //ncrunch: no coverage end
		return reader.Read7BitEncodedInt();
	}

	[Test]
	public void RunSumWithProgramArguments()
	{
		using var runner = new Runner(TestPackage.Instance, SumFilePath);
		runner.Run(programArgs: ["5", "10", "20"]);
		Assert.That(writer.ToString(), Does.Contain("35"));
	}

	private string SumFilePath => GetExamplesFilePath("Sum");

	[Test]
	public void RunSumWithNoArgumentsUsesEmptyList()
	{
		using var runner = new Runner(TestPackage.Instance, SumFilePath);
		runner.Run(programArgs: ["0"]);
		Assert.That(writer.ToString(), Does.Contain("0"));
	}

	[Test]
	public void RunFibonacciRunner()
	{
		using var _ = new Runner(TestPackage.Instance, GetExamplesFilePath("FibonacciRunner")).Run();
		var output = writer.ToString();
		Assert.That(output, Does.Contain("Fibonacci(10) = 55"));
		Assert.That(output, Does.Contain("Fibonacci(5) = 5"));
	}

	[Test]
	public void RunNumberStats()
	{
		using var _ = new Runner(TestPackage.Instance, GetExamplesFilePath("NumberStats")).Run();
		var output = writer.ToString();
		Assert.That(output, Does.Contain("Sum: 150"));
		Assert.That(output, Does.Contain("Maximum: 50"));
	}

	[Test]
	public void RunGcdCalculator()
	{
		using var _ = new Runner(TestPackage.Instance, GetExamplesFilePath("GcdCalculator")).Run();
		var output = writer.ToString();
		Assert.That(output, Does.Contain("GCD(48, 18) = 6"));
		Assert.That(output, Does.Contain("GCD(12, 8) = 4"));
	}

	[Test]
	public void RunPixel()
	{
		using var _ = new Runner(TestPackage.Instance, GetExamplesFilePath("Pixel")).Run();
		var output = writer.ToString();
		Assert.That(output, Does.Contain("(100, 150, 200).Brighten is 250"));
		Assert.That(output, Does.Contain("(100, 150, 200).Darken is 50"));
		Assert.That(output, Does.Contain("(60, 120, 180).Darken is 30"));
	}

	[Test]
	public void RunTemperatureConverter()
	{
		using var _ = new Runner(TestPackage.Instance, GetExamplesFilePath("TemperatureConverter")).Run();
		var output = writer.ToString();
		Assert.That(output, Does.Contain("100C in Fahrenheit: 212"));
		Assert.That(output, Does.Contain("0C in Fahrenheit: 32"));
		Assert.That(output, Does.Contain("100C in Kelvin: 373"));
	}

	[Test]
	public void RunExpressionWithSingleConstructorArgAndMethod()
	{
		using var runner = new Runner(TestPackage.Instance, GetExamplesFilePath("FibonacciRunner"));
		runner.RunExpression("FibonacciRunner(5).Compute");
		Assert.That(writer.ToString(), Does.Contain("5"));
	}

	[Test]
	public void RunExpressionWithMultipleConstructorArgs()
	{
		using var runner = new Runner(TestPackage.Instance, GetExamplesFilePath("Pixel"));
		runner.RunExpression("Pixel(100, 150, 200).Brighten");
		Assert.That(writer.ToString(), Does.Contain("250"));
	}

	[Test]
	public void RunExpressionWithZeroConstructorArgValue()
	{
		using var runner = new Runner(TestPackage.Instance, GetExamplesFilePath("TemperatureConverter"));
		runner.RunExpression("TemperatureConverter(0).ToFahrenheit");
		Assert.That(writer.ToString(), Does.Contain("32"));
	}

	//ncrunch: no coverage start
	[Test]
	[Category("Slow")]
	public void RunMemoryPressureProgramTwiceKeepsMemoryBoundedAfterCollection()
	{
		var memoryPressureFilePath = GetExamplesFilePath("MemoryPressure");
		var binaryFilePath = Path.ChangeExtension(memoryPressureFilePath, BytecodeSerializer.Extension);
		if (File.Exists(binaryFilePath))
			File.Delete(binaryFilePath);
		binaryFilePath = GetExamplesBinaryFile("MemoryPressure");
		writer.GetStringBuilder().Clear();
		ForceGarbageCollection();
		using (var firstRun = new Runner(TestPackage.Instance, binaryFilePath).Run())
			;
		ForceGarbageCollection();
		var memoryAfterFirstRun = GC.GetTotalMemory(true);
		using (var secondRun = new Runner(TestPackage.Instance, binaryFilePath).Run())
			;
		ForceGarbageCollection();
		var memoryAfterSecondRun = GC.GetTotalMemory(true);
		var outputLines = writer.ToString().Split(Environment.NewLine,
			StringSplitOptions.RemoveEmptyEntries);
		Assert.That(outputLines.Length, Is.EqualTo(2));
		Assert.That(outputLines[0], Is.EqualTo("allocated: 20001"));
		Assert.That(outputLines[1], Is.EqualTo("allocated: 20001"));
	}

	private static void ForceGarbageCollection()
	{
		GC.Collect();
		GC.WaitForPendingFinalizers();
		GC.Collect();
	}
}