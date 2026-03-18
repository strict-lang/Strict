using Strict.Bytecode;
using Strict.Bytecode.Serialization;
using Strict.Compiler;
using Strict.Compiler.Assembly;
using Strict.Expressions;
using Strict.Language;
using Strict.Language.Tests;
using System.Diagnostics;
using System.IO.Compression;

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
	public void RestoreConsole()
	{
		Console.SetOut(rememberConsole);
		CleanupGeneratedFiles();
	}

	private static void CleanupGeneratedFiles()
	{
		var examplesDir = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "..", "..", "..", "..",
			"Examples");
		if (!Directory.Exists(examplesDir))
			return; //ncrunch: no coverage start
		foreach (var ext in new[] { ".ll", ".mlir", ".llvm.mlir", ".asm", ".obj", ".exe" })
		foreach (var file in Directory.GetFiles(examplesDir, "*" + ext))
			File.Delete(file);
		foreach (var strict in Directory.GetFiles(examplesDir, "*.strict"))
		{
			var noExt = Path.ChangeExtension(strict, null);
			if (File.Exists(noExt) && !noExt.EndsWith(".strict", StringComparison.Ordinal))
				File.Delete(noExt);
		}
	} //ncrunch: no coverage end

	[Test]
	public void RunSimpleCalculator()
	{
		var asmFilePath = Path.ChangeExtension(SimpleCalculatorFilePath, ".asm");
		if (File.Exists(asmFilePath))
			File.Delete(asmFilePath); //ncrunch: no coverage
		using var _ = new Runner(SimpleCalculatorFilePath, TestPackage.Instance).Run();
		Assert.That(writer.ToString(),
			Is.EqualTo("2 + 3 = 5" + Environment.NewLine + "2 * 3 = 6" + Environment.NewLine));
		Assert.That(File.Exists(asmFilePath), Is.False,
			".asm file should NOT be created without a platform flag");
		Assert.That(writer.ToString(), Does.Not.Contain("NASM assembly"));
	}

	[Test]
	public void RunBaseTypesTestPackageFromDirectory()
	{
		using var _ = new Runner(GetExamplesDirectoryPath("BaseTypesTest"), TestPackage.Instance).Run();
		var output = writer.ToString();
		Assert.That(output, Does.Contain("Hello, World!"));
		Assert.That(output, Does.Contain("Hello, Strict!"));
		Assert.That(output, Does.Contain("3 + 4 = 7"));
		Assert.That(output, Does.Contain("10 * 3 = 30"));
		Assert.That(output, Does.Contain("(1, 2, 3).Sum = 6"));
	}

	private static string SimpleCalculatorFilePath => GetExamplesFilePath("SimpleCalculator");

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
		new Runner(GetExamplesFilePath(filename), TestPackage.Instance).Run().Dispose();
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
		using var _ = new Runner(SimpleCalculatorFilePath, TestPackage.Instance,
			enableTestsAndDetailedOutput: true).Run();
		Assert.That(writer.ToString().Length, Is.GreaterThan(1000));
	}

	[Test]
	public void RunFromBytecodeFileProducesSameOutput()
	{
		var binaryFilePath = GetExamplesBinaryFile("SimpleCalculator");
		using var runner = new Runner(binaryFilePath, TestPackage.Instance).Run();
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
			new Runner(copiedSourceFilePath, TestPackage.Instance).Run().Dispose();
			Assert.That(File.Exists(copiedBinaryFilePath), Is.True);
			writer.GetStringBuilder().Clear();
			File.Delete(copiedSourceFilePath);
			using var _ = new Runner(copiedBinaryFilePath, TestPackage.Instance).Run();
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
		using var _ = new Runner(GetExamplesFilePath("FizzBuzz"), TestPackage.Instance).Run();
		var output = writer.ToString();
		Assert.That(output, Does.Contain("FizzBuzz(3) = Fizz"));
		Assert.That(output, Does.Contain("FizzBuzz(5) = Buzz"));
		Assert.That(output, Does.Contain("FizzBuzz(15) = FizzBuzz"));
		Assert.That(output, Does.Contain("FizzBuzz(7) = 7"));
	}

	[Test]
	public void RunAreaCalculator()
	{
		using var _ = new Runner(GetExamplesFilePath("AreaCalculator"), TestPackage.Instance).Run();
		var output = writer.ToString();
		Assert.That(output, Does.Contain("Area: 50"));
		Assert.That(output, Does.Contain("Perimeter: 30"));
	}

	[Test]
	public void RunGreeter()
	{
		using var _ = new Runner(GetExamplesFilePath("Greeter"), TestPackage.Instance).Run();
		var output = writer.ToString();
		Assert.That(output, Does.Contain("Hello, World!"));
		Assert.That(output, Does.Contain("Hello, Strict!"));
	}

	[Test]
	public async Task RunWithPlatformWindowsCreatesAsmFileWithWindowsEntryPoint()
	{
		var pureAdderPath = GetExamplesFilePath("PureAdder");
		var asmPath = Path.ChangeExtension(pureAdderPath, ".asm");
		var runner = new Runner(pureAdderPath, TestPackage.Instance);
		if (!NativeExecutableLinker.IsNasmAvailable)
			return; //ncrunch: no coverage
		await runner.Build(Platform.Windows, CompilerBackend.Nasm);
		Assert.That(File.Exists(asmPath), Is.True, ".asm file should be created");
		Assert.That(writer.ToString(), Does.Contain("Saved Windows NASM assembly to:"));
		var asmContent = await File.ReadAllTextAsync(asmPath);
		Assert.That(asmContent, Does.Contain("section .text"));
		Assert.That(asmContent, Does.Contain("global PureAdder"));
		Assert.That(asmContent, Does.Contain("global main"));
		Assert.That(asmContent, Does.Contain("extern ExitProcess"));
	}

	[Test]
	public async Task RunWithPlatformLinuxCreatesAsmFileWithStartEntryPoint()
	{
		var pureAdderPath = GetExamplesFilePath("PureAdder");
		var asmPath = Path.ChangeExtension(pureAdderPath, ".asm");
		var executablePath = Path.ChangeExtension(asmPath, null);
		var runner = new Runner(pureAdderPath, TestPackage.Instance);
		if (!NativeExecutableLinker.IsNasmAvailable)
			return; //ncrunch: no coverage start
		if (OperatingSystem.IsLinux())
		{
			await runner.Build(Platform.Linux, CompilerBackend.Nasm);
			Assert.That(File.Exists(executablePath), Is.True, "Linux executable should be created");
			Assert.That(writer.ToString(), Does.Contain("Saved Linux NASM assembly to:"));
			Assert.That(File.Exists(asmPath), Is.True, ".asm file should be created");
			var asmContent = await File.ReadAllTextAsync(asmPath);
			Assert.That(asmContent, Does.Contain("global _start"));
			Assert.That(asmContent, Does.Contain("_start:"));
		} //ncrunch: no coverage end
		else
			await runner.Build(Platform.Windows, CompilerBackend.Nasm);
	}

	[Test]
	public async Task RunWithPlatformWindowsSupportsProgramsWithRuntimeMethodCalls()
	{
		var llvmPath = Path.ChangeExtension(SimpleCalculatorFilePath, ".ll");
		var runner = new Runner(SimpleCalculatorFilePath, TestPackage.Instance);
		if (!LlvmLinker.IsClangAvailable)
			return; //ncrunch: no coverage start
		await runner.Build(Platform.Windows);
		Assert.That(File.Exists(llvmPath), Is.True, ".ll file should be created");
		Assert.That(writer.ToString(), Does.Contain("Saved Windows MLIR to:"));
	} //ncrunch: no coverage end

	[Test]
	public async Task RunFromBytecodeWithPlatformWindowsSupportsRuntimeMethodCalls()
	{
		var binaryFilePath = GetExamplesBinaryFile("SimpleCalculator");
		var llvmPath = Path.ChangeExtension(binaryFilePath, ".ll");
		if (File.Exists(llvmPath))
			File.Delete(llvmPath); //ncrunch: no coverage
		var runner = new Runner(binaryFilePath, TestPackage.Instance);
		if (!LlvmLinker.IsClangAvailable)
			return; //ncrunch: no coverage start
		await runner.Build(Platform.Windows);
		Assert.That(File.Exists(llvmPath), Is.True,
			".ll file should be created for bytecode platform compilation");
		Assert.That(writer.ToString(), Does.Contain("Saved Windows MLIR to:"));
	} //ncrunch: no coverage end

	[TestCase(CompilerBackend.MlirDefault)]
	[TestCase(CompilerBackend.Llvm)]
	[TestCase(CompilerBackend.Nasm)]
	public async Task RunWithPlatformDoesNotExecuteProgram(CompilerBackend backend)
	{
		var calculatorFilePath = GetExamplesFilePath("SimpleCalculator");
		var runner = new Runner(calculatorFilePath, TestPackage.Instance);
		await runner.Build(OperatingSystem.IsWindows()
			? Platform.Windows
			: Platform.Linux, backend);
		Assert.That(writer.ToString(), Does.Not.Contain("executed"),
			"Platform compilation should not execute the program");
		Assert.That(writer.ToString(), Does.Contain("Saved"),
			"Should report that assembly was saved");
		if (OperatingSystem.IsWindows())
		{
			var process = new Process();
			process.StartInfo.FileName = Path.ChangeExtension(calculatorFilePath, ".exe");
			process.StartInfo.UseShellExecute = false;
			process.StartInfo.RedirectStandardOutput = true;
			process.StartInfo.RedirectStandardError = true;
			process.Start();
			Assert.That(process.StandardOutput.ReadToEnd().Replace("\r\n", "\n"),
				Is.EqualTo("2 + 3 = 5\n2 * 3 = 6\n"));
		}
	}

	[Test]
	public void BuildWithExpressionEntryPointThrows()
	{
		var runner = new Runner(SimpleCalculatorFilePath, TestPackage.Instance, "(1, 2, 3).Length");
		Assert.That(async () => await runner.Build(Platform.Windows),
			Throws.TypeOf<NotSupportedException>().With.Message.Contains("expression"));
	}

	[Test]
	public async Task BuildSumExecutableAcceptsRuntimeArguments()
	{
		var tempDirectory = Path.Combine(Path.GetTempPath(), "Strict" + Guid.NewGuid().ToString("N"));
		Directory.CreateDirectory(tempDirectory);
		try
		{
			var sumFilePath = Path.Combine(tempDirectory, Path.GetFileName(SumFilePath));
			File.Copy(SumFilePath, sumFilePath);
			await new Runner(sumFilePath, TestPackage.Instance).Build(OperatingSystem.IsWindows()
				? Platform.Windows
				: OperatingSystem.IsMacOS()
					? Platform.MacOS
					: Platform.Linux);
			var executablePath = OperatingSystem.IsWindows()
				? Path.ChangeExtension(sumFilePath, ".exe")
				: Path.ChangeExtension(sumFilePath, null);
			using var process = new Process();
			process.StartInfo.FileName = executablePath;
			process.StartInfo.Arguments = "5 10";
			process.StartInfo.WorkingDirectory = tempDirectory;
			process.StartInfo.UseShellExecute = false;
			process.StartInfo.RedirectStandardOutput = true;
			process.Start();
			var output = await process.StandardOutput.ReadToEndAsync();
			await process.WaitForExitAsync();
			Assert.That(output.Replace("\r\n", "\n"), Does.Contain("15\n"));
		}
		finally
		{
			if (Directory.Exists(tempDirectory))
				Directory.Delete(tempDirectory, true);
		}
	}

	[Test]
	public void RunWithPlatformWindowsThrowsToolNotFoundWhenNasmMissing()
	{
		if (NativeExecutableLinker.IsNasmAvailable)
			return; //ncrunch: no coverage start
		var runner = new Runner(GetExamplesFilePath("PureAdder"), TestPackage.Instance);
		Assert.That(async () => await runner.Build(Platform.Windows),
			Throws.TypeOf<ToolNotFoundException>());
	} //ncrunch: no coverage end

	[Test]
	public async Task RunWithMlirBackendCreatesMlirFileForLinux()
	{
		if (!LlvmLinker.IsClangAvailable)
			return; //ncrunch: no coverage
		var pureAdderPath = GetExamplesFilePath("PureAdder");
		var llvmPath = Path.ChangeExtension(pureAdderPath, ".ll");
		var runner = new Runner(pureAdderPath, TestPackage.Instance);
		await runner.Build(Platform.Linux);
		Assert.That(File.Exists(llvmPath), Is.True, ".ll file should be created");
		Assert.That(writer.ToString(), Does.Contain("Saved Linux MLIR to:"));
		var irContent = await File.ReadAllTextAsync(llvmPath);
		Assert.That(irContent, Does.Contain("define double @PureAdder("));
		Assert.That(irContent, Does.Contain("define i32 @main()"));
		Assert.That(irContent, Does.Contain("ret i32 0"));
	}

	[Test]
	public async Task RunWithLlvmBackendProducesLinuxExecutable()
	{
		if (!LlvmLinker.IsClangAvailable || !OperatingSystem.IsLinux())
			return; //ncrunch: no coverage start
		var pureAdderPath = GetExamplesFilePath("PureAdder");
		var llvmPath = Path.ChangeExtension(pureAdderPath, ".ll");
		var exePath = Path.ChangeExtension(llvmPath, null);
		var runner = new Runner(pureAdderPath, TestPackage.Instance);
		await runner.Build(Platform.Linux);
		Assert.That(writer.ToString(), Does.Contain("via LLVM"));
		Assert.That(File.Exists(exePath), Is.True,
			"Linux executable should be created by LLVM backend");
	} //ncrunch: no coverage end

	[Test]
	public async Task AsmFileIsNotCreatedWhenRunningFromPrecompiledBytecode()
	{
		var asmPath = Path.ChangeExtension(SimpleCalculatorFilePath, ".asm");
		writer.GetStringBuilder().Clear();
		if (File.Exists(asmPath))
			File.Delete(asmPath); //ncrunch: no coverage
		var runner = new Runner(Path.ChangeExtension(SimpleCalculatorFilePath, BytecodeSerializer.Extension),
			TestPackage.Instance);
		await runner.Run();
		Assert.That(File.Exists(asmPath), Is.False,
			".asm file should not be created when loading precompiled bytecode");
	}

	[Test]
	public void SaveStrictBinaryWithTypeBytecodeEntriesOnly()
	{
		using var archive = ZipFile.OpenRead(
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
		var binaryFilePath = GetExamplesBinaryFile("SimpleCalculator");
		using var archive = ZipFile.OpenRead(binaryFilePath);
		var numberMethodCount = ReadMethodHeaderCount(archive, "Strict/Number.bytecode");
		Assert.That(numberMethodCount, Is.LessThanOrEqualTo(3));
	}

	private static int ReadMethodHeaderCount(ZipArchive archive,
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
	public async Task RunSumWithProgramArguments()
	{
		var runner = new Runner(SumFilePath, TestPackage.Instance, "5 10 20");
		await runner.Run();
		Assert.That(writer.ToString(), Does.Contain("35"));
	}

	[Test]
	public async Task RunSumWithDifferentProgramArgumentsDoesNotReuseCachedEntryPoint()
	{
		await new Runner(SumFilePath, TestPackage.Instance, "5 10 20").Run();
		writer.GetStringBuilder().Clear();
		await new Runner(SumFilePath, TestPackage.Instance, "1 2").Run();
		Assert.That(writer.ToString(), Does.Contain("3"));
	}

	private static string SumFilePath => GetExamplesFilePath("Sum");

	[Test]
	public async Task RunSumWithNoArgumentsUsesEmptyList()
	{
		var runner = new Runner(SumFilePath, TestPackage.Instance, "0");
		await runner.Run();
		Assert.That(writer.ToString(), Does.Contain("0"));
	}

	[Test]
	public async Task RunFibonacciRunner()
	{
		await new Runner(GetExamplesFilePath("FibonacciRunner"), TestPackage.Instance).Run();
		var output = writer.ToString();
		Assert.That(output, Does.Contain("Fibonacci(10) = 55"));
		Assert.That(output, Does.Contain("Fibonacci(5) = 5"));
	}

	[Test]
	public async Task RunNumberStats()
	{
		await new Runner(GetExamplesFilePath("NumberStats"), TestPackage.Instance).Run();
		var output = writer.ToString();
		Assert.That(output, Does.Contain("Sum: 150"));
		Assert.That(output, Does.Contain("Maximum: 50"));
	}

	[Test]
	public async Task RunGcdCalculator()
	{
		await new Runner(GetExamplesFilePath("GcdCalculator"), TestPackage.Instance).Run();
		var output = writer.ToString();
		Assert.That(output, Does.Contain("GCD(48, 18) = 6"));
		Assert.That(output, Does.Contain("GCD(12, 8) = 4"));
	}

	[Test]
	public async Task RunPixel()
	{
		await new Runner(GetExamplesFilePath("Pixel"), TestPackage.Instance).Run();
		var output = writer.ToString();
		Assert.That(output, Does.Contain("(100, 150, 200).Brighten is 250"));
		Assert.That(output, Does.Contain("(100, 150, 200).Darken is 50"));
		Assert.That(output, Does.Contain("(60, 120, 180).Darken is 30"));
	}

	[Test]
	public async Task RunTemperatureConverter()
	{
		await new Runner(GetExamplesFilePath("TemperatureConverter"), TestPackage.Instance).Run();
		var output = writer.ToString();
		Assert.That(output, Does.Contain("100C in Fahrenheit: 212"));
		Assert.That(output, Does.Contain("0C in Fahrenheit: 32"));
		Assert.That(output, Does.Contain("100C in Kelvin: 373"));
	}

	[Test]
	public async Task RunExpressionWithSingleConstructorArgAndMethod()
	{
		await new Runner(GetExamplesFilePath("FibonacciRunner"), TestPackage.Instance,
			"FibonacciRunner(5).Compute").Run();
		Assert.That(writer.ToString(), Does.Contain("5"));
	}

	[Test]
	public async Task RunExpressionWithMultipleConstructorArgs()
	{
		await new Runner(GetExamplesFilePath("Pixel"), TestPackage.Instance,
			"Pixel(100, 150, 200).Brighten").Run();
		Assert.That(writer.ToString(), Does.Contain("250"));
	}

	[Test]
	public async Task RunExpressionWithZeroConstructorArgValue()
	{
		await new Runner(GetExamplesFilePath("TemperatureConverter"), TestPackage.Instance,
			"TemperatureConverter(0).ToFahrenheit").Run();
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
		new Runner(binaryFilePath, TestPackage.Instance).Run().Dispose();
		ForceGarbageCollection();
		new Runner(binaryFilePath, TestPackage.Instance).Run().Dispose();
		ForceGarbageCollection();
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

	[Test]
	public void GeneratedBinaryHasOnlyQualifiedMainTypeEntryAndIncludesNumberDependency()
	{
		var parser = new MethodExpressionParser();
		var typeName = Path.GetFileNameWithoutExtension(SimpleCalculatorFilePath);
		var sourceLines = File.ReadAllLines(SimpleCalculatorFilePath);
		using var mainType = new Language.Type(TestPackage.Instance,
			new TypeLines(typeName, sourceLines)).ParseMembersAndMethods(parser);
		var expression = parser.ParseExpression(new Body(new Method(mainType, 0, parser,
			new[] { nameof(GeneratedBinaryHasOnlyQualifiedMainTypeEntryAndIncludesNumberDependency) })),
			Method.Run);
		var binary = new BinaryGenerator(expression).Generate();
		var tempBinaryPath = Path.Combine(Path.GetTempPath(), "strictbinary-test-" +
			Guid.NewGuid().ToString("N") + BytecodeSerializer.Extension);
		try
		{
			binary.Serialize(tempBinaryPath);
			using var zip = ZipFile.OpenRead(tempBinaryPath);
			var entryNames = zip.Entries
				.Where(entry => entry.FullName.EndsWith(".bytecode", StringComparison.Ordinal))
				.Select(entry => entry.FullName[..^".bytecode".Length].Replace('\\', '/'))
				.ToList();
			Assert.That(entryNames, Does.Not.Contain(typeName),
				"Main type must be stored as a qualified package path, not duplicated as plain type name");
			Assert.That(entryNames.Any(entryName =>
				entryName.EndsWith("/" + typeName, StringComparison.Ordinal)), Is.True,
				"Main type entry must exist with its package path");
			Assert.That(entryNames.Any(entryName =>
				entryName.EndsWith("/Number", StringComparison.Ordinal) ||
				entryName == Language.Type.Number), Is.True,
				"Binary must include Number base type dependency entry");
		}
		finally
		{
			if (File.Exists(tempBinaryPath))
				File.Delete(tempBinaryPath);
		}
	}
}