using Strict.Bytecode.Serialization;
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
		using var _ = new Runner(TestPackage.Instance, "Examples/SimpleCalculator.strict").Run();
		Assert.That(writer.ToString(), Does.StartWith("2 + 3 = 5" + Environment.NewLine + "2 * 3 = 6"));
	}

	[Test]
	public void RunWithFullDiagnostics()
	{
		using var _ = new Runner(TestPackage.Instance, "Examples/SimpleCalculator.strict", true).Run();
		Assert.That(writer.ToString().Length, Is.GreaterThan(1000));
	}

	[Test]
	public void RunFromBytecodeFileProducesSameOutput()
	{
		const string StrictFilePath = "Examples/SimpleCalculator.strict";
		var binaryFilePath = Path.ChangeExtension(StrictFilePath, BytecodeSerializer.Extension);
		try
		{
			if (!File.Exists(binaryFilePath))
			{ //ncrunch: no coverage start, only needed once
				new Runner(TestPackage.Instance, StrictFilePath).Run().Dispose();
				Assert.That(File.Exists(binaryFilePath), Is.True,
					BytecodeSerializer.Extension + " file should have been created");
				writer.GetStringBuilder().Clear();
			} //ncrunch: no coverage end
			using var runner = new Runner(TestPackage.Instance, binaryFilePath).Run();
			Assert.That(writer.ToString(),
				Does.StartWith("2 + 3 = 5" + Environment.NewLine + "2 * 3 = 6"));
		}
		finally
		{
			if (File.Exists(binaryFilePath))
				File.Delete(binaryFilePath);
		}
	}

	[Test]
	[Description("Without source on disk, sub-method calls fail because the VM needs method " +
		"bodies. Once all methods are pre-compiled into the .strictbinary this will work.")]
	public void RunFromBytecodeFileWithoutStrictSourceFileNotYetSupported()
	{
		const string SourceFilePath = "Examples/SimpleCalculator.strict";
		var tempDirectory = Path.Combine(Path.GetTempPath(), "Strict" + Guid.NewGuid().ToString("N"));
		Directory.CreateDirectory(tempDirectory);
		var copiedSourceFilePath = Path.Combine(tempDirectory, Path.GetFileName(SourceFilePath));
		var copiedBinaryFilePath = Path.ChangeExtension(copiedSourceFilePath, BytecodeSerializer.Extension);
		try
		{
			File.Copy(SourceFilePath, copiedSourceFilePath);
			new Runner(TestPackage.Instance, copiedSourceFilePath).Run().Dispose();
			Assert.That(File.Exists(copiedBinaryFilePath), Is.True);
			File.Delete(copiedSourceFilePath);
			Assert.That(() => new Runner(TestPackage.Instance, copiedBinaryFilePath).Run(),
				Throws.TypeOf<Method.CannotCallBodyOnTraitMethod>());
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
		using var _ = new Runner(TestPackage.Instance, "Examples/FizzBuzz.strict").Run();
		var output = writer.ToString();
		Assert.That(output, Does.Contain("FizzBuzz(3) = Fizz"));
		Assert.That(output, Does.Contain("FizzBuzz(5) = Buzz"));
		Assert.That(output, Does.Contain("FizzBuzz(15) = FizzBuzz"));
		Assert.That(output, Does.Contain("FizzBuzz(7) = 7"));
	}

	[Test]
	public void RunAreaCalculator()
	{
		using var _ = new Runner(TestPackage.Instance, "Examples/AreaCalculator.strict").Run();
		var output = writer.ToString();
		Assert.That(output, Does.Contain("Area: 50"));
		Assert.That(output, Does.Contain("Perimeter: 30"));
	}

	[Test]
	public void RunGreeter()
	{
		using var _ = new Runner(TestPackage.Instance, "Examples/Greeter.strict").Run();
		var output = writer.ToString();
		Assert.That(output, Does.Contain("Hello, World!"));
		Assert.That(output, Does.Contain("Hello, Strict!"));
	}

	[Test]
	public void RunningFromSourceCreatesAsmFileNextToBinary()
	{
		const string StrictFilePath = "Examples/SimpleCalculator.strict";
		var asmPath = Path.ChangeExtension(StrictFilePath, ".asm");
		var binaryPath = Path.ChangeExtension(StrictFilePath, BytecodeSerializer.Extension);
		try
		{
			using var _ = new Runner(TestPackage.Instance, StrictFilePath).Run();
			Assert.That(File.Exists(asmPath), Is.True, ".asm file should be created next to .strictbinary");
			Assert.That(writer.ToString(), Does.Contain("Saved x64 NASM assembly to:"));
			Assert.That(File.ReadAllText(asmPath), Does.Contain("section .text"));
			Assert.That(File.ReadAllText(asmPath), Does.Contain("global SimpleCalculator"));
		}
		finally
		{
			if (File.Exists(asmPath))
				File.Delete(asmPath);
			if (File.Exists(binaryPath))
				File.Delete(binaryPath);
		}
	}

	[Test]
	public void AsmFileIsNotCreatedWhenRunningFromPrecompiledBytecode()
	{
		const string StrictFilePath = "Examples/SimpleCalculator.strict";
		var binaryPath = Path.ChangeExtension(StrictFilePath, BytecodeSerializer.Extension);
		var asmPath = Path.ChangeExtension(StrictFilePath, ".asm");
		try
		{
			new Runner(TestPackage.Instance, StrictFilePath).Run().Dispose();
			writer.GetStringBuilder().Clear();
			if (File.Exists(asmPath))
				File.Delete(asmPath);
			using var _ = new Runner(TestPackage.Instance, binaryPath).Run();
			Assert.That(File.Exists(asmPath), Is.False, ".asm file should not be created when loading precompiled bytecode");
		}
		finally
		{
			if (File.Exists(binaryPath))
				File.Delete(binaryPath);
			if (File.Exists(asmPath))
				File.Delete(asmPath);
		}
	}
}