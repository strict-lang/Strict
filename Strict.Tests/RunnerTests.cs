using Strict.Bytecode;
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
			new Runner(TestPackage.Instance, StrictFilePath).Run().Dispose();
			Assert.That(File.Exists(binaryFilePath), Is.True,
				BytecodeSerializer.Extension + " file should have been created");
			writer.GetStringBuilder().Clear();
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
}