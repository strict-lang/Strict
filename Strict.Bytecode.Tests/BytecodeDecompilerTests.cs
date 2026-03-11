using Strict.Bytecode.Instructions;
using Strict.Bytecode.Serialization;

namespace Strict.Bytecode.Tests;

public sealed class BytecodeDecompilerTests : TestBytecode
{
	[Test]
	public void DecompileSimpleArithmeticBytecodeCreatesStrictFile()
	{
		var instructions = new BytecodeGenerator(
			GenerateMethodCallFromSource("Add", "Add(10, 5).Calculate",
				"has First Number", "has Second Number", "Calculate Number",
				"\tAdd(10, 5).Calculate is 15", "\tFirst + Second")).Generate();
		var (binaryFilePath, outputFolder) = SerializeAndDecompile(instructions, "Add");
		try
		{
			var content = File.ReadAllText(Path.Combine(outputFolder, "Add.strict"));
			Assert.That(content, Does.Contain("has First Number"));
		}
		finally
		{
			Cleanup(binaryFilePath, outputFolder);
		}
	}

	[Test]
	public void DecompileRunMethodReconstructsConstantDeclarationFromMethodCall()
	{
		var instructions = new BytecodeGenerator(
			GenerateMethodCallFromSource("Counter", "Counter(5).Calculate",
				"has count Number",
				"Double Number",
				"\tCounter(3).Double is 6",
				"\tcount * 2",
				"Calculate Number",
				"\tCounter(5).Calculate is 10",
				"\tconstant doubled = Counter(3).Double",
				"\tdoubled * 2")).Generate();
		var (binaryFilePath, outputFolder) = SerializeAndDecompile(instructions, "Counter");
		try
		{
			var content = File.ReadAllText(Path.Combine(outputFolder, "Counter.strict"));
			Assert.That(content, Does.Contain("has count Number"));
			Assert.That(content, Does.Contain("Run"));
			Assert.That(content, Does.Contain("Counter(3).Double"));
		}
		finally
		{
			Cleanup(binaryFilePath, outputFolder);
		}
	}

	//TODO: improve based on caller
	private static (string binaryFilePath, string outputFolder) SerializeAndDecompile(
		List<Instruction> instructions, string typeName)
	{
		var binaryFilePath = GetTempStrictBinaryFilePath();
		BytecodeSerializer.Serialize(instructions, binaryFilePath, typeName);
		var outputFolder = Path.Combine(Path.GetTempPath(), "decompiled_" + Path.GetRandomFileName());
		new BytecodeDecompiler(TestPackage.Instance).Decompile(binaryFilePath, outputFolder);
		Assert.That(Directory.Exists(outputFolder), Is.True, "Output folder should be created");
		Assert.That(File.Exists(Path.Combine(outputFolder, typeName + ".strict")), Is.True,
			typeName + ".strict should be created");
		return (binaryFilePath, outputFolder);
	}

	private static void Cleanup(string binaryFilePath, string outputFolder)
	{
		if (Directory.Exists(outputFolder))
			Directory.Delete(outputFolder, recursive: true);
		if (File.Exists(binaryFilePath))
			File.Delete(binaryFilePath);
	}

	private static string GetTempStrictBinaryFilePath() =>
		Path.Combine(Path.GetTempPath(), nameof(BytecodeDecompilerTests) + decompTestCounter++ +
			BytecodeSerializer.Extension);

	private static int decompTestCounter;
}
