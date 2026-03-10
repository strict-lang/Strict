using Strict.Bytecode.Instructions;

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
		var binaryFilePath = GetTempStrictBinaryFilePath();
		BytecodeSerializer.Serialize(instructions, binaryFilePath, "Add");
		var outputFolder = Path.Combine(Path.GetTempPath(), "decompiled_" + Path.GetRandomFileName());
		try
		{
			BytecodeDecompiler.Decompile(binaryFilePath, outputFolder, TestPackage.Instance);
			Assert.That(Directory.Exists(outputFolder), Is.True, "Output folder should be created");
			var outputFile = Path.Combine(outputFolder, "Add.strict");
			Assert.That(File.Exists(outputFile), Is.True, "Add.strict should be created");
			var content = File.ReadAllText(outputFile);
			Assert.That(content, Does.Contain("Decompiled from Add.bytecode"));
		}
		finally
		{
			if (Directory.Exists(outputFolder))
				Directory.Delete(outputFolder, recursive: true);
			if (File.Exists(binaryFilePath))
				File.Delete(binaryFilePath);
		}
	}

	private static string GetTempStrictBinaryFilePath() =>
		Path.Combine(Path.GetTempPath(), "decomp_test" + decompTestCounter++ + BytecodeSerializer.Extension);

	private static int decompTestCounter;
}
