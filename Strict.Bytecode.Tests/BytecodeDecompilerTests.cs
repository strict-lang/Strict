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
		var outputFolder = SerializeAndDecompile(instructions, "Add");
		try
		{
			var content = File.ReadAllText(Path.Combine(outputFolder, "Add.strict"));
			Assert.That(content, Does.Contain("constant First"));
		}
		finally
		{
			if (Directory.Exists(outputFolder))
				Directory.Delete(outputFolder, recursive: true);
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
		var outputFolder = SerializeAndDecompile(instructions, "Counter");
		try
		{
			var content = File.ReadAllText(Path.Combine(outputFolder, "Counter.strict"));
			Assert.That(content, Does.Contain("Run"));
			Assert.That(content, Does.Contain("Counter(3).Double"));
		}
		finally
		{
			if (Directory.Exists(outputFolder))
				Directory.Delete(outputFolder, recursive: true);
		}
	}

	private static string SerializeAndDecompile(List<Instruction> instructions, string typeName)
	{
		var binaryFilePath = new BytecodeSerializer(
			new Dictionary<string, IList<Instruction>> { [typeName] = instructions },
			Path.GetTempPath(),
			nameof(BytecodeDecompilerTests) + decompTestCounter++).OutputFilePath;
		var outputFolder = Path.Combine(Path.GetTempPath(), "decompiled_" + Path.GetRandomFileName());
		var bytecodeTypes = new BytecodeDeserializer(binaryFilePath).Deserialize(TestPackage.Instance);
		new BytecodeDecompiler().Decompile(bytecodeTypes, outputFolder);
		Assert.That(Directory.Exists(outputFolder), Is.True, "Output folder should be created");
		Assert.That(File.Exists(Path.Combine(outputFolder, typeName + ".strict")), Is.True,
			typeName + ".strict should be created");
		return outputFolder;
	}

	private static int decompTestCounter;
}