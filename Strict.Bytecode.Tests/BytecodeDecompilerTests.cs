using Strict.Bytecode.Instructions;
using Strict.Bytecode.Serialization;
using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict.Bytecode.Tests;

public sealed class BytecodeDecompilerTests : TestBytecode
{
	[Test]
	public void DecompileSimpleArithmeticBytecodeCreatesStrictFile()
	{
		var instructions = new BinaryGenerator(
			GenerateMethodCallFromSource("Add", "Add(10, 5).Calculate",
				"has First Number", "has Second Number", "Calculate Number",
				"\tAdd(10, 5).Calculate is 15", "\tFirst + Second")).Generate();
		var outputFolder = DecompileToTemp(instructions, "Add");
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
		var instructions = new BinaryGenerator(
			GenerateMethodCallFromSource("Counter", "Counter(5).Calculate",
				"has count Number",
				"Double Number",
				"\tCounter(3).Double is 6",
				"\tcount * 2",
				"Calculate Number",
				"\tCounter(5).Calculate is 10",
				"\tconstant doubled = Counter(3).Double",
				"\tdoubled * 2")).Generate();
		var outputFolder = DecompileToTemp(instructions, "Counter");
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

	private static string DecompileToTemp(IReadOnlyList<Instruction> instructions, string typeName)
	{
		var strictBinary = new BinaryExecutable(TestPackage.Instance);
		strictBinary.MethodsPerType[typeName] = CreateTypeMethods(instructions);
		var outputFolder = Path.Combine(Path.GetTempPath(), "decompiled_" + Path.GetRandomFileName());
		new Decompiler().Decompile(strictBinary, outputFolder);
		Assert.That(Directory.Exists(outputFolder), Is.True, "Output folder should be created");
		Assert.That(File.Exists(Path.Combine(outputFolder, typeName + ".strict")), Is.True,
			typeName + ".strict should be created");
		return outputFolder;
	}

	private static BinaryType CreateTypeMethods(IReadOnlyList<Instruction> instructions)
	{
		var methods = new BinaryType();
		methods.Members = [];
		methods.MethodGroups = new Dictionary<string, List<BinaryType.BinaryMethod>>
		{
			[Method.Run] =
			[
				new BinaryType.BinaryMethod([], Type.None, instructions)
			]
		};
		return methods;
	}
}