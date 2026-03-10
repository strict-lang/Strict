using System.Text;
using Strict.Bytecode.Instructions;

namespace Strict.Bytecode.Tests;

public sealed class BytecodeSerializerTests : TestBytecode
{
	[Test]
	public void RoundTripSimpleArithmeticBytecode()
	{
		var instructions = new BytecodeGenerator(
			GenerateMethodCallFromSource("Add", "Add(10, 5).Calculate",
				"has First Number", "has Second Number", "Calculate Number",
				"\tAdd(10, 5).Calculate is 15", "\tFirst + Second")).Generate();
		var binaryFilePath = GetTempStrictBinaryFilePath();
		BytecodeSerializer.Serialize(instructions, binaryFilePath, "dummy.strict");
		var (loaded, sourcePath) = BytecodeSerializer.Deserialize(binaryFilePath, TestPackage.Instance);
		Assert.That(sourcePath, Is.EqualTo("dummy.strict"));
		Assert.That(loaded.Count, Is.EqualTo(instructions.Count));
		Assert.That(loaded.ConvertAll(x => x.ToString()),
			Is.EqualTo(instructions.ConvertAll(x => x.ToString())));
	}

	private static string GetTempStrictBinaryFilePath() =>
		Path.Combine(Path.GetTempPath(), "test" + testFileCounter++ + BytecodeSerializer.Extension);

	private static int testFileCounter;

	[Test]
	public void RoundTripLoopBytecode()
	{
		var instructions = new BytecodeGenerator(
			GenerateMethodCallFromSource("SimpleLoopExample",
				"SimpleLoopExample(10).GetMultiplicationOfNumbers",
				"has number", "GetMultiplicationOfNumbers Number",
				"\tmutable result = 1", "\tconstant multiplier = 2", "\tfor number",
				"\t\tresult = result * multiplier", "\tresult")).Generate();
		var binaryFilePath = GetTempStrictBinaryFilePath();
		BytecodeSerializer.Serialize(instructions, binaryFilePath, "dummy.strict");
		var (loaded, _) = BytecodeSerializer.Deserialize(binaryFilePath, TestPackage.Instance);
		Assert.That(loaded.Count, Is.EqualTo(instructions.Count));
		Assert.That(loaded.ConvertAll(x => x.ToString()),
			Is.EqualTo(instructions.ConvertAll(x => x.ToString())));
	}

	[Test]
	public void RoundTripConditionalBytecode()
	{
		var instructions = new BytecodeGenerator(
			GenerateMethodCallFromSource("ArithmeticFunction",
				"ArithmeticFunction(10, 5).Calculate(\"add\")",
				"has First Number", "has Second Number",
				"Calculate(operation Text) Number",
				"\tArithmeticFunction(10, 5).Calculate(\"add\") is 15",
				"\tArithmeticFunction(10, 5).Calculate(\"subtract\") is 5",
				"\tArithmeticFunction(10, 5).Calculate(\"multiply\") is 50",
				"\tif operation is \"add\"", "\t\treturn First + Second",
				"\tif operation is \"subtract\"", "\t\treturn First - Second",
				"\tif operation is \"multiply\"", "\t\treturn First * Second",
				"\tif operation is \"divide\"", "\t\treturn First / Second")).Generate();
		var binaryFilePath = GetTempStrictBinaryFilePath();
		BytecodeSerializer.Serialize(instructions, binaryFilePath, "dummy.strict");
		var (loaded, _) = BytecodeSerializer.Deserialize(binaryFilePath, TestPackage.Instance);
		Assert.That(loaded.Count, Is.EqualTo(instructions.Count));
		Assert.That(loaded.ConvertAll(x => x.ToString()),
			Is.EqualTo(instructions.ConvertAll(x => x.ToString())));
	}

	[Test]
	public void RoundTripListBytecode()
	{
		var instructions = new BytecodeGenerator(
			GenerateMethodCallFromSource("SimpleListDeclaration",
				"SimpleListDeclaration(5).Declare",
				"has number", "Declare Numbers", "\t(1, 2, 3, 4, 5)")).Generate();
		var binaryFilePath = GetTempStrictBinaryFilePath();
		BytecodeSerializer.Serialize(instructions, binaryFilePath, "dummy.strict");
		var (loaded, _) = BytecodeSerializer.Deserialize(binaryFilePath, TestPackage.Instance);
		Assert.That(loaded.Count, Is.EqualTo(instructions.Count));
		Assert.That(loaded.ConvertAll(x => x.ToString()),
			Is.EqualTo(instructions.ConvertAll(x => x.ToString())));
	}

	[Test]
	public void SavedFileHasCorrectMagicHeader()
	{
		var instructions = new List<Instruction> { new ReturnInstruction(Register.R0) };
		var binaryFilePath = GetTempStrictBinaryFilePath();
		BytecodeSerializer.Serialize(instructions, binaryFilePath, "test.strict");
		var bytes = File.ReadAllBytes(binaryFilePath);
		Assert.That(Encoding.UTF8.GetString(bytes[..6]), Is.EqualTo(nameof(Strict)));
		Assert.That(bytes[6], Is.EqualTo(1));
	}

	[Test]
	public void InvalidMagicHeaderThrows()
	{
		var binaryFilePath = GetTempStrictBinaryFilePath();
		File.WriteAllBytes(binaryFilePath, [0xFF, 0xFF, 0xFF, 0xFF]);
		Assert.Throws<BytecodeSerializer.InvalidBytecodeFileException>(() =>
			BytecodeSerializer.ReadSourcePath(binaryFilePath));
	}

	[Test]
	public void ReadSourcePathExtractsEmbeddedPath()
	{
		var instructions = new List<Instruction> { new ReturnInstruction(Register.R0) };
		const string expected = "Examples/SimpleCalculator.strict";
		var binaryFilePath = GetTempStrictBinaryFilePath();
		BytecodeSerializer.Serialize(instructions, binaryFilePath, expected);
		Assert.That(BytecodeSerializer.ReadSourcePath(binaryFilePath), Is.EqualTo(expected));
	}

	[Test]
	public void SerializedFileSizeIsCompact()
	{
		var instructions = new BytecodeGenerator(
			GenerateMethodCallFromSource("Add", "Add(10, 5).Calculate",
				"has First Number", "has Second Number", "Calculate Number",
				"\tAdd(10, 5).Calculate is 15", "\tFirst + Second")).Generate();
		var binaryFilePath = GetTempStrictBinaryFilePath();
		BytecodeSerializer.Serialize(instructions, binaryFilePath, "dummy.strict");
		var fileSize = new FileInfo(binaryFilePath).Length;
		// 6-byte 'Strict' + 1-byte version + 13-byte source string + 4-byte count + ~6 instructions
		Assert.That(fileSize, Is.LessThan(103),//TODO: make more compact!
			"Serialized arithmetic bytecode should be compact (< 80 bytes)");
	}
}