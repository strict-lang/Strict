using Strict.Bytecode.Instructions;
using Type = Strict.Language.Type;

namespace Strict.Bytecode.Tests;

public sealed class BytecodeSerializerTests : TestBytecode
{
	private static readonly string TempSbcPath =
		Path.Combine(Path.GetTempPath(), "strict_test.sbc");

	[TearDown]
	public void DeleteTempFile()
	{
		if (File.Exists(TempSbcPath))
			File.Delete(TempSbcPath);
	}

	[Test]
	public void RoundTripSimpleArithmeticBytecode()
	{
		var instructions = new BytecodeGenerator(
			GenerateMethodCallFromSource("Add", "Add(10, 5).Calculate",
				"has First Number", "has Second Number", "Calculate Number",
				"\tAdd(10, 5).Calculate is 15", "\tFirst + Second")).Generate();
		BytecodeSerializer.Serialize(instructions, TempSbcPath, "dummy.strict");
		var (loaded, sourcePath) = BytecodeSerializer.Deserialize(TempSbcPath, TestPackage.Instance);
		Assert.That(sourcePath, Is.EqualTo("dummy.strict"));
		Assert.That(loaded.Count, Is.EqualTo(instructions.Count));
		Assert.That(loaded.ConvertAll(x => x.ToString()),
			Is.EqualTo(instructions.ConvertAll(x => x.ToString())));
	}

	[Test]
	public void RoundTripLoopBytecode()
	{
		var instructions = new BytecodeGenerator(
			GenerateMethodCallFromSource("SimpleLoopExample",
				"SimpleLoopExample(10).GetMultiplicationOfNumbers",
				"has number", "GetMultiplicationOfNumbers Number",
				"\tmutable result = 1", "\tconstant multiplier = 2", "\tfor number",
				"\t\tresult = result * multiplier", "\tresult")).Generate();
		BytecodeSerializer.Serialize(instructions, TempSbcPath, "dummy.strict");
		var (loaded, _) = BytecodeSerializer.Deserialize(TempSbcPath, TestPackage.Instance);
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
		BytecodeSerializer.Serialize(instructions, TempSbcPath, "dummy.strict");
		var (loaded, _) = BytecodeSerializer.Deserialize(TempSbcPath, TestPackage.Instance);
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
		BytecodeSerializer.Serialize(instructions, TempSbcPath, "dummy.strict");
		var (loaded, _) = BytecodeSerializer.Deserialize(TempSbcPath, TestPackage.Instance);
		Assert.That(loaded.Count, Is.EqualTo(instructions.Count));
		Assert.That(loaded.ConvertAll(x => x.ToString()),
			Is.EqualTo(instructions.ConvertAll(x => x.ToString())));
	}

	[Test]
	public void SavedFileHasCorrectMagicHeader()
	{
		var instructions =
			new List<Instruction> { new ReturnInstruction(Register.R0) };
		BytecodeSerializer.Serialize(instructions, TempSbcPath, "test.strict");
		var bytes = File.ReadAllBytes(TempSbcPath);
		Assert.That(bytes[0], Is.EqualTo(0x73)); // 's'
		Assert.That(bytes[1], Is.EqualTo(0x62)); // 'b'
		Assert.That(bytes[2], Is.EqualTo(0x63)); // 'c'
		Assert.That(bytes[3], Is.EqualTo(0x01)); // version 1
	}

	[Test]
	public void InvalidMagicHeaderThrows()
	{
		File.WriteAllBytes(TempSbcPath, [0xFF, 0xFF, 0xFF, 0xFF]);
		Assert.Throws<BytecodeSerializer.InvalidBytecodeFileException>(() =>
			BytecodeSerializer.ReadSourcePath(TempSbcPath));
	}

	[Test]
	public void ReadSourcePathExtractsEmbeddedPath()
	{
		var instructions = new List<Instruction> { new ReturnInstruction(Register.R0) };
		const string expected = "Examples/SimpleCalculator.strict";
		BytecodeSerializer.Serialize(instructions, TempSbcPath, expected);
		Assert.That(BytecodeSerializer.ReadSourcePath(TempSbcPath), Is.EqualTo(expected));
	}

	[Test]
	public void SerializedFileSizeIsCompact()
	{
		var instructions = new BytecodeGenerator(
			GenerateMethodCallFromSource("Add", "Add(10, 5).Calculate",
				"has First Number", "has Second Number", "Calculate Number",
				"\tAdd(10, 5).Calculate is 15", "\tFirst + Second")).Generate();
		BytecodeSerializer.Serialize(instructions, TempSbcPath, "dummy.strict");
		var fileSize = new FileInfo(TempSbcPath).Length;
		// 4-byte magic + 1-byte source-path length prefix + 12-byte source string + 4-byte count
		// + ~6 instructions ≈ ~70 bytes total; allow up to 150 bytes for safe headroom.
		Assert.That(fileSize, Is.LessThan(150),
			"Serialized arithmetic bytecode should be compact (< 150 bytes)");
	}
}
