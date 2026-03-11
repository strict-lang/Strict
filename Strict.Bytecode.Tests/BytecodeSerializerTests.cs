using System.IO.Compression;
using System.Text;
using Strict.Bytecode.Instructions;
using Strict.Bytecode.Serialization;

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
		BytecodeSerializer.Serialize(instructions, binaryFilePath, "Add");
		var loaded = BytecodeSerializer.Deserialize(binaryFilePath, TestPackage.Instance);
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
		BytecodeSerializer.Serialize(instructions, binaryFilePath, "SimpleLoopExample");
		var loaded = BytecodeSerializer.Deserialize(binaryFilePath, TestPackage.Instance);
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
		BytecodeSerializer.Serialize(instructions, binaryFilePath, "ArithmeticFunction");
		var loaded = BytecodeSerializer.Deserialize(binaryFilePath, TestPackage.Instance);
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
		BytecodeSerializer.Serialize(instructions, binaryFilePath, "SimpleListDeclaration");
		var loaded = BytecodeSerializer.Deserialize(binaryFilePath, TestPackage.Instance);
		Assert.That(loaded.Count, Is.EqualTo(instructions.Count));
		Assert.That(loaded.ConvertAll(x => x.ToString()),
			Is.EqualTo(instructions.ConvertAll(x => x.ToString())));
	}

	[Test]
	public void SavedFileIsZipWithCorrectMagicInEntry()
	{
		var instructions = new List<Instruction> { new ReturnInstruction(Register.R0) };
		var binaryFilePath = GetTempStrictBinaryFilePath();
		BytecodeSerializer.Serialize(instructions, binaryFilePath, "test");
		using var zip = ZipFile.OpenRead(binaryFilePath);
		using var stream = zip.Entries.Single().Open();
		using var reader = new BinaryReader(stream);
		Assert.That(Encoding.UTF8.GetString(reader.ReadBytes(6)), Is.EqualTo(nameof(Strict)));
		Assert.That(reader.ReadByte(), Is.EqualTo(BytecodeSerializer.Version));
	}

	[Test]
	public void InvalidFileThrows()
	{
		var binaryFilePath = GetTempStrictBinaryFilePath();
		File.WriteAllBytes(binaryFilePath, [0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF]);
		Assert.Throws<BytecodeSerializer.InvalidBytecodeFileException>(() =>
			BytecodeSerializer.Deserialize(binaryFilePath, TestPackage.Instance));
	}

	[Test]
	public void SerializedEntryContentIsCompact()
	{
		var instructions = new BytecodeGenerator(
			GenerateMethodCallFromSource("Add", "Add(10, 5).Calculate",
				"has First Number", "has Second Number", "Calculate Number",
				"\tAdd(10, 5).Calculate is 15", "\tFirst + Second")).Generate();
		var binaryFilePath = GetTempStrictBinaryFilePath();
		BytecodeSerializer.Serialize(instructions, binaryFilePath, "Add");
		using var zip = ZipFile.OpenRead(binaryFilePath);
		// Uncompressed entry content for a simple 6-instruction arithmetic method should be < 50 bytes
		Assert.That(zip.Entries.Single().Length, Is.LessThan(50),
			"Serialized arithmetic bytecode entry should be compact (< 50 bytes)");
	}
}