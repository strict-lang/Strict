using Strict.Bytecode.Instructions;
using Strict.Bytecode.Serialization;
using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict.Bytecode.Tests;

public sealed class StrictBinaryTests : TestBytecode
{
	[Test]
	public void SerializeAndLoadPreservesMethodInstructions()
	{
		var sourceBinary = new StrictBinary(TestPackage.Instance);
		sourceBinary.MethodsPerType[Type.Number] = CreateMethods([new ReturnInstruction(Register.R0)]);
		var filePath = CreateTempFilePath();
		sourceBinary.Serialize(filePath);
		var loadedBinary = new StrictBinary(filePath, TestPackage.Instance);
		Assert.That(loadedBinary.MethodsPerType[Type.Number].InstructionsPerMethodGroup[Method.Run][0].Instructions.Count,
			Is.EqualTo(1));
	}

	[Test]
	public void SerializeAndLoadPreservesMemberMetadata()
	{
		var sourceBinary = new StrictBinary(TestPackage.Instance);
		sourceBinary.MethodsPerType[Type.Number] = new BytecodeMembersAndMethods
		{
			Members = [new BytecodeMember("value", Type.Number, null)],
			InstructionsPerMethodGroup = new Dictionary<string, List<BytecodeMembersAndMethods.MethodInstructions>>
			{
				[Method.Run] = [new BytecodeMembersAndMethods.MethodInstructions([], Type.None, [new ReturnInstruction(Register.R0)])]
			}
		};
		var filePath = CreateTempFilePath();
		sourceBinary.Serialize(filePath);
		var loadedBinary = new StrictBinary(filePath, TestPackage.Instance);
		Assert.That(loadedBinary.MethodsPerType[Type.Number].Members[0].Name, Is.EqualTo("value"));
	}

	[Test]
	public void FindInstructionsReturnsMatchingMethodOverload()
	{
		var binary = new StrictBinary(TestPackage.Instance);
		binary.MethodsPerType[Type.Number] = new BytecodeMembersAndMethods
		{
			Members = [],
			InstructionsPerMethodGroup = new Dictionary<string, List<BytecodeMembersAndMethods.MethodInstructions>>
			{
				["Compute"] =
				[
					new BytecodeMembersAndMethods.MethodInstructions([], Type.None, [new ReturnInstruction(Register.R0)]),
					new BytecodeMembersAndMethods.MethodInstructions([new BytecodeMember("value", Type.Number, null)],
						Type.Number, [new ReturnInstruction(Register.R1)])
				]
			}
		};
		var found = binary.FindInstructions(Type.Number, "Compute", 1, Type.Number);
		Assert.That(found![0].InstructionType, Is.EqualTo(InstructionType.Return));
	}

	[Test]
	public void ReadingUnknownInstructionTypeThrowsInvalidFile()
	{
		var binary = new StrictBinary(TestPackage.Instance);
		using var stream = new MemoryStream([(byte)255]);
		using var reader = new BinaryReader(stream);
		Assert.That(() => binary.ReadInstruction(reader, new NameTable()),
			Throws.TypeOf<StrictBinary.InvalidFile>().With.Message.Contains("Unknown instruction type"));
	}

	[Test]
	public void InvalidZipThrowsInvalidFile()
	{
		var filePath = CreateTempFilePath();
		File.WriteAllBytes(filePath, [0x42, 0x00, 0x10]);
		Assert.That(() => new StrictBinary(filePath, TestPackage.Instance),
			Throws.TypeOf<StrictBinary.InvalidFile>());
	}

	private static BytecodeMembersAndMethods CreateMethods(IReadOnlyList<Instruction> instructions) =>
		new()
		{
			Members = [],
			InstructionsPerMethodGroup = new Dictionary<string, List<BytecodeMembersAndMethods.MethodInstructions>>
			{
				[Method.Run] = [new BytecodeMembersAndMethods.MethodInstructions([], Type.None, instructions)]
			}
		};

	private static string CreateTempFilePath() =>
		Path.Combine(Path.GetTempPath(), "strictbinary-tests-" + Guid.NewGuid() + StrictBinary.Extension);
}
