using Strict.Bytecode.Instructions;
using Strict.Bytecode.Serialization;
using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict.Bytecode.Tests;

public sealed class BinaryExecutableTests : TestBytecode
{
	[Test]
	public void SerializeAndLoadPreservesMethodInstructions()
	{
		var sourceBinary = new BinaryExecutable(TestPackage.Instance);
		sourceBinary.MethodsPerType[Type.Number] = CreateMethods([new ReturnInstruction(Register.R0)]);
		var filePath = CreateTempFilePath();
		sourceBinary.Serialize(filePath);
		var loadedBinary = new BinaryExecutable(filePath, TestPackage.Instance);
		Assert.That(loadedBinary.MethodsPerType[Type.Number].MethodGroups[Method.Run][0].instructions.Count,
			Is.EqualTo(1));
	}

	[Test]
	public void SerializeAndLoadPreservesMemberMetadata()
	{
		var sourceBinary = new BinaryExecutable(TestPackage.Instance);
		sourceBinary.MethodsPerType[Type.Number] = new BinaryType
		{
			Members = [new BinaryMember("value", Type.Number, null)],
			MethodGroups = new Dictionary<string, List<BinaryMethod>>
			{
				[Method.Run] = [new BinaryMethod("", [], Type.None, [new ReturnInstruction(Register.R0)])]
			}
		};
		var filePath = CreateTempFilePath();
		sourceBinary.Serialize(filePath);
		var loadedBinary = new BinaryExecutable(filePath, TestPackage.Instance);
		Assert.That(loadedBinary.MethodsPerType[Type.Number].Members[0].Name, Is.EqualTo("value"));
	}

	[Test]
	public void FindInstructionsReturnsMatchingMethodOverload()
	{
		var binary = new BinaryExecutable(TestPackage.Instance);
		binary.MethodsPerType[Type.Number] = new BinaryType
		{
			Members = [],
			MethodGroups = new Dictionary<string, List<BinaryMethod>>
			{
				["Compute"] =
				[
					new BinaryMethod("", [], Type.None, [new ReturnInstruction(Register.R0)]),
					new BinaryMethod("", [new BinaryMember("value", Type.Number, null)],
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
		var binary = new BinaryExecutable(TestPackage.Instance);
		using var stream = new MemoryStream([(byte)255]);
		using var reader = new BinaryReader(stream);
		Assert.That(() => binary.ReadInstruction(reader, new NameTable()),
			Throws.TypeOf<BinaryExecutable.InvalidFile>().With.Message.Contains("Unknown instruction type"));
	}

	[Test]
	public void InvalidZipThrowsInvalidFile()
	{
		var filePath = CreateTempFilePath();
		File.WriteAllBytes(filePath, [0x42, 0x00, 0x10]);
		Assert.That(() => new BinaryExecutable(filePath, TestPackage.Instance),
			Throws.TypeOf<BinaryExecutable.InvalidFile>());
	}

	private static BinaryType CreateMethods(List<Instruction> instructions) =>
		new()
		{
			Members = [],
			MethodGroups = new Dictionary<string, List<BinaryMethod>>
			{
				[Method.Run] = [new BinaryMethod("", [], Type.None, instructions)]
			}
		};

	private static string CreateTempFilePath() =>
		Path.Combine(Path.GetTempPath(), "strictbinary-tests-" + Guid.NewGuid() + BinaryExecutable.Extension);
}
