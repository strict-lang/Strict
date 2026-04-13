using Strict.Bytecode.Instructions;
using Strict.Bytecode.Serialization;
using System.IO.Compression;
using Type = Strict.Language.Type;

namespace Strict.Bytecode.Tests;

public sealed class BinaryExecutableTests : TestBytecode
{
	[Test]
	public void SerializeAndLoadPreservesMethodInstructions()
	{
		var sourceBinary = new BinaryExecutable(TestPackage.Instance);
		sourceBinary.MethodsPerType[Type.Number] = CreateMethods(sourceBinary, Type.Number,
			[new ReturnInstruction(Register.R0)]);
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
		sourceBinary.MethodsPerType[Type.Number] = new BinaryType(sourceBinary, Type.Number,
			[new BinaryMember("value", Type.Number, null)],
			new Dictionary<string, List<BinaryMethod>>
			{
				[Method.Run] = [new BinaryMethod("", [], Type.None, [new ReturnInstruction(Register.R0)])]
			});
		var filePath = CreateTempFilePath();
		sourceBinary.Serialize(filePath);
		var loadedBinary = new BinaryExecutable(filePath, TestPackage.Instance);
		Assert.That(loadedBinary.MethodsPerType[Type.Number].Members[0].Name, Is.EqualTo("value"));
	}

	[Test]
	public void FindInstructionsReturnsMatchingMethodOverload()
	{
		var binary = new BinaryExecutable(TestPackage.Instance);
		binary.MethodsPerType[Type.Number] = new BinaryType(binary, Type.Number, [],
			new Dictionary<string, List<BinaryMethod>>
			{
				["Compute"] =
				[
					new BinaryMethod("", [], Type.None, [new ReturnInstruction(Register.R0)]),
					new BinaryMethod("", [new BinaryMember("value", Type.Number, null)],
						Type.Number, [new ReturnInstruction(Register.R1)])
				]
			});
		var found = binary.FindInstructions(Type.Number, "Compute", 1, Type.Number);
		Assert.That(found![0].InstructionType, Is.EqualTo(InstructionType.Return));
	}

	[Test]
	public void ReadingUnknownInstructionTypeThrowsInvalidFile()
	{
		var binary = new BinaryExecutable(TestPackage.Instance);
		using var stream = new MemoryStream([105]);
		using var reader = new BinaryReader(stream);
		Assert.That(() => binary.ReadInstruction(reader, new NameTable(Type.Number)),
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

	private static BinaryType CreateMethods(BinaryExecutable executable, string typeFullName,
		List<Instruction> instructions) =>
		new(executable, typeFullName, [], new Dictionary<string, List<BinaryMethod>>
		{
			[Method.Run] = [new BinaryMethod("", [], Type.None, instructions)]
		});

	private static string CreateTempFilePath() =>
		Path.Combine(Path.GetTempPath(), "strictbinary-tests-" + Guid.NewGuid() + BinaryExecutable.Extension);

	[Test]
	public void ZipWithNoBytecodeEntriesCreatesEmptyStrictBinary()
	{
		var filePath = CreateEmptyZipWithDummyEntry();
		var binary = new BinaryExecutable(filePath, TestPackage.Instance);
		Assert.That(binary.MethodsPerType, Is.Empty);
	}

	[Test]
	public void EntryWithBadMagicBytesThrows()
	{
		var filePath = CreateZipWithSingleEntry([0xBA, 0x01]);
		Assert.That(() => new BinaryExecutable(filePath, TestPackage.Instance),
			Throws.TypeOf<BinaryType.InvalidBytecodeEntry>().With.Message.Contains("magic byte"));
	}

	[Test]
	public void VersionZeroThrows()
	{
		var filePath = CreateZipWithSingleEntry(BuildEntryBytes(writer =>
		{
			writer.Write(MagicBytes);
			writer.Write((byte)0);
		}));
		Assert.That(() => new BinaryExecutable(filePath, TestPackage.Instance),
			Throws.TypeOf<BinaryType.InvalidVersion>().With.Message.Contains("version"));
	}

	[Test]
	public void UnknownValueKindThrows()
	{
		var filePath = CreateZipWithSingleEntry(BuildEntryBytes(writer =>
		{
			WriteHeader(writer, ["member", "Number"]);
			writer.Write7BitEncodedInt(1);
			writer.Write7BitEncodedInt(0);
			writer.Write7BitEncodedInt(1);
			writer.Write(true);
			writer.Write((byte)InstructionType.LoadConstantToRegister);
			writer.Write((byte)Register.R0);
			writer.Write((byte)0xFF);
			writer.Write7BitEncodedInt(0);
			writer.Write7BitEncodedInt(0);
		}));
		Assert.That(() => new BinaryExecutable(filePath, TestPackage.Instance),
			Throws.TypeOf<BinaryExecutable.InvalidFile>().With.Message.Contains("Unknown ValueKind"));
	}

	private static void WriteHeader(BinaryWriter writer, string[] names)
	{
		writer.Write(MagicBytes);
		writer.Write(BinaryType.Version);
		writer.Write7BitEncodedInt(names.Length);
		foreach (var name in names)
			writer.Write(name);
	}

	private static string CreateEmptyZipWithDummyEntry()
	{
		var filePath = GetTempFilePath();
		using var fileStream = new FileStream(filePath, FileMode.Create);
		using var zip = new ZipArchive(fileStream, ZipArchiveMode.Create);
		zip.CreateEntry("dummy.txt");
		return filePath;
	}

	private static string CreateZipWithSingleEntry(byte[] entryBytes)
	{
		var filePath = GetTempFilePath();
		using var fileStream = new FileStream(filePath, FileMode.Create);
		using var zip = new ZipArchive(fileStream, ZipArchiveMode.Create);
		var entry = zip.CreateEntry("Number" + BinaryType.BytecodeEntryExtension);
		using var stream = entry.Open();
		stream.Write(entryBytes);
		return filePath;
	}

	private static string GetTempFilePath() =>
		Path.Combine(Path.GetTempPath(), "strictbinary" + fileCounter++ + BinaryExecutable.Extension);

	private static int fileCounter;
	private static readonly byte[] MagicBytes = [(byte)'S'];

	private static byte[] BuildEntryBytes(Action<BinaryWriter> writeContent)
	{
		using var stream = new MemoryStream();
		using var writer = new BinaryWriter(stream);
		writeContent(writer);
		writer.Flush();
		return stream.ToArray();
	}

	[Test]
	public void RoundTripSimpleArithmeticBytecode()
	{
		var binary = new BinaryGenerator(
			GenerateMethodCallFromSource("Add", "Add(10, 5).Calculate",
				"has First Number", "has Second Number", "Calculate Number",
				"\tAdd(10, 5).Calculate is 15", "\tFirst + Second")).Generate();
		AssertRoundTripToString([.. binary.ToInstructions()]);
	}

	[Test]
	public void RoundTripSetInstruction() =>
		AssertRoundTripInstructionTypes([
			new SetInstruction(Number(42), Register.R0),
			new ReturnInstruction(Register.R0)
		]);

	[Test]
	public void RoundTripStoreFromRegisterInstruction() =>
		AssertRoundTripInstructionTypes([
			new StoreFromRegisterInstruction(Register.R1, "result"),
			new ReturnInstruction(Register.R0)
		]);

	[Test]
	public void RoundTripAllBinaryOperators() =>
		AssertRoundTripInstructionTypes([
			new BinaryInstruction(InstructionType.Add, Register.R0, Register.R1),
			new BinaryInstruction(InstructionType.Subtract, Register.R0, Register.R1),
			new BinaryInstruction(InstructionType.Multiply, Register.R0, Register.R1),
			new BinaryInstruction(InstructionType.Divide, Register.R0, Register.R1),
			new BinaryInstruction(InstructionType.Modulo, Register.R0, Register.R1),
			new BinaryInstruction(InstructionType.Equal, Register.R0, Register.R1),
			new BinaryInstruction(InstructionType.NotEqual, Register.R0, Register.R1),
			new BinaryInstruction(InstructionType.LessThan, Register.R0, Register.R1),
			new BinaryInstruction(InstructionType.GreaterThan, Register.R0, Register.R1),
			new ReturnInstruction(Register.R0)
		]);

	[Test]
	public void RoundTripJumpInstructions() =>
		AssertRoundTripInstructionTypes([
			new Jump(3),
			new Jump(2, InstructionType.JumpIfTrue),
			new Jump(1, InstructionType.JumpIfFalse),
			new JumpIfNotZero(5, Register.R0),
			new JumpToId(10, InstructionType.JumpEnd),
			new JumpToId(20, InstructionType.JumpToIdIfFalse),
			new JumpToId(30, InstructionType.JumpToIdIfTrue),
			new ReturnInstruction(Register.R0)
		]);

	[Test]
	public void RoundTripCollectionMutationInstructions() =>
		AssertRoundTripInstructionTypes([
			new WriteToListInstruction(Register.R0, "myList"),
			new WriteToTableInstruction(Register.R0, Register.R1, "myTable"),
			new RemoveInstruction(Register.R2, "myList"),
			new ListCallInstruction(Register.R0, Register.R1, "myList"),
			new ReturnInstruction(Register.R0)
		]);

	[Test]
	public void RoundTripPrintInstructionWithNumberRegister()
	{
		var loaded = RoundTripInstructions([
			new PrintInstruction("Value = ", Register.R2),
			new ReturnInstruction(Register.R0)
		]);
		var print = (PrintInstruction)loaded[0];
		Assert.That(print.TextPrefix, Is.EqualTo("Value = "));
		Assert.That(print.ValueRegister, Is.EqualTo(Register.R2));
		Assert.That(print.ValueIsText, Is.False);
	}

	[Test]
	public void RoundTripDictionaryValue()
	{
		var dictionaryType = TestPackage.Instance.GetDictionaryImplementationType(NumberType, NumberType);
		var items = new Dictionary<ValueInstance, ValueInstance>
		{
			{ Number(1), Number(10) },
			{ Number(2), Number(20) }
		};
		var loaded = RoundTripInstructions([
			new LoadConstantInstruction(Register.R0, new ValueInstance(dictionaryType, items)),
			new ReturnInstruction(Register.R0)
		]);
		var loadedDictionary = ((LoadConstantInstruction)loaded[0]).Constant;
		Assert.That(loadedDictionary.IsDictionary, Is.True);
		Assert.That(loadedDictionary.GetDictionaryItems()[Number(1)].Number, Is.EqualTo(10));
	}

	private static void AssertRoundTripInstructionTypes(IList<Instruction> instructions)
	{
		var loaded = RoundTripInstructions(instructions);
		Assert.That(loaded.Count, Is.EqualTo(instructions.Count));
		for (var index = 0; index < instructions.Count; index++)
			Assert.That(loaded[index].InstructionType, Is.EqualTo(instructions[index].InstructionType));
	}

	private static void AssertRoundTripToString(IList<Instruction> instructions) =>
		Assert.That(RoundTripInstructions(instructions).ConvertAll(instruction => instruction.ToString()),
			Is.EqualTo(instructions.ToList().ConvertAll(instruction => instruction.ToString())));

	private static List<Instruction> RoundTripInstructions(IList<Instruction> instructions)
	{
		using var stream = new MemoryStream();
		using var writer = new BinaryWriter(stream);
		var table = new NameTable(Type.Number);
		foreach (var instruction in instructions)
			table.CollectStrings(instruction);
		table.Write(writer);
		writer.Write7BitEncodedInt(instructions.Count);
		foreach (var instruction in instructions)
			instruction.Write(writer, table);
		writer.Flush();
		stream.Position = 0;
		using var reader = new BinaryReader(stream);
		var readTable = new NameTable(reader, Type.Number);
		var count = reader.Read7BitEncodedInt();
		var binary = new BinaryExecutable(TestPackage.Instance);
		var loaded = new List<Instruction>(count);
		for (var index = 0; index < count; index++)
			loaded.Add(binary.ReadInstruction(reader, readTable));
		return loaded;
	}

	[Test]
	public void BinaryTypeHeaderUsesSingleMagicByteAndVersion()
	{
		var binary = new BinaryGenerator(
			GenerateMethodCallFromSource("Add", "Add(10, 5).Calculate",
				"has First Number", "has Second Number", "Calculate Number",
				"\tFirst + Second")).Generate();
		var typeToWrite = binary.MethodsPerType.Values.First();
		using var stream = new MemoryStream();
		using var writer = new BinaryWriter(stream);
		typeToWrite.Write(writer);
		writer.Flush();
		var bytes = stream.ToArray();
		Assert.That(bytes[0], Is.EqualTo((byte)'S'));
		Assert.That(bytes[1], Is.EqualTo(BinaryType.Version));
	}

	[Test]
	public void NameTableWritesOnlyCustomNamesAndPrefillsBaseTypes()
	{
		var table = new NameTable(Type.Number);
		table.Add("CustomIdentifier");
		using var stream = new MemoryStream();
		using (var writer = new BinaryWriter(stream, System.Text.Encoding.UTF8, leaveOpen: true))
			table.Write(writer);
		stream.Position = 0;
		using var headerReader = new BinaryReader(stream, System.Text.Encoding.UTF8, leaveOpen: true);
		Assert.That(headerReader.Read7BitEncodedInt(), Is.EqualTo(1));
		stream.Position = 0;
		using var reader = new BinaryReader(stream);
		var readTable = new NameTable(reader, Type.Number);
		Assert.That(readTable.names.Contains(Type.Number), Is.True);
		Assert.That(readTable.names.Contains("CustomIdentifier"), Is.True);
	}

	[Test]
	public void EntryNameTableDoesNotStoreBaseFullNamesOrEntryTypeName()
	{
		var binary = new BinaryGenerator(
			GenerateMethodCallFromSource("Add", "Add(10, 5).Calculate",
				"has First Number", "has Second Number", "Calculate Number", "\tFirst + Second")).Generate();
		var addTypeKey = binary.MethodsPerType.Keys.First(typeName => !typeName.StartsWith("Strict/",
			StringComparison.Ordinal));
		var addType = binary.MethodsPerType[addTypeKey];
		using var stream = new MemoryStream();
		using (var writer = new BinaryWriter(stream, System.Text.Encoding.UTF8, leaveOpen: true))
			addType.Write(writer);
		stream.Position = 2;
		using var reader = new BinaryReader(stream);
		var customNamesCount = reader.Read7BitEncodedInt();
		var customNames = new List<string>(customNamesCount);
		for (var nameIndex = 0; nameIndex < customNamesCount; nameIndex++)
			customNames.Add(reader.ReadString());
		Assert.That(customNames, Does.Not.Contain("Strict/Number"));
		Assert.That(customNames, Does.Not.Contain("Strict/Text"));
		Assert.That(customNames, Does.Not.Contain("Strict/Boolean"));
		Assert.That(customNames, Does.Not.Contain("Add"));
	}

	[Test]
	public void NameTablePrefillsRequestedCommonNames()
	{
		var table = new NameTable(Type.Number);
		using var stream = new MemoryStream();
		using (var writer = new BinaryWriter(stream, System.Text.Encoding.UTF8, leaveOpen: true))
		{
			table.Add("index");
			table.Add("value");
			table.Add("first");
			table.Add("second");
			table.Add("from");
			table.Add("Run");
			table.Add("characters");
			table.Add("NewLine");
			table.Add("Tab");
			table.Add("textWriter");
			table.Write(writer);
		}
		stream.Position = 0;
		using var reader = new BinaryReader(stream);
		Assert.That(reader.Read7BitEncodedInt(), Is.EqualTo(0));
	}
}