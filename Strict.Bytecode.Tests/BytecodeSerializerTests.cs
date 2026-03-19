using Strict.Bytecode.Instructions;
using Strict.Bytecode.Serialization;
using Type = Strict.Language.Type;

namespace Strict.Bytecode.Tests;

public sealed class BytecodeSerializerTests : TestBytecode
{
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
		var table = new NameTable();
		foreach (var instruction in instructions)
			table.CollectStrings(instruction);
		table.Write(writer);
		writer.Write7BitEncodedInt(instructions.Count);
		foreach (var instruction in instructions)
			instruction.Write(writer, table);
		writer.Flush();
		stream.Position = 0;
		using var reader = new BinaryReader(stream);
		var readTable = new NameTable(reader);
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
		var table = new NameTable();
		table.Add(Type.Number);
		table.Add("CustomIdentifier");
		using var stream = new MemoryStream();
		using (var writer = new BinaryWriter(stream, System.Text.Encoding.UTF8, leaveOpen: true))
			table.Write(writer);
		stream.Position = 0;
		using var headerReader = new BinaryReader(stream, System.Text.Encoding.UTF8, leaveOpen: true);
		Assert.That(headerReader.Read7BitEncodedInt(), Is.EqualTo(1));
		stream.Position = 0;
		using var reader = new BinaryReader(stream);
		var readTable = new NameTable(reader);
		Assert.That(readTable.Names.Contains(Type.Number), Is.True);
		Assert.That(readTable.Names.Contains("CustomIdentifier"), Is.True);
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
		var table = new NameTable();
		using var stream = new MemoryStream();
		using (var writer = new BinaryWriter(stream, System.Text.Encoding.UTF8, leaveOpen: true))
		{
			table.Add("first");
			table.Add("second");
			table.Add("from");
			table.Add("Run");
			table.Add("characters");
			table.Add("Strict/List(Character)");
			table.Add("Strict/List(Number)");
			table.Add("Strict/List(Text)");
			table.Add("zeroCharacter");
			table.Add("NewLine");
			table.Add("Tab");
			table.Add("textWriter");
			table.Write(writer);
		}
		stream.Position = 0;
		using var reader = new BinaryReader(stream);
		Assert.That(reader.Read7BitEncodedInt(), Is.EqualTo(0));
	}

	private readonly Type boolType = TestPackage.Instance.GetType(Type.Boolean);
}

/*old tests, TODO: integrate!
 * using System.IO.Compression;
using System.Text;
using Strict.Bytecode.Instructions;
using Strict.Bytecode.Serialization;
using Type = Strict.Language.Type;

namespace Strict.Bytecode.Tests;

public sealed class BytecodeSerializerTests : TestBytecode
{
	[Test]
	public void RoundTripSimpleArithmeticBytecode()
	{
		var instructions = new BinaryGenerator(
			GenerateMethodCallFromSource("Add", "Add(10, 5).Calculate",
				"has First Number", "has Second Number", "Calculate Number",
				"\tAdd(10, 5).Calculate is 15", "\tFirst + Second")).Generate();
		var loaded = RoundTripToInstructions("Add", instructions);
		Assert.That(loaded.Count, Is.EqualTo(instructions.Count));
		Assert.That(loaded.ConvertAll(x => x.ToString()),
			Is.EqualTo(instructions.ConvertAll(x => x.ToString())));
		AssertRoundTripToString(instructions);
	}

	private static Dictionary<string, byte[]> SerializeToMemory(string typeName,
		IList<Instruction> instructions) =>
		BytecodeSerializer.SerializeToEntryBytes(
			new Dictionary<string, IList<Instruction>> { [typeName] = instructions });

	private static List<Instruction> RoundTripToInstructions(string typeName,
		IList<Instruction> instructions) =>
		new BytecodeDeserializer(SerializeToMemory(typeName, instructions), TestPackage.Instance).
			Instructions![typeName];

	private static string SerializeToTemp(string typeName, IList<Instruction> instructions) =>
		new BytecodeSerializer(
			new Dictionary<string, IList<Instruction>> { [typeName] = instructions },
			Path.GetTempPath(), "test" + testFileCounter++).OutputFilePath;

	private static int testFileCounter;

	private static string GetTempStrictBinaryFilePath() =>
		Path.Combine(Path.GetTempPath(), "test" + testFileCounter++ + BytecodeSerializer.Extension);

	[Test]
	public void RoundTripLoopBytecode()
	{
		var instructions = new BinaryGenerator(
			GenerateMethodCallFromSource("SimpleLoopExample",
				"SimpleLoopExample(10).GetMultiplicationOfNumbers",
				"has number", "GetMultiplicationOfNumbers Number",
				"\tmutable result = 1", "\tconstant multiplier = 2", "\tfor number",
				"\t\tresult = result * multiplier", "\tresult")).Generate();
		var loaded = RoundTripToInstructions("SimpleLoopExample", instructions);
		Assert.That(loaded.Count, Is.EqualTo(instructions.Count));
		Assert.That(loaded.ConvertAll(x => x.ToString()),
			Is.EqualTo(instructions.ConvertAll(x => x.ToString())));
	}

	[Test]
	public void RoundTripConditionalBytecode()
	{
		var instructions = new BinaryGenerator(
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
		var loaded = RoundTripToInstructions("ArithmeticFunction", instructions);
		Assert.That(loaded.Count, Is.EqualTo(instructions.Count));
		Assert.That(loaded.ConvertAll(x => x.ToString()),
			Is.EqualTo(instructions.ConvertAll(x => x.ToString())));
	}

	[Test]
	public void RoundTripListBytecode()
	{
		var instructions = new BinaryGenerator(
			GenerateMethodCallFromSource("SimpleListDeclaration",
				"SimpleListDeclaration(5).Declare",
				"has number", "Declare Numbers", "\t(1, 2, 3, 4, 5)")).Generate();
		var loaded = RoundTripToInstructions("SimpleListDeclaration", instructions);
		Assert.That(loaded.Count, Is.EqualTo(instructions.Count));
		Assert.That(loaded.ConvertAll(x => x.ToString()),
			Is.EqualTo(instructions.ConvertAll(x => x.ToString())));
	}

	[Test]
	public void SavedFileIsZipWithCorrectMagicInEntry()
	{
		var binaryFilePath = new BytecodeSerializer(
			new Dictionary<string, IList<Instruction>>
			{
				["test"] = new List<Instruction> { new ReturnInstruction(Register.R0) }
			}, Path.GetTempPath(), "test" + testFileCounter++).OutputFilePath;
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
		Assert.Throws<BytecodeDeserializer.InvalidBytecodeFileException>(() =>
			new BytecodeDeserializer(binaryFilePath).Deserialize(TestPackage.Instance));
	}

	[Test]
	public void SerializedEntryContentIsCompact()
	{
		var instructions = new BinaryGenerator(
			GenerateMethodCallFromSource("Add", "Add(10, 5).Calculate",
				"has First Number", "has Second Number", "Calculate Number",
				"\tAdd(10, 5).Calculate is 15", "\tFirst + Second")).Generate();
		var entryBytes = SerializeToMemory("Add", instructions)["Add"];
		Assert.That(entryBytes.Length, Is.LessThan(60),
			"Serialized arithmetic bytecode entry should be compact (< 60 bytes)");
	}

	[Test]
	public void RoundTripSetInstruction()
	{
		var instructions = new List<Instruction>
		{
	public void RoundTripSetInstruction() =>
		AssertRoundTripInstructionTypes([
			new SetInstruction(Number(42), Register.R0),
			new ReturnInstruction(Register.R0)
		};
		AssertRoundTrip(instructions);
	}
		]);

	private static void AssertRoundTrip(IList<Instruction> instructions, string typeName = "main")
	{
		var loaded = RoundTripToInstructions(typeName, instructions);
		Assert.That(loaded.Count, Is.EqualTo(instructions.Count));
		for (var index = 0; index < instructions.Count; index++)
			Assert.That(loaded[index].InstructionType, Is.EqualTo(instructions[index].InstructionType));
	}

	[Test]
	public void RoundTripStoreFromRegisterInstruction()
	{
		var instructions = new List<Instruction>
		{
	public void RoundTripStoreFromRegisterInstruction() =>
		AssertRoundTripInstructionTypes([
			new StoreFromRegisterInstruction(Register.R1, "result"),
			new ReturnInstruction(Register.R0)
		};
		AssertRoundTrip(instructions);
	}
		]);

	[Test]
	public void RoundTripLoadVariableToRegister()
	{
		var instructions = new List<Instruction>
		{
			new StoreVariableInstruction(Number(5), "count"),
			new LoadVariableToRegister(Register.R0, "count"),
			new ReturnInstruction(Register.R0)
		};
		AssertRoundTrip(instructions);
	}

	[Test]
	public void RoundTripAllBinaryOperators()
	{
		var instructions = new List<Instruction>
		{
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
		};
		var loaded = RoundTripToInstructions("BinaryOps", instructions);
		Assert.That(loaded.Count, Is.EqualTo(instructions.Count));
		for (var index = 0; index < instructions.Count; index++)
			Assert.That(loaded[index].InstructionType, Is.EqualTo(instructions[index].InstructionType));
	}
		]);

	[Test]
	public void RoundTripJumpInstructions()
	{
		var instructions = new List<Instruction>
		{
	public void RoundTripJumpInstructions() =>
		AssertRoundTripInstructionTypes([
			new Jump(3),
			new Jump(2, InstructionType.JumpIfTrue),
			new Jump(1, InstructionType.JumpIfFalse),
			new JumpIfNotZero(5, Register.R0),
			new JumpToId(InstructionType.JumpEnd, 10),
			new JumpToId(InstructionType.JumpToIdIfFalse, 20),
			new JumpToId(InstructionType.JumpToIdIfTrue, 30),
			new ReturnInstruction(Register.R0)
		};
		var loaded = RoundTripToInstructions("Jumps", instructions);
		Assert.That(loaded.Count, Is.EqualTo(instructions.Count));
		for (var index = 0; index < instructions.Count; index++)
			Assert.That(loaded[index].InstructionType, Is.EqualTo(instructions[index].InstructionType));
	}

	[Test]
	public void RoundTripLoopBeginWithRange()
	{
		var instructions = new List<Instruction>
		{
			new LoopBeginInstruction(Register.R0, Register.R5),
			new LoopEndInstruction(3),
			new ReturnInstruction(Register.R0)
		};
		AssertRoundTrip(instructions);
	}

	[Test]
	public void RoundTripLoopBeginWithoutRange()
	{
		var instructions = new List<Instruction>
		{
			new LoopBeginInstruction(Register.R0),
			new LoopEndInstruction(2),
			new JumpToId(10, InstructionType.JumpEnd),
			new JumpToId(20, InstructionType.JumpToIdIfFalse),
			new JumpToId(30, InstructionType.JumpToIdIfTrue),
			new ReturnInstruction(Register.R0)
		};
		AssertRoundTrip(instructions);
	}
		]);

	[Test]
	public void RoundTripWriteToListInstruction()
	{
		var instructions = new List<Instruction>
		{
	public void RoundTripCollectionMutationInstructions() =>
		AssertRoundTripInstructionTypes([
			new WriteToListInstruction(Register.R0, "myList"),
			new ReturnInstruction(Register.R0)
		};
		AssertRoundTrip(instructions);
	}

	[Test]
	public void RoundTripWriteToTableInstruction()
	{
		var instructions = new List<Instruction>
		{
			new WriteToTableInstruction(Register.R0, Register.R1, "myTable"),
			new ReturnInstruction(Register.R0)
		};
		AssertRoundTrip(instructions);
	}

	[Test]
	public void RoundTripRemoveInstruction()
	{
		var instructions = new List<Instruction>
		{
			new RemoveInstruction("myList", Register.R0),
			new ReturnInstruction(Register.R0)
		};
		AssertRoundTrip(instructions);
	}

	[Test]
	public void RoundTripListCallInstruction()
	{
		var instructions = new List<Instruction>
		{
			new RemoveInstruction(Register.R2, "myList"),
			new ListCallInstruction(Register.R0, Register.R1, "myList"),
			new ReturnInstruction(Register.R0)
		};
		AssertRoundTrip(instructions);
	}
		]);

	[Test]
	public void RoundTripSmallNumberValue()
	{
		var instructions = new List<Instruction>
	public void RoundTripPrintInstructionWithNumberRegister()
	{
			new LoadConstantInstruction(Register.R0, Number(42)),
		var loaded = RoundTripInstructions([
			new PrintInstruction("Value = ", Register.R2),
			new ReturnInstruction(Register.R0)
		};
		AssertRoundTripValues(instructions);
	}

	private static void AssertRoundTripValues(IList<Instruction> instructions,
		string typeName = "main")
	{
		var loaded = RoundTripToInstructions(typeName, instructions);
		Assert.That(loaded.Count, Is.EqualTo(instructions.Count));
		Assert.That(loaded.ConvertAll(x => x.ToString()),
			Is.EqualTo(instructions.ToList().ConvertAll(x => x.ToString())));
		]);
		var print = (PrintInstruction)loaded[0];
		Assert.That(print.TextPrefix, Is.EqualTo("Value = "));
		Assert.That(print.ValueRegister, Is.EqualTo(Register.R2));
		Assert.That(print.ValueIsText, Is.False);
	}

	[Test]
	public void RoundTripIntegerNumberValue() =>
		AssertRoundTripValues(new List<Instruction>
		{
			new LoadConstantInstruction(Register.R0, Number(1000)),
			new ReturnInstruction(Register.R0)
		});

	[Test]
	public void RoundTripLargeDoubleNumberValue() =>
		AssertRoundTripValues(new List<Instruction>
		{
			new LoadConstantInstruction(Register.R0, Number(3.14159)),
			new ReturnInstruction(Register.R0)
		});

	[Test]
	public void RoundTripTextValue() =>
		AssertRoundTripValues(new List<Instruction>
		{
			new LoadConstantInstruction(Register.R0, Text("hello world")),
			new ReturnInstruction(Register.R0)
		});

	[Test]
	public void RoundTripBooleanValue() =>
		AssertRoundTripValues(new List<Instruction>
		{
			new LoadConstantInstruction(Register.R0, new ValueInstance(boolType, true)),
			new LoadConstantInstruction(Register.R1, new ValueInstance(boolType, false)),
			new ReturnInstruction(Register.R0)
		});

	private readonly Type boolType = TestPackage.Instance.GetType(Type.Boolean);

	[Test]
	public void RoundTripNoneValue() =>
		AssertRoundTripValues(new List<Instruction>
		{
			new StoreVariableInstruction(new ValueInstance(noneType), "nothing"),
			new ReturnInstruction(Register.R0)
		});

	private readonly Type noneType = TestPackage.Instance.GetType(Type.None);

	[Test]
	public void RoundTripListValue() =>
		AssertRoundTripValues(new List<Instruction>
		{
			new LoadConstantInstruction(Register.R0,
				new ValueInstance(ListType, [Number(1), Number(2), Number(3)])),
			new ReturnInstruction(Register.R0)
		});

	[Test]
	public void RoundTripStoreVariableAsMember() =>
		AssertRoundTripValues(new List<Instruction>
		{
			new StoreVariableInstruction(Number(10), "First", isMember: true),
			new StoreVariableInstruction(Number(20), "Second", isMember: true),
			new ReturnInstruction(Register.R0)
		});

	[Test]
	public void RoundTripInvokeWithMethodCallExpressions() =>
		AssertRoundTripValues(
			new BinaryGenerator(GenerateMethodCallFromSource("Greeter", "Greeter(\"world\").Greet",
				"has text Text", "Greet Text", "\tGreeter(\"world\").Greet is \"hello world\"",
				"\t\"hello \" + text")).Generate(), "Greeter");

	[Test]
	public void RoundTripNegativeIntegerValue() =>
		AssertRoundTripValues(new List<Instruction>
		{
			new LoadConstantInstruction(Register.R0, Number(-500)), new ReturnInstruction(Register.R0)
		});

	[Test]
	public void RoundTripMultipleTypesInSingleZip()
	{
		var addInstructions = new List<Instruction>
		{
			new LoadConstantInstruction(Register.R0, Number(10)),
			new ReturnInstruction(Register.R0)
		};
		var subInstructions = new List<Instruction>
		{
			new LoadConstantInstruction(Register.R0, Number(5)),
			new ReturnInstruction(Register.R0)
		};
		var deserializer = new BytecodeDeserializer(BytecodeSerializer.SerializeToEntryBytes(
			new Dictionary<string, IList<Instruction>>
			{
				["TypeA"] = addInstructions,
				["TypeB"] = subInstructions
			}), TestPackage.Instance);
		Assert.That(deserializer.Instructions!, Has.Count.EqualTo(2));
		Assert.That(deserializer.Instructions!["TypeA"], Has.Count.EqualTo(2));
		Assert.That(deserializer.Instructions!["TypeB"], Has.Count.EqualTo(2));
	}

	[Test]
	public void RoundTripDictionaryValue()
	{
		var dictType = TestPackage.Instance.GetDictionaryImplementationType(NumberType, NumberType);
		var dictionaryType = TestPackage.Instance.GetDictionaryImplementationType(NumberType, NumberType);
		var items = new Dictionary<ValueInstance, ValueInstance>
		{
			{ Number(1), Number(10) },
			{ Number(2), Number(20) }
		};
		var instructions = new List<Instruction>
		{
			new LoadConstantInstruction(Register.R0, new ValueInstance(dictType, items)),
		var loaded = RoundTripInstructions([
			new LoadConstantInstruction(Register.R0, new ValueInstance(dictionaryType, items)),
			new ReturnInstruction(Register.R0)
		]);
		var loaded = RoundTripToInstructions("DictTest", instructions);
		Assert.That(loaded.Count, Is.EqualTo(instructions.Count));
		var loadedDict = ((LoadConstantInstruction)loaded[0]).ValueInstance;
		Assert.That(loadedDict.IsDictionary, Is.True);
		var loadedItems = loadedDict.GetDictionaryItems();
		Assert.That(loadedItems.Count, Is.EqualTo(2));
		Assert.That(loadedItems[Number(1)].Number, Is.EqualTo(10));
		Assert.That(loadedItems[Number(2)].Number, Is.EqualTo(20));
		]);
		var loadedDictionary = ((LoadConstantInstruction)loaded[0]).Constant;
		Assert.That(loadedDictionary.IsDictionary, Is.True);
		Assert.That(loadedDictionary.GetDictionaryItems()[Number(1)].Number, Is.EqualTo(10));
	}

	[Test]
	public void ZipContainsNoBytecodeSourceEntries()
	private static void AssertRoundTripInstructionTypes(IList<Instruction> instructions)
	{
		var instructions = new List<Instruction> { new ReturnInstruction(Register.R0) };
		var binaryFilePath = SerializeToTemp("CleanZip", instructions);
		using var zip = ZipFile.OpenRead(binaryFilePath);
		Assert.That(zip.Entries.All(entry => entry.Name.EndsWith(".bytecode",
			StringComparison.OrdinalIgnoreCase)), Is.True);
		var loaded = RoundTripInstructions(instructions);
		Assert.That(loaded.Count, Is.EqualTo(instructions.Count));
		for (var index = 0; index < instructions.Count; index++)
			Assert.That(loaded[index].InstructionType, Is.EqualTo(instructions[index].InstructionType));
	}

	[Test]
	public void RoundTripInvokeWithIntegerNumberArgument()
	{
		var instructions = new BinaryGenerator(
			GenerateMethodCallFromSource("LargeAdder", "LargeAdder(1000).GetSum",
				//@formatter off
				"has number",
				"GetSum Number",
				"\tLargeAdder(1000).GetSum is 1500",
				"\tAddOffset(500)",
				"AddOffset(offset Number) Number",
				"\tnumber + offset")).Generate();
		AssertRoundTripValues(instructions, "LargeAdder");
	}
	private static void AssertRoundTripToString(IList<Instruction> instructions) =>
		Assert.That(RoundTripInstructions(instructions).ConvertAll(instruction => instruction.ToString()),
			Is.EqualTo(instructions.ToList().ConvertAll(instruction => instruction.ToString())));

	[Test]
	public void RoundTripInvokeWithDoubleNumberArgument()
	private static List<Instruction> RoundTripInstructions(IList<Instruction> instructions)
	{
		var instructions = new BinaryGenerator(
			GenerateMethodCallFromSource("DoubleCalc", "DoubleCalc(3.14).GetHalf",
				"has number",
				"GetHalf Number",
				"\tHalve(number)",
				"Halve(value Number) Number",
				"\tvalue / 2")).Generate();
		AssertRoundTripValues(instructions, "DoubleCalc");
	}

	[Test]
	public void RoundTripInvokeWithBooleanArgument()
	{
		var instructions = new BinaryGenerator(
			GenerateMethodCallFromSource("BoolCheck", "BoolCheck(true).GetResult",
				"has flag Boolean",
				"GetResult Number",
				"\tSelectValue(flag)",
				"SelectValue(condition Boolean) Number",
				"\tcondition then 1 else 0")).Generate();
		AssertRoundTripValues(instructions, "BoolCheck");
	}

	[Test]
	public void RoundTripListMemberWithIteration()
	{
		var instructions = new BinaryGenerator(
			GenerateMethodCallFromSource("ListSum", "ListSum(1, 2, 3).Total",
				"has numbers",
				"Total Number",
				"\tListSum(1, 2, 3).Total is 6",
				"\tmutable sum = 0",
				"\tfor numbers",
				"\t\tsum = sum + value",
				"\tsum")).Generate();
		//@formatter on
		AssertRoundTripValues(instructions, "ListSum");
	}

	[Test]
	public void RoundTripDictionaryWriteAndRead()
	{
		var dictType = TestPackage.Instance.GetDictionaryImplementationType(NumberType, NumberType);
		var emptyDict = new Dictionary<ValueInstance, ValueInstance>();
		var instructions = new List<Instruction>
		{
			new StoreVariableInstruction(new ValueInstance(dictType, emptyDict), "table"),
			new LoadConstantInstruction(Register.R0, Number(1)),
			new LoadConstantInstruction(Register.R1, Number(10)),
			new WriteToTableInstruction(Register.R0, Register.R1, "table"),
			new LoadConstantInstruction(Register.R2, Number(2)),
			new LoadConstantInstruction(Register.R3, Number(20)),
			new WriteToTableInstruction(Register.R2, Register.R3, "table"),
			new ReturnInstruction(Register.R0)
		};
		AssertRoundTripValues(instructions, "DictOps");
	}

	[Test]
	public void RoundTripMethodWithParameters()
	{
		var instructions = new BinaryGenerator(
			GenerateMethodCallFromSource("Multiplier",
				"Multiplier(10).Scale(3)",
				"has number",
				"Scale(factor Number) Number",
				"\tMultiplier(10).Scale(3) is 30",
				"\tnumber * factor")).Generate();
		AssertRoundTripValues(instructions, "Multiplier");
	}

	[Test]
	public void MethodNotFoundThrows()
	{
		var entryBytes = CreateBytecodeWithUnknownOperator();
		Assert.Throws<BytecodeDeserializer.MethodNotFoundException>(() =>
			new BytecodeDeserializer(new Dictionary<string, byte[]> { ["main"] = entryBytes },
				TestPackage.Instance));
	}

	private static byte[] CreateBytecodeWithUnknownOperator()
	{
		using var stream = new MemoryStream();
		using var writer = new BinaryWriter(stream, Encoding.UTF8, leaveOpen: true);
		WriteEntryMagicAndVersion(writer);
		var names = new[] { "main", "Run", "None", "Number", "$$bogus$$" };
		WriteNameTable(writer, names);
		writer.Write7BitEncodedInt(0);
		writer.Write7BitEncodedInt(0);
		writer.Write7BitEncodedInt(1);
		writer.Write((byte)InstructionType.Invoke);
		writer.Write((byte)Register.R0);
		writer.Write(true);
		writer.Write7BitEncodedInt(0);
		writer.Write7BitEncodedInt(1);
		writer.Write7BitEncodedInt(0);
		writer.Write7BitEncodedInt(2);
		writer.Write(false);
		writer.Write7BitEncodedInt(1);
		writer.Write(BinaryExprKind);
		writer.Write7BitEncodedInt(4);
		writer.Write(SmallNumberKind);
		writer.Write((byte)1);
		writer.Write(SmallNumberKind);
		writer.Write((byte)2);
		writer.Write(false);
		writer.Write7BitEncodedInt(0);
		writer.Flush();
		return stream.ToArray();
	}

	[Test]
	public void EnsureResolvedTypeCreatesStubForUnknownType()
	{
		var entryBytes = CreateBytecodeWithCustomTypeName("UnknownStubType");
		var deserialized =
			new BytecodeDeserializer(new Dictionary<string, byte[]> { ["main"] = entryBytes },
				TestPackage.Instance);
		Assert.That(deserialized.Instructions!.Values.First(), Has.Count.EqualTo(1));
	}

	private static byte[] CreateBytecodeWithCustomTypeName(string typeName)
	{
		using var stream = new MemoryStream();
		using var writer = new BinaryWriter(stream, Encoding.UTF8, leaveOpen: true);
		WriteEntryMagicAndVersion(writer);
		var names = new[] { "main", "Run", typeName, "None" };
		WriteNameTable(writer, names);
		writer.Write7BitEncodedInt(0);
		writer.Write7BitEncodedInt(0);
		writer.Write7BitEncodedInt(1);
		writer.Write((byte)InstructionType.Invoke);
		writer.Write((byte)Register.R0);
		writer.Write(true);
		writer.Write7BitEncodedInt(0);
		writer.Write7BitEncodedInt(3);
		writer.Write7BitEncodedInt(2);
		writer.Write7BitEncodedInt(4);
		writer.Write(false);
		writer.Write7BitEncodedInt(2);
		writer.Write(SmallNumberKind);
		writer.Write((byte)1);
		writer.Write(SmallNumberKind);
		writer.Write((byte)2);
		writer.Write(false);
		writer.Write7BitEncodedInt(0);
		writer.Flush();
		return stream.ToArray();
	}

	[Test]
	public void BuildMethodHeaderWithParametersCreatesMethod()
	{
		var entryBytes = CreateBytecodeWithMethodParameters(2);
		var deserialized =
			new BytecodeDeserializer(new Dictionary<string, byte[]> { ["main"] = entryBytes },
				TestPackage.Instance);
		Assert.That(deserialized.Instructions!.Values.First(), Has.Count.EqualTo(1));
	}

	private static byte[] CreateBytecodeWithMethodParameters(int paramCount)
	{
		using var stream = new MemoryStream();
		using var writer = new BinaryWriter(stream, Encoding.UTF8, leaveOpen: true);
		WriteEntryMagicAndVersion(writer);
		var names = new[] { "Main", "Run", "None", "Compute", "Number" };
		WriteNameTable(writer, names);
		writer.Write7BitEncodedInt(0);
		writer.Write7BitEncodedInt(0);
		writer.Write7BitEncodedInt(1);
		writer.Write((byte)InstructionType.Invoke);
		writer.Write((byte)Register.R0);
		writer.Write(true);
		writer.Write7BitEncodedInt(0);
		writer.Write7BitEncodedInt(3);
		writer.Write7BitEncodedInt(paramCount);
		writer.Write7BitEncodedInt(4);
		writer.Write(false);
		writer.Write7BitEncodedInt(paramCount);
		for (var index = 0; index < paramCount; index++)
		{
			writer.Write(SmallNumberKind);
			writer.Write((byte)(index + 1));
		}
		writer.Write(false);
		writer.Write7BitEncodedInt(0);
		writer.Flush();
		return stream.ToArray();
	}

	[Test]
	public void TypeNotFoundForLowercaseThrows()
	{
		var entryBytes = CreateBytecodeWithCustomTypeName("lowercase");
		Assert.Throws<BytecodeDeserializer.TypeNotFoundForBytecode>(() =>
			new BytecodeDeserializer(new Dictionary<string, byte[]> { ["main"] = entryBytes },
				TestPackage.Instance));
	}

	[Test]
	public void InvalidVersionThrows()
	{
		using var stream = new MemoryStream();
		using var writer = new BinaryWriter(stream, Encoding.UTF8, leaveOpen: true);
		writer.Write(Encoding.UTF8.GetBytes("Strict"));
		writer.Write((byte)0);
		using var writer = new BinaryWriter(stream);
		var table = new NameTable();
		foreach (var instruction in instructions)
			table.CollectStrings(instruction);
		table.Write(writer);
		writer.Write7BitEncodedInt(instructions.Count);
		foreach (var instruction in instructions)
			instruction.Write(writer, table);
		writer.Flush();
		Assert.Throws<BytecodeDeserializer.InvalidVersion>(() =>
			new BytecodeDeserializer(new Dictionary<string, byte[]> { ["main"] = stream.ToArray() },
				TestPackage.Instance));
	}

	private static void WriteEntryMagicAndVersion(BinaryWriter writer)
	{
		writer.Write(Encoding.UTF8.GetBytes("Strict"));
		writer.Write(BytecodeSerializer.Version);
	}

	[Test]
	public void RoundTripPrintInstruction()
	{
		var instructions = new List<Instruction>
		{
			new PrintInstruction("Hello World"),
			new ReturnInstruction(Register.R0)
		};
		var loaded = RoundTripToInstructions("TestType", instructions);
		Assert.That(loaded.Count, Is.EqualTo(2));
		Assert.That(loaded[0], Is.InstanceOf<PrintInstruction>());
		var print = (PrintInstruction)loaded[0];
		Assert.That(print.TextPrefix, Is.EqualTo("Hello World"));
		Assert.That(print.ValueRegister, Is.Null);
	}

	[Test]
	public void RoundTripPrintInstructionWithNumberRegister()
	{
		var instructions = new List<Instruction>
		{
			new PrintInstruction("Value = ", Register.R2),
			new ReturnInstruction(Register.R0)
		};
		var loaded = RoundTripToInstructions("PrintTypePair", instructions);
		Assert.That(loaded.Count, Is.EqualTo(2));
		var print = (PrintInstruction)loaded[0];
		Assert.That(print.TextPrefix, Is.EqualTo("Value = "));
		Assert.That(print.ValueRegister, Is.EqualTo(Register.R2));
		Assert.That(print.ValueIsText, Is.False);
	}

	private static void WriteNameTable(BinaryWriter writer, string[] names)
	{
		writer.Write7BitEncodedInt(names.Length);
		foreach (var name in names)
			writer.Write(name);
		stream.Position = 0;
		using var reader = new BinaryReader(stream);
		var readTable = new NameTable(reader);
		var count = reader.Read7BitEncodedInt();
		var binary = new Binary(TestPackage.Instance);
		var loaded = new List<Instruction>(count);
		for (var index = 0; index < count; index++)
			loaded.Add(binary.ReadInstruction(reader, readTable));
		return loaded;
	}

	private const byte SmallNumberKind = 0;
	private const byte BinaryExprKind = 7;
	private readonly Type boolType = TestPackage.Instance.GetType(Type.Boolean);
} */