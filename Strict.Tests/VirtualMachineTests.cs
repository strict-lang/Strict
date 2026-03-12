using System.Globalization;
using Strict.Bytecode;
using Strict.Bytecode.Instructions;
using Strict.Bytecode.Tests;
using Strict.Expressions;
using Strict.Language;
using Strict.Language.Tests;
using Type = Strict.Language.Type;

namespace Strict.Tests;

public sealed class VirtualMachineTests : TestBytecode
{
	[SetUp]
	public void Setup() => vm = new VirtualMachine(TestPackage.Instance);

	private VirtualMachine vm = null!;

	private void CreateSampleEnum()
	{
		if (type.Package.FindDirectType("Days") == null)
			new Type(type.Package,
					new TypeLines("Days", "constant Monday = 1", "constant Tuesday = 2",
						"constant Wednesday = 3", "constant Friday = 5")).
				ParseMembersAndMethods(new MethodExpressionParser());
	}

	[Test]
	public void ReturnEnum()
	{
		CreateSampleEnum();
		var instructions = new BytecodeGenerator(GenerateMethodCallFromSource(nameof(ReturnEnum),
			nameof(ReturnEnum) + "(5).GetMonday", "has dummy Number", "GetMonday Number",
			"\tDays.Monday")).Generate();
		var result = vm.Execute(instructions).Returns;
		Assert.That(result!.Value.Number, Is.EqualTo(1));
	}

	[Test]
	public void EnumIfConditionComparison()
	{
		CreateSampleEnum();
		var instructions = new BytecodeGenerator(GenerateMethodCallFromSource(
			nameof(EnumIfConditionComparison),
			nameof(EnumIfConditionComparison) + "(5).GetMonday(Days.Monday)",
			// @formatter:off
			"has dummy Number",
			"GetMonday(days) Boolean",
			"\tif days is Days.Monday",
			"\t\treturn true",
			"\telse",
			"\t\treturn false")).Generate();
		// @formatter:on
		var result = vm.Execute(instructions).Returns!;
		Assert.That(result.Value.Number, Is.EqualTo(1));
	}

	[TestCase(InstructionType.Add, 15, 5, 10)]
	[TestCase(InstructionType.Subtract, 5, 8, 3)]
	[TestCase(InstructionType.Multiply, 4, 2, 2)]
	[TestCase(InstructionType.Divide, 3, 7.5, 2.5)]
	[TestCase(InstructionType.Modulo, 1, 5, 2)]
	[TestCase(InstructionType.Add, "510", "5", 10)]
	[TestCase(InstructionType.Add, "510", 5, "10")]
	[TestCase(InstructionType.Add, "510", "5", "10")]
	public void Execute(InstructionType operation, object expected, params object[] inputs)
	{
		var result = vm.Execute(BuildInstructions(inputs, operation)).Memory.Registers[Register.R1];
		var actual = expected is string
			? (object)result.Text
			: result.Number;
		Assert.That(actual, Is.EqualTo(expected));
	}

	private static Instruction[] BuildInstructions(IReadOnlyList<object> inputs,
		InstructionType operation) =>
	[
		new SetInstruction(inputs[0] is string s0
			? Text(s0)
			: Number(Convert.ToDouble(inputs[0])), Register.R0),
		new SetInstruction(inputs[1] is string s1
			? Text(s1)
			: inputs[1] is double d
				? Number(d)
				: Number(Convert.ToDouble(inputs[1])), Register.R1),
		new BinaryInstruction(operation, Register.R0, Register.R1)
	];

	[Test]
	public void LoadVariable() =>
		Assert.That(vm.Execute([
			new LoadConstantInstruction(Register.R0, Number(5))
		]).Memory.Registers[Register.R0].Number, Is.EqualTo(5));

	[Test]
	public void SetAndAdd() =>
		Assert.That(vm.Execute([
			new LoadConstantInstruction(Register.R0, Number(10)),
			new LoadConstantInstruction(Register.R1, Number(5)),
			new BinaryInstruction(InstructionType.Add, Register.R0, Register.R1, Register.R2)
		]).Memory.Registers[Register.R2].Number, Is.EqualTo(15));

	[Test]
	public void AddFiveTimes() =>
		Assert.That(vm.Execute([
			new SetInstruction(Number(5), Register.R0),
			new SetInstruction(Number(1), Register.R1),
			new SetInstruction(Number(0), Register.R2),
			new BinaryInstruction(InstructionType.Add, Register.R0, Register.R2, Register.R2),
			new BinaryInstruction(InstructionType.Subtract, Register.R0, Register.R1, Register.R0),
			new JumpIfNotZero(-3, Register.R0)
		]).Memory.Registers[Register.R2].Number, Is.EqualTo(0 + 5 + 4 + 3 + 2 + 1));

	[TestCase("ArithmeticFunction(10, 5).Calculate(\"add\")", 15)]
	[TestCase("ArithmeticFunction(10, 5).Calculate(\"subtract\")", 5)]
	[TestCase("ArithmeticFunction(10, 5).Calculate(\"multiply\")", 50)]
	[TestCase("ArithmeticFunction(10, 5).Calculate(\"divide\")", 2)]
	public void RunArithmeticFunctionExample(string methodCall, int expectedResult)
	{
		var instructions = new BytecodeGenerator(GenerateMethodCallFromSource("ArithmeticFunction",
			methodCall,
			// @formatter:off
			"has First Number",
			"has Second Number",
			"Calculate(operation Text) Number",
			"\tArithmeticFunction(10, 5).Calculate(\"add\") is 15",
			"\tArithmeticFunction(10, 5).Calculate(\"subtract\") is 5",
			"\tArithmeticFunction(10, 5).Calculate(\"multiply\") is 50",
			"\tif operation is \"add\"",
			"\t\treturn First + Second",
			"\tif operation is \"subtract\"",
			"\t\treturn First - Second",
			"\tif operation is \"multiply\"",
			"\t\treturn First * Second",
			"\tif operation is \"divide\"",
			"\t\treturn First / Second")).Generate();
		// @formatter:on
		Assert.That(vm.Execute(instructions).Returns!.Value.Number, Is.EqualTo(expectedResult));
	}

	[Test]
	public void AccessListByIndex()
	{
		var instructions = new BytecodeGenerator(GenerateMethodCallFromSource(nameof(AccessListByIndex),
			nameof(AccessListByIndex) + "(1, 2, 3, 4, 5).Get(2)", "has numbers",
			"Get(index Number) Number", "\tnumbers(index)")).Generate();
		Assert.That(vm.Execute(instructions).Returns!.Value.Number, Is.EqualTo(3));
	}

	[Test]
	public void AccessListByIndexNonNumberType()
	{
		var instructions = new BytecodeGenerator(GenerateMethodCallFromSource(
			nameof(AccessListByIndexNonNumberType),
			nameof(AccessListByIndexNonNumberType) + "(\"1\", \"2\", \"3\", \"4\", \"5\").Get(2)",
			"has texts", "Get(index Number) Text", "\ttexts(index)")).Generate();
		Assert.That(vm.Execute(instructions).Returns!.Value.Text, Is.EqualTo("3"));
	}

	[Test]
	public void ReduceButGrowLoopExample() =>
		Assert.That(vm.Execute([
			new StoreVariableInstruction(Number(10), "number"),
			new StoreVariableInstruction(Number(1), "result"),
			new StoreVariableInstruction(Number(2), "multiplier"),
			new LoadVariableToRegister(Register.R0, "number"),
			new LoopBeginInstruction(Register.R0),
			new LoadVariableToRegister(Register.R2, "result"),
			new LoadVariableToRegister(Register.R3, "multiplier"),
			new BinaryInstruction(InstructionType.Multiply, Register.R2, Register.R3,
				Register.R4),
			new StoreFromRegisterInstruction(Register.R4, "result"),
			new LoopEndInstruction(5),
			new LoadVariableToRegister(Register.R5, "result"),
			new ReturnInstruction(Register.R5)
		]).Returns!.Value.Number, Is.EqualTo(1024));

	[TestCase("NumberConvertor", "NumberConvertor(5).ConvertToText", "5", "has number",
		"ConvertToText Text", "\t5 to Text")]
	[TestCase("TextConvertor", "TextConvertor(\"5\").ConvertToNumber", 5, "has text",
		"ConvertToNumber Number", "\ttext to Number")]
	public void ExecuteToOperator(string programName, string methodCall, object expected,
		params string[] code)
	{
		var instructions = new BytecodeGenerator(GenerateMethodCallFromSource(programName,
			methodCall, code)).Generate();
		var result = vm.Execute(instructions).Returns!.Value;
		var actual = expected is string
			? (object)result.Text
			: result.Number;
		Assert.That(actual, Is.EqualTo(expected));
	}

	//ncrunch: no coverage start
	private static IEnumerable<TestCaseData> MethodCallTests
	{
		get
		{
			// @formatter:off
			yield return new TestCaseData("AddNumbers", "AddNumbers(2, 5).GetSum",
				(string[])[
					"has firstNumber Number",
					"has secondNumber Number",
					"GetSum Number",
					"\tSumNumbers(firstNumber, secondNumber)",
					"SumNumbers(fNumber Number, sNumber Number) Number",
					"\tfNumber + sNumber"
				], 7);
			yield return new TestCaseData("CallWithConstants", "CallWithConstants(2, 5).GetSum",
				(string[])[
					"has firstNumber Number",
					"has secondNumber Number",
					"GetSum Number",
					"\tSumNumbers(5, 1)",
					"SumNumbers(fNumber Number, sNumber Number) Number",
					"\tfNumber + sNumber"
				], 6);
			yield return new TestCaseData("CallWithoutArguments", "CallWithoutArguments(2, 5).GetSum",
				(string[])[
					"has firstNumber Number",
					"has secondNumber Number",
					"GetSum Number",
					"\tSumNumbers",
					"SumNumbers Number",
					"\t10 + 532"
				], 542);
			yield return new TestCaseData("CurrentlyFailing", "CurrentlyFailing(10).SumEvenNumbers",
				(string[])[
					"has number",
					"SumEvenNumbers Number",
					"\tComputeSum",
					"ComputeSum Number",
					"\tmutable sum = 0",
					"\tfor number",
					"\t\tif index % 2 is 0",
					"\t\t\tsum = sum + index",
					"\tsum"
				], 20);
			// @formatter:on
		}
	} //ncrunch: no coverage end

	[TestCaseSource(nameof(MethodCallTests))]
	// ReSharper disable TooManyArguments, makes below tests easier
	public void MethodCall(string programName, string methodCall, string[] source, object expected)
	{
		var instructions =
			new BytecodeGenerator(GenerateMethodCallFromSource(programName, methodCall, source)).
				Generate();
		Assert.That(vm.Execute(instructions).Returns!.Value.Number, Is.EqualTo(expected));
	}

	[Test]
	public void IfAndElseTest()
	{
		var instructions = new BytecodeGenerator(GenerateMethodCallFromSource("IfAndElseTest",
				"IfAndElseTest(3).IsEven",
				//
				"has number", "IsEven Text", "\tmutable result = \"\"",
				"\tif number > 10", "\t\tresult = \"Number is more than 10\"", "\t\treturn result",
				"\telse", "\t\tresult = \"Number is less or equal than 10\"", "\t\treturn result")).
			Generate();
		Assert.That(vm.Execute(instructions).Returns!.Value.Text,
			Is.EqualTo("Number is less or equal than 10"));
	}

	[TestCase("EvenSumCalculator(100).IsEven", 2450, "EvenSumCalculator",
		new[]
		{
			"has number", "IsEven Number", "\tmutable sum = 0", "\tfor number",
			"\t\tif index % 2 is 0", "\t\t\tsum = sum + index", "\tsum"
		})]
	[TestCase("EvenSumCalculatorForList(100, 200, 300).IsEvenList", 2, "EvenSumCalculatorForList",
		new[]
		{
			"has numbers", "IsEvenList Number", "\tmutable sum = 0", "\tfor numbers",
			"\t\tif index % 2 is 0", "\t\t\tsum = sum + index", "\tsum"
		})]
	public void CompileCompositeBinariesInIfCorrectlyWithModulo(string methodCall,
		object expectedResult, string methodName, params string[] code)
	{
		var instructions =
			new BytecodeGenerator(GenerateMethodCallFromSource(methodName, methodCall, code)).
				Generate();
		Assert.That(vm.Execute(instructions).Returns!.Value.Number, Is.EqualTo(expectedResult));
	}

	[TestCase("AddToTheList(5).Add", "100 200 300 400 0 1 2 3", "AddToTheList",
		new[]
		{
			"has number", "Add Numbers", "\tmutable myList = (100, 200, 300, 400)", "\tfor myList",
			"\t\tif value % 2 is 0", "\t\t\tmyList = myList + index", "\tmyList"
		})]
	[TestCase("RemoveFromTheList(5).Remove", "100 200 300", "RemoveFromTheList",
		new[]
		{
			"has number", "Remove Numbers", "\tmutable myList = (100, 200, 300, 400)", "\tfor myList",
			"\t\tif value is 400", "\t\t\tmyList = myList - 400", "\tmyList"
		})]
	[TestCase("RemoveB(\"s\", \"b\", \"s\").Remove", "s s", "RemoveB",
		new[]
		{
			"has texts", "Remove Texts", "\tmutable textList = texts", "\tfor texts",
			"\t\tif value is \"b\"", "\t\t\ttextList = textList - value", "\ttextList"
		})]
	[TestCase("ListRemove(\"s\", \"b\", \"s\").Remove", "s s", "ListRemove",
		new[]
		{
			"has texts", "Remove Texts", "\tmutable textList = texts", "\ttextList.Remove(\"b\")",
			"\ttextList"
		})]
	[TestCase("ListRemoveMultiple(\"s\", \"b\", \"s\").Remove", "b", "ListRemoveMultiple",
		new[]
		{
			"has texts", "Remove Texts", "\tmutable textList = texts", "\ttextList.Remove(\"s\")",
			"\ttextList"
		})]
	public void ExecuteListBinaryOperations(string methodCall, object expectedResult,
		string programName, params string[] code)
	{
		var instructions = new BytecodeGenerator(GenerateMethodCallFromSource(programName,
			methodCall, code)).Generate();
		var result = vm.Execute(instructions).Returns!.Value;
		var elements = result.List.Items.Aggregate("", (current, item) => current + (item.IsText
			? item.Text
			: item.Number) + " ");
		Assert.That(elements.Trim(), Is.EqualTo(expectedResult));
	}

	[TestCase("TestContains(\"s\", \"b\", \"s\").Contains(\"b\")", "true", "TestContains",
		new[]
		{
			"has elements Texts", "Contains(other Text) Boolean", "\tfor elements",
			"\t\tif value is other", "\t\t\treturn true", "\tfalse"
		})]
	public void CallCommonMethodCalls(string methodCall, object expectedResult, string programName,
		params string[] code)
	{
		var instructions = new BytecodeGenerator(GenerateMethodCallFromSource(programName,
			methodCall, code)).Generate();
		var result = vm.Execute(instructions).Returns!.Value;
		Assert.That(result.ToExpressionCodeString(), Is.EqualTo(expectedResult));
	}

	[TestCase("NumbersAdder(5).AddNumberToList", "1 2 3 5", "has number", "AddNumberToList Numbers",
		"\tmutable numbers = (1, 2, 3)", "\tnumbers.Add(number)", "\tnumbers")]
	public void CollectionAdd(string methodCall, string expected, params string[] code)
	{
		var instructions =
			new BytecodeGenerator(GenerateMethodCallFromSource("NumbersAdder", methodCall, code)).
				Generate();
		var result = ExpressionListToSpaceSeparatedString(instructions);
		Assert.That(result.TrimEnd(), Is.EqualTo(expected));
	}

	private string ExpressionListToSpaceSeparatedString(IList<Instruction> instructions)
	{
		var result = vm.Execute(instructions).Returns!.Value;
		return result.List.Items.Aggregate("", (current, item) => current + (item.IsText
			? item.Text
			: item.Number) + " ");
	}

	[Test]
	public void DictionaryAdd()
	{
		string[] code =
		[
			"has number",
			"RemoveFromDictionary Number",
			"\tmutable values = Dictionary(Number, Number)", "\tvalues.Add(1, number)", "\tnumber"
		];
		Assert.That(
			vm.Execute(new BytecodeGenerator(GenerateMethodCallFromSource(nameof(DictionaryAdd),
					"DictionaryAdd(5).RemoveFromDictionary", code)).Generate()).Memory.Variables["values"].
				GetDictionaryItems().Count, Is.EqualTo(1));
	}

	[Test]
	public void CreateEmptyDictionaryFromConstructor()
	{
		var dictionaryType = TestPackage.Instance.GetType(Type.Dictionary).
			GetGenericImplementation(NumberType, NumberType);
		var methodCall = CreateFromMethodCall(dictionaryType);
		var instructions = new List<Instruction> { new Invoke(Register.R0, methodCall, new Registry()) };
		var result = vm.Execute(instructions).Memory.Registers[Register.R0];
		Assert.That(result.IsDictionary, Is.True);
		Assert.That(result.GetDictionaryItems().Count, Is.EqualTo(0));
	}

	[TestCase("DictionaryGet(5).AddToDictionary", "5", "has number", "AddToDictionary Number",
		"\tmutable values = Dictionary(Number, Number)", "\tvalues.Add(1, number)",
		"\tvalues.Get(1)")]
	public void DictionaryGet(string methodCall, string expected, params string[] code)
	{
		var instructions =
			new BytecodeGenerator(
				GenerateMethodCallFromSource(nameof(DictionaryGet), methodCall, code)).Generate();
		var result = vm.Execute(instructions).Returns!.Value;
		var actual = result.IsText
			? result.Text
			: result.Number.ToString(CultureInfo.InvariantCulture);
		Assert.That(actual, Is.EqualTo(expected));
	}

	[TestCase("DictionaryRemove(5).AddToDictionary", "5", "has number", "AddToDictionary Number",
		"\tmutable values = Dictionary(Number, Number)", "\tvalues.Add(1, number)",
		"\tvalues.Add(2, number + 10)", "\tvalues.Get(2)")]
	public void DictionaryRemove(string methodCall, string expected, params string[] code)
	{
		var instructions =
			new BytecodeGenerator(
				GenerateMethodCallFromSource(nameof(DictionaryRemove), methodCall, code)).Generate();
		var result = vm.Execute(instructions).Returns!.Value;
		var actual = result.IsText
			? result.Text
			: result.Number.ToString(CultureInfo.InvariantCulture);
		Assert.That(actual, Is.EqualTo("15"));
	}

	[Test]
	public void ReturnWithinALoop()
	{
		var source = new[] { "has number", "GetAll Number", "\tfor number", "\t\tvalue" };
		var instructions = new BytecodeGenerator(GenerateMethodCallFromSource(nameof(ReturnWithinALoop),
			"ReturnWithinALoop(5).GetAll", source)).Generate();
		Assert.That(() => vm.Execute(instructions).Returns!.Value.Number,
			Is.EqualTo(1 + 2 + 3 + 4 + 5));
	}

	[Test]
	public void ReverseWithRange()
	{
		var source = new[]
		{
			"has numbers", "Reverse Numbers", "\tmutable result = Numbers",
			"\tlet len = numbers.Length - 1", "\tfor Range(len, 0)", "\t\tresult.Add(numbers(index))",
			"\tresult"
		};
		var instructions = new BytecodeGenerator(GenerateMethodCallFromSource(nameof(ReverseWithRange),
			"ReverseWithRange(1, 2, 3).Reverse", source)).Generate();
		Assert.That(() => ExpressionListToSpaceSeparatedString(instructions), Is.EqualTo("3 2 1 "));
	}

	[Test]
	public void ConditionalJump() =>
		Assert.That(vm.Execute([
			new SetInstruction(Number(5), Register.R0),
			new SetInstruction(Number(1), Register.R1),
			new SetInstruction(Number(10), Register.R2),
			new BinaryInstruction(InstructionType.LessThan, Register.R2, Register.R0),
			new JumpIf(InstructionType.JumpIfTrue, 2),
			new BinaryInstruction(InstructionType.Add, Register.R2, Register.R0, Register.R0)
		]).Memory.Registers[Register.R0].Number, Is.EqualTo(15));

	[Test]
	public void JumpIfTrueSkipsNextInstruction() =>
		Assert.That(vm.Execute([
			new SetInstruction(Number(1), Register.R0),
			new SetInstruction(Number(1), Register.R1),
			new SetInstruction(Number(0), Register.R2),
			new BinaryInstruction(InstructionType.Equal, Register.R0, Register.R1),
			new JumpIfTrue(1, Register.R0),
			new BinaryInstruction(InstructionType.Add, Register.R0, Register.R1, Register.R2)
		]).Memory.Registers[Register.R2].Number, Is.EqualTo(0));

	[Test]
	public void JumpIfFalseSkipsNextInstruction() =>
		Assert.That(vm.Execute([
			new SetInstruction(Number(1), Register.R0),
			new SetInstruction(Number(2), Register.R1),
			new SetInstruction(Number(0), Register.R2),
			new BinaryInstruction(InstructionType.Equal, Register.R0, Register.R1),
			new JumpIfFalse(1, Register.R0),
			new BinaryInstruction(InstructionType.Add, Register.R0, Register.R1, Register.R2)
		]).Memory.Registers[Register.R2].Number, Is.EqualTo(0));

	[TestCase(InstructionType.GreaterThan, new[] { 1, 2 }, 2 - 1)]
	[TestCase(InstructionType.LessThan, new[] { 1, 2 }, 1 + 2)]
	[TestCase(InstructionType.Equal, new[] { 5, 5 }, 5 + 5)]
	[TestCase(InstructionType.NotEqual, new[] { 5, 5 }, 5 - 5)]
	public void ConditionalJumpIfAndElse(InstructionType conditional, int[] registers,
		int expected) =>
		Assert.That(vm.Execute([
			new SetInstruction(Number(registers[0]), Register.R0),
			new SetInstruction(Number(registers[1]), Register.R1),
			new BinaryInstruction(conditional, Register.R0, Register.R1),
			new JumpIf(InstructionType.JumpIfTrue, 2),
			new BinaryInstruction(InstructionType.Subtract, Register.R1, Register.R0, Register.R0),
			new JumpIf(InstructionType.JumpIfFalse, 2),
			new BinaryInstruction(InstructionType.Add, Register.R0, Register.R1, Register.R0)
		]).Memory.Registers[Register.R0].Number, Is.EqualTo(expected));

	[TestCase(InstructionType.Add)]
	[TestCase(InstructionType.GreaterThan)]
	public void OperandsRequired(InstructionType instruction) =>
		Assert.That(() => vm.Execute([new BinaryInstruction(instruction, Register.R0)]),
			Throws.InstanceOf<VirtualMachine.OperandsRequired>());

	[Test]
	public void LoopOverEmptyListSkipsBody()
	{
		var numbersListType = ListType.GetGenericImplementation(NumberType);
		var emptyList = new ValueInstance(numbersListType, Array.Empty<ValueInstance>());
		var result = vm.Execute([
			new StoreVariableInstruction(emptyList, "numbers"),
			new StoreVariableInstruction(Number(0), "result"),
			new LoadVariableToRegister(Register.R0, "numbers"),
			new LoopBeginInstruction(Register.R0),
			new LoadVariableToRegister(Register.R1, "result"),
			new LoadConstantInstruction(Register.R2, Number(1)),
			new BinaryInstruction(InstructionType.Add, Register.R1, Register.R2, Register.R3),
			new StoreFromRegisterInstruction(Register.R3, "result"),
			new LoopEndInstruction(5),
			new LoadVariableToRegister(Register.R4, "result"),
			new ReturnInstruction(Register.R4)
		]).Returns;
		Assert.That(result!.Value.Number, Is.EqualTo(0));
	}

	[Test]
	public void LoopOverTextStopsWhenIndexExceedsLength()
	{
		var text = Text("Hi");
		var loopBegin = new LoopBeginInstruction(Register.R0);
		var result = vm.Execute([
			new StoreVariableInstruction(text, "words"),
			new StoreVariableInstruction(Number(0), "count"),
			new LoadVariableToRegister(Register.R0, "words"),
			loopBegin,
			new LoadVariableToRegister(Register.R1, "count"),
			new LoadConstantInstruction(Register.R2, Number(1)),
			new BinaryInstruction(InstructionType.Add, Register.R1, Register.R2, Register.R3),
			new StoreFromRegisterInstruction(Register.R3, "count"),
			new LoopEndInstruction(5),
			new LoadVariableToRegister(Register.R4, "count"),
			new ReturnInstruction(Register.R4)
		]).Returns;
		Assert.That(result!.Value.Number, Is.EqualTo(2));
	}

	[Test]
	public void LoopOverListStopsWhenIndexExceedsCount()
	{
		var source = new[]
		{
			"has numbers",
			"CountItems Number",
			"\tmutable count = 0",
			"\tfor numbers",
			"\t\tcount = count + 1",
			"\tcount"
		};
		var instructions = new BytecodeGenerator(GenerateMethodCallFromSource(
			nameof(LoopOverListStopsWhenIndexExceedsCount),
			$"{nameof(LoopOverListStopsWhenIndexExceedsCount)}(1, 2, 3).CountItems", source)).Generate();
		var result = vm.Execute(instructions).Returns!.Value.Number;
		Assert.That(result, Is.EqualTo(3));
	}

	[Test]
	public void LoopOverSingleCharTextStopsAtEnd()
	{
		var source = new[]
		{
			"has letter Text",
			"CountChars Number",
			"\tmutable count = 0",
			"\tfor letter",
			"\t\tcount = count + 1",
			"\tcount"
		};
		var instructions = new BytecodeGenerator(GenerateMethodCallFromSource(
			nameof(LoopOverSingleCharTextStopsAtEnd),
			$"{nameof(LoopOverSingleCharTextStopsAtEnd)}(\"X\").CountChars", source)).Generate();
		var result = vm.Execute(instructions).Returns!.Value.Number;
		Assert.That(result, Is.EqualTo(1));
	}

	[Test]
	public void LoopOverSingleItemListStopsAtEnd()
	{
		var numbersListType = ListType.GetGenericImplementation(NumberType);
		var singleItemList = new ValueInstance(numbersListType, [new ValueInstance(NumberType, 42)]);
		var result = vm.Execute([
			new StoreVariableInstruction(singleItemList, "items"),
			new StoreVariableInstruction(Number(0), "count"),
			new LoadVariableToRegister(Register.R0, "items"),
			new LoopBeginInstruction(Register.R0),
			new LoadVariableToRegister(Register.R1, "count"),
			new LoadConstantInstruction(Register.R2, Number(1)),
			new BinaryInstruction(InstructionType.Add, Register.R1, Register.R2, Register.R3),
			new StoreFromRegisterInstruction(Register.R3, "count"),
			new LoopEndInstruction(5),
			new LoadVariableToRegister(Register.R4, "count"),
			new ReturnInstruction(Register.R4)
		]).Returns;
		Assert.That(result!.Value.Number, Is.EqualTo(1));
	}

	[TestCase("add", 1)]
	[TestCase("subtract", 2)]
	[TestCase("other", 3)]
	public void SelectorIfReturnsCorrectCase(string operation, double expected)
	{
		var instructions = new BytecodeGenerator(GenerateMethodCallFromSource(
			nameof(SelectorIfReturnsCorrectCase),
			$"{nameof(SelectorIfReturnsCorrectCase)}(\"{operation}\").GetResult",
			// @formatter:off
			"has operation Text",
			"GetResult Number",
			"\tif operation is",
			"\t\t\"add\" then 1",
			"\t\t\"subtract\" then 2",
			"\t\telse 3")).Generate();
		// @formatter:on
		var result = vm.Execute(instructions).Returns!;
		Assert.That(result.Value.Number, Is.EqualTo(expected));
	}

	[Test]
	public void RoundTripInvokeWithDoubleNumberArgument()
	{
		var instructions = new BytecodeGenerator(
			GenerateMethodCallFromSource("DoubleCalc", "DoubleCalc(3.14).GetHalf",
				"has number",
				"GetHalf Number",
				"\tnumber / 2")).Generate();
		var result = vm.Execute(instructions).Returns!;
		Assert.That(result.Value.Number, Is.EqualTo(3.14 / 2));
	}

	[Test]
	public void CreateInstanceWithLoggerTraitMember()
	{
		if (type.Package.FindDirectType("TypeWithLogger") == null)
			new Type(type.Package, new TypeLines("TypeWithLogger", "has logger",
				"GetZero Number", "\t0")).ParseMembersAndMethods(new MethodExpressionParser());
		var typeWithLogger = type.Package.FindDirectType("TypeWithLogger")!;
		var fromMethodCall = CreateFromMethodCall(typeWithLogger);
		var instructions = new List<Instruction> { new Invoke(Register.R0, fromMethodCall, new Registry()) };
		var result = vm.Execute(instructions).Memory.Registers[Register.R0];
		Assert.That(result.TryGetValueTypeInstance(), Is.Not.Null);
	}

	[Test]
	public void CreateInstanceWithTextWriterTraitMemberCreatesSystemMemberValue()
	{
		if (type.Package.FindDirectType("TypeWithTextWriter") == null)
			new Type(type.Package, new TypeLines("TypeWithTextWriter", "has writer TextWriter",
				"GetZero Number", "\t0")).ParseMembersAndMethods(new MethodExpressionParser());
		var typeWithTextWriter = type.Package.FindDirectType("TypeWithTextWriter")!;
		var fromMethodCall = CreateFromMethodCall(typeWithTextWriter);
		var instructions = new List<Instruction> { new Invoke(Register.R0, fromMethodCall, new Registry()) };
		var result = vm.Execute(instructions).Memory.Registers[Register.R0];
		var typeInstance = result.TryGetValueTypeInstance();
		Assert.That(typeInstance, Is.Not.Null);
		Assert.That(typeInstance!["writer"].GetType().Name, Is.EqualTo(Type.System));
	}

	[Test]
	public void AddHundredElementsToMutableList()
	{
		var source = new[]
		{
			"has count Number",
			"AddMany Numbers",
			"\tmutable myList = (0)",
			"\tfor count",
			"\t\tmyList = myList + value",
			"\tmyList"
		};
		var instructions = new BytecodeGenerator(GenerateMethodCallFromSource(
			nameof(AddHundredElementsToMutableList),
			$"{nameof(AddHundredElementsToMutableList)}(100).AddMany",
			source)).Generate();
		var startTime = DateTime.UtcNow;
		//TODO: still horrible performance, this needs to be optimized, the VM recreates the mutable list every time, which makes no sense, it just needs to mutate it
		var result = vm.Execute(instructions).Returns!.Value;
		var elapsedMs = (DateTime.UtcNow - startTime).TotalMilliseconds;
		Assert.That(result.List.Items.Count, Is.EqualTo(101));
		Assert.That(elapsedMs, Is.LessThan(800));
	}
}