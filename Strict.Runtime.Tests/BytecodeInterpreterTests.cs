using System.Globalization;

namespace Strict.Runtime.Tests;

public class BytecodeInterpreterTests : BaseVirtualMachineTests
{
	[SetUp]
	public void Setup() => vm = new BytecodeInterpreter();

	protected BytecodeInterpreter vm = null!;

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
		var statements = new ByteCodeGenerator(GenerateMethodCallFromSource(nameof(ReturnEnum),
			nameof(ReturnEnum) + "(5).GetMonday", "has dummy Number", "GetMonday Number",
			"\tDays.Monday")).Generate();
		var result = vm.Execute(statements).Returns;
		Assert.That(result!.Value.Number, Is.EqualTo(1));
	}

	[Test]
	public void EnumIfConditionComparison()
	{
		CreateSampleEnum();
		var statements = new ByteCodeGenerator(GenerateMethodCallFromSource(
			nameof(EnumIfConditionComparison),
			nameof(EnumIfConditionComparison) + "(5).GetMonday(Days.Monday)", "has dummy Number",
			"GetMonday(days) Boolean", "\tif days is Days.Monday", "\t\treturn true", "\telse",
			"\t\treturn false")).Generate();
		var result = vm.Execute(statements).Returns!;
		Assert.That(result.Value.Number, Is.EqualTo(1));
	}

	[TestCase(Instruction.Add, 15, 5, 10)]
	[TestCase(Instruction.Subtract, 5, 8, 3)]
	[TestCase(Instruction.Multiply, 4, 2, 2)]
	[TestCase(Instruction.Divide, 3, 7.5, 2.5)]
	[TestCase(Instruction.Modulo, 1, 5, 2)]
	[TestCase(Instruction.Add, "510", "5", 10)]
	[TestCase(Instruction.Add, "510", 5, "10")]
	[TestCase(Instruction.Add, "510", "5", "10")]
	public void Execute(Instruction operation, object expected, params object[] inputs)
	{
		var result = vm.Execute(BuildStatements(inputs, operation)).Memory.Registers[Register.R1];
		var actual = expected is string
			? (object)result.Text
			: result.Number;
		Assert.That(actual, Is.EqualTo(expected));
	}

	private static Statement[]
		BuildStatements(IReadOnlyList<object> inputs, Instruction operation) =>
	[
		new SetStatement(inputs[0] is string s0
			? Text(s0)
			: Number(Convert.ToDouble(inputs[0])), Register.R0),
		new SetStatement(inputs[1] is string s1
			? Text(s1)
			: inputs[1] is double d
				? Number(d)
				: Number(Convert.ToDouble(inputs[1])), Register.R1),
		new Binary(operation, Register.R0, Register.R1)
	];

	[Test]
	public void LoadVariable() =>
		Assert.That(vm.Execute([
			new LoadConstantStatement(Register.R0, Number(5))
		]).Memory.Registers[Register.R0].Number, Is.EqualTo(5));

	[Test]
	public void SetAndAdd() =>
		Assert.That(vm.Execute([
			new LoadConstantStatement(Register.R0, Number(10)),
			new LoadConstantStatement(Register.R1, Number(5)),
			new Binary(Instruction.Add, Register.R0, Register.R1, Register.R2)
		]).Memory.Registers[Register.R2].Number, Is.EqualTo(15));

	[Test]
	public void AddFiveTimes() =>
		Assert.That(vm.Execute([
			new SetStatement(Number(5), Register.R0),
			new SetStatement(Number(1), Register.R1),
			new SetStatement(Number(0), Register.R2),
			new Binary(Instruction.Add, Register.R0, Register.R2, Register.R2),
			new Binary(Instruction.Subtract, Register.R0, Register.R1, Register.R0),
			new JumpIfNotZero(-3, Register.R0)
		]).Memory.Registers[Register.R2].Number, Is.EqualTo(0 + 5 + 4 + 3 + 2 + 1));

	[TestCase("ArithmeticFunction(10, 5).Calculate(\"add\")", 15)]
	[TestCase("ArithmeticFunction(10, 5).Calculate(\"subtract\")", 5)]
	[TestCase("ArithmeticFunction(10, 5).Calculate(\"multiply\")", 50)]
	[TestCase("ArithmeticFunction(10, 5).Calculate(\"divide\")", 2)]
	public void RunArithmeticFunctionExample(string methodCall, int expectedResult)
	{
		var statements = new ByteCodeGenerator(GenerateMethodCallFromSource("ArithmeticFunction",
			methodCall, ArithmeticFunctionExample)).Generate();
		Assert.That(vm.Execute(statements).Returns!.Value.Number, Is.EqualTo(expectedResult));
	}

	[Test]
	public void AccessListByIndex()
	{
		var statements = new ByteCodeGenerator(GenerateMethodCallFromSource(nameof(AccessListByIndex),
			nameof(AccessListByIndex) + "(1, 2, 3, 4, 5).Get(2)", "has numbers",
			"Get(index Number) Number", "\tnumbers(index)")).Generate();
		Assert.That(vm.Execute(statements).Returns!.Value.Number, Is.EqualTo(3));
	}

	[Test]
	public void AccessListByIndexNonNumberType()
	{
		var statements = new ByteCodeGenerator(GenerateMethodCallFromSource(
			nameof(AccessListByIndexNonNumberType),
			nameof(AccessListByIndexNonNumberType) + "(\"1\", \"2\", \"3\", \"4\", \"5\").Get(2)",
			"has texts", "Get(index Number) Text", "\ttexts(index)")).Generate();
		Assert.That(vm.Execute(statements).Returns!.Value.Text, Is.EqualTo("3"));
	}

	[Test]
	public void ReduceButGrowLoopExample() =>
		Assert.That(vm.Execute([
			new StoreVariableStatement(Number(10), "number"),
			new StoreVariableStatement(Number(1), "result"),
			new StoreVariableStatement(Number(2), "multiplier"),
			new LoadVariableToRegister(Register.R0, "number"),
			new LoopBeginStatement(Register.R0), new LoadVariableToRegister(Register.R2, "result"),
			new LoadVariableToRegister(Register.R3, "multiplier"),
			new Binary(Instruction.Multiply, Register.R2, Register.R3, Register.R4),
			new StoreFromRegisterStatement(Register.R4, "result"),
			new LoopEndStatement(5),
			new LoadVariableToRegister(Register.R5, "result"), new Return(Register.R5)
		]).Returns!.Value.Number, Is.EqualTo(1024));

	[TestCase("NumberConvertor", "NumberConvertor(5).ConvertToText", "5", "has number",
		"ConvertToText Text", "\t5 to Text")]
	[TestCase("TextConvertor", "TextConvertor(\"5\").ConvertToNumber", 5, "has text",
		"ConvertToNumber Number", "\ttext to Number")]
	public void ExecuteToOperator(string programName, string methodCall, object expected,
		params string[] code)
	{
		var statements = new ByteCodeGenerator(GenerateMethodCallFromSource(programName,
			methodCall, code)).Generate();
		var result = vm.Execute(statements).Returns!.Value;
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
			yield return new TestCaseData("AddNumbers", "AddNumbers(2, 5).GetSum", SimpleMethodCallCode,
				7);
			yield return new TestCaseData("CallWithConstants", "CallWithConstants(2, 5).GetSum",
				MethodCallWithConstantValues, 6);
			yield return new TestCaseData("CallWithoutArguments", "CallWithoutArguments(2, 5).GetSum",
				MethodCallWithLocalWithNoArguments, 542);
			yield return new TestCaseData("CurrentlyFailing", "CurrentlyFailing(10).SumEvenNumbers",
				CurrentlyFailingTest, 20);
		}
	} //ncrunch: no coverage end

	[TestCaseSource(nameof(MethodCallTests))]
	// ReSharper disable TooManyArguments, makes below tests easier
	public void MethodCall(string programName, string methodCall, string[] source, object expected)
	{
		var statements =
			new ByteCodeGenerator(GenerateMethodCallFromSource(programName, methodCall, source)).
				Generate();
		Assert.That(vm.Execute(statements).Returns!.Value.Number, Is.EqualTo(expected));
	}

	[Test]
	public void IfAndElseTest()
	{
		var statements = new ByteCodeGenerator(GenerateMethodCallFromSource("IfAndElseTest",
			"IfAndElseTest(3).IsEven", IfAndElseTestCode)).Generate();
		Assert.That(vm.Execute(statements).Returns!.Value.Text,
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
		var statements =
			new ByteCodeGenerator(GenerateMethodCallFromSource(methodName, methodCall, code)).
				Generate();
		Assert.That(vm.Execute(statements).Returns!.Value.Number, Is.EqualTo(expectedResult));
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
		var statements = new ByteCodeGenerator(GenerateMethodCallFromSource(programName,
			methodCall, code)).Generate();
		var result = vm.Execute(statements).Returns!.Value;
		var elements = result.List.Items.Aggregate("", (current, item) => current + (item.IsText
			? item.Text
			: item.Number) + " ");
		Assert.That(elements.Trim(), Is.EqualTo(expectedResult));
	} //ncrunch: no coverage end

	[TestCase("TestContains(\"s\", \"b\", \"s\").Contains(\"b\")", "true", "TestContains",
		new[]
		{
			"has elements Texts", "Contains(other Text) Boolean", "\tfor elements",
			"\t\tif value is other", "\t\t\treturn true", "\tfalse"
		})]
	public void CallCommonMethodCalls(string methodCall, object expectedResult, string programName,
		params string[] code)
	{
		var statements = new ByteCodeGenerator(GenerateMethodCallFromSource(programName,
			methodCall, code)).Generate();
		var result = vm.Execute(statements).Returns!.Value;
		Assert.That(result.ToExpressionCodeString(), Is.EqualTo(expectedResult));
	}

	[TestCase("NumbersAdder(5).AddNumberToList", "1 2 3 5", "has number", "AddNumberToList Numbers",
		"\tmutable numbers = (1, 2, 3)", "\tnumbers.Add(number)", "\tnumbers")]
	public void CollectionAdd(string methodCall, string expected, params string[] code)
	{
		var statements =
			new ByteCodeGenerator(GenerateMethodCallFromSource("NumbersAdder", methodCall, code)).
				Generate();
		var result = ExpressionListToSpaceSeparatedString(statements);
		Assert.That(result.TrimEnd(), Is.EqualTo(expected));
	}

	private string ExpressionListToSpaceSeparatedString(IList<Statement> statements)
	{
		var result = vm.Execute(statements).Returns!.Value;
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
			vm.Execute(new ByteCodeGenerator(GenerateMethodCallFromSource(nameof(DictionaryAdd),
					"DictionaryAdd(5).RemoveFromDictionary", code)).Generate()).Memory.Variables["values"].
				GetDictionaryItems().Count, Is.EqualTo(1));
	}

	[Test]
	public void CreateEmptyDictionaryFromConstructor()
	{
		var dictionaryType = TestPackage.Instance.GetType(Type.Dictionary).
			GetGenericImplementation(NumberType, NumberType);
		var methodCall = CreateFromMethodCall(dictionaryType);
		var statements = new List<Statement> { new Invoke(Register.R0, methodCall, new Registry()) };
		var result = vm.Execute(statements).Memory.Registers[Register.R0];
		Assert.That(result.IsDictionary, Is.True);
		Assert.That(result.GetDictionaryItems().Count, Is.EqualTo(0));
	}

	[TestCase("DictionaryGet(5).AddToDictionary", "5", "has number", "AddToDictionary Number",
		"\tmutable values = Dictionary(Number, Number)", "\tvalues.Add(1, number)",
		"\tvalues.Get(1)")]
	public void DictionaryGet(string methodCall, string expected, params string[] code)
	{
		var statements =
			new ByteCodeGenerator(
				GenerateMethodCallFromSource(nameof(DictionaryGet), methodCall, code)).Generate();
		var result = vm.Execute(statements).Returns!.Value;
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
		var statements =
			new ByteCodeGenerator(
				GenerateMethodCallFromSource(nameof(DictionaryRemove), methodCall, code)).Generate();
		var result = vm.Execute(statements).Returns!.Value;
		var actual = result.IsText
			? result.Text
			: result.Number.ToString(CultureInfo.InvariantCulture);
		Assert.That(actual, Is.EqualTo("15"));
	}

	[Test]
	public void ReturnWithinALoop()
	{
		var source = new[] { "has number", "GetAll Number", "\tfor number", "\t\tvalue" };
		var statements = new ByteCodeGenerator(GenerateMethodCallFromSource(nameof(ReturnWithinALoop),
			"ReturnWithinALoop(5).GetAll", source)).Generate();
		Assert.That(() => vm.Execute(statements).Returns!.Value.Number,
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
		var statements = new ByteCodeGenerator(GenerateMethodCallFromSource(nameof(ReverseWithRange),
			"ReverseWithRange(1, 2, 3).Reverse", source)).Generate();
		Assert.That(() => ExpressionListToSpaceSeparatedString(statements), Is.EqualTo("3 2 1 "));
	}

	[Test]
	public void ConditionalJump() =>
		Assert.That(vm.Execute([
			new SetStatement(Number(5), Register.R0),
			new SetStatement(Number(1), Register.R1),
			new SetStatement(Number(10), Register.R2),
			new Binary(Instruction.LessThan, Register.R2, Register.R0),
			new JumpIf(Instruction.JumpIfTrue, 2),
			new Binary(Instruction.Add, Register.R2, Register.R0, Register.R0)
		]).Memory.Registers[Register.R0].Number, Is.EqualTo(15));

	[Test]
	public void JumpIfTrueSkipsNextInstruction() =>
		Assert.That(vm.Execute([
			new SetStatement(Number(1), Register.R0),
			new SetStatement(Number(1), Register.R1),
			new SetStatement(Number(0), Register.R2),
			new Binary(Instruction.Equal, Register.R0, Register.R1),
			new JumpIfTrue(1, Register.R0),
			new Binary(Instruction.Add, Register.R0, Register.R1, Register.R2)
		]).Memory.Registers[Register.R2].Number, Is.EqualTo(0));

	[Test]
	public void JumpIfFalseSkipsNextInstruction() =>
		Assert.That(vm.Execute([
			new SetStatement(Number(1), Register.R0),
			new SetStatement(Number(2), Register.R1),
			new SetStatement(Number(0), Register.R2),
			new Binary(Instruction.Equal, Register.R0, Register.R1),
			new JumpIfFalse(1, Register.R0),
			new Binary(Instruction.Add, Register.R0, Register.R1, Register.R2)
		]).Memory.Registers[Register.R2].Number, Is.EqualTo(0));

	[TestCase(Instruction.GreaterThan, new[] { 1, 2 }, 2 - 1)]
	[TestCase(Instruction.LessThan, new[] { 1, 2 }, 1 + 2)]
	[TestCase(Instruction.Equal, new[] { 5, 5 }, 5 + 5)]
	[TestCase(Instruction.NotEqual, new[] { 5, 5 }, 5 - 5)]
	public void ConditionalJumpIfAndElse(Instruction conditional, int[] registers, int expected) =>
		Assert.That(vm.Execute([
			new SetStatement(Number(registers[0]), Register.R0),
			new SetStatement(Number(registers[1]), Register.R1),
			new Binary(conditional, Register.R0, Register.R1),
			new JumpIf(Instruction.JumpIfTrue, 2),
			new Binary(Instruction.Subtract, Register.R1, Register.R0, Register.R0),
			new JumpIf(Instruction.JumpIfFalse, 2),
			new Binary(Instruction.Add, Register.R0, Register.R1, Register.R0)
		]).Memory.Registers[Register.R0].Number, Is.EqualTo(expected));

	[TestCase(Instruction.Add)]
	[TestCase(Instruction.GreaterThan)]
	public void OperandsRequired(Instruction instruction) =>
		Assert.That(() => vm.Execute([new Binary(instruction, Register.R0)]),
			Throws.InstanceOf<BytecodeInterpreter.OperandsRequired>());
}