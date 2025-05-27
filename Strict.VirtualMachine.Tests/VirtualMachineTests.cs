using NUnit.Framework;
using Strict.Language;
using Strict.Language.Expressions;
using Type = Strict.Language.Type;

namespace Strict.VirtualMachine.Tests;

[Ignore("TODO: fix later")]
public class VirtualMachineTests : BaseVirtualMachineTests
{
	[SetUp]
	public void Setup() => vm = new VirtualMachine();

	protected VirtualMachine vm = null!;

	private void CreateSampleEnum() =>
		new Type(type.Package,
			new TypeLines("Days", "has Monday = 1", "has Tuesday = 2", "has Wednesday = 3",
				"has Thursday = 4", "has Friday = 5", "has Saturday = 6")).ParseMembersAndMethods(new MethodExpressionParser());

	[Test]
	public void ReturnEnum()
	{
		CreateSampleEnum();
		var statements = new ByteCodeGenerator(GenerateMethodCallFromSource("WeekDays",
			"WeekDays(5).GetMonday", "has dummy Number", "GetMonday Number",
			"\tconstant monday = Days.Monday", "\tmonday")).Generate();
		var result = vm.Execute(statements).Returns;
		Assert.That(result!.Value, Is.EqualTo(1));
	}

	[Test]
	public void EnumIfConditionComparison()
	{
		CreateSampleEnum();
		var statements = new ByteCodeGenerator(GenerateMethodCallFromSource("WeekDays",
			"WeekDays(5).GetMonday(Days.Monday)", "has dummy Number", "GetMonday(days) Boolean",
			"\tif days is Days.Monday", "\t\treturn true", "\telse", "\t\treturn false")).Generate();
		var result = vm.Execute(statements).Returns;
		Assert.That(result!.Value, Is.EqualTo(true));
	}

	[TestCase(Instruction.Add, 15, 5, 10)]
	[TestCase(Instruction.Subtract, 5, 8, 3)]
	[TestCase(Instruction.Multiply, 4, 2, 2)]
	[TestCase(Instruction.Divide, 3, 7.5, 2.5)]
	[TestCase(Instruction.Modulo, 1, 5, 2)]
	[TestCase(Instruction.Add, "510", "5", 10)]
	[TestCase(Instruction.Add, "510", 5, "10")]
	[TestCase(Instruction.Add, "510", "5", "10")]
	public void Execute(Instruction operation, object expected, params object[] inputs) =>
		Assert.That(vm.Execute(BuildStatements(inputs, operation)).Memory.Registers[Register.R1].Value,
			Is.EqualTo(expected));

	private static Statement[]
		BuildStatements(IReadOnlyList<object> inputs, Instruction operation) =>
	[
		new SetStatement(new Instance(inputs[0] is int
				? NumberType
				: TextType, inputs[0]), Register.R0),
			new SetStatement(new Instance(inputs[1] is int
				? NumberType
				: TextType, inputs[1]), Register.R1),
			new BinaryStatement(operation, Register.R0, Register.R1)
	];

	[Test]
	public void LoadVariable() =>
		Assert.That(
			vm.Execute([
				new LoadConstantStatement(Register.R0, new Instance(NumberType, 5))
			]).Memory.Registers[Register.R0].Value, Is.EqualTo(5));

	[Test]
	public void SetAndAdd() =>
		Assert.That(
			vm.Execute([
				new LoadConstantStatement(Register.R0, new Instance(NumberType, 10)),
				new LoadConstantStatement(Register.R1, new Instance(NumberType, 5)),
				new BinaryStatement(Instruction.Add, Register.R0, Register.R1, Register.R2)
			]).Memory.Registers[Register.R2].Value, Is.EqualTo(15));

	[Test]
	public void AddFiveTimes() =>
		Assert.That(vm.Execute([
			new SetStatement(new Instance(NumberType, 5), Register.R0),
			new SetStatement(new Instance(NumberType, 1), Register.R1),
			new SetStatement(new Instance(NumberType, 0), Register.R2),
			new BinaryStatement(Instruction.Add, Register.R0, Register.R2, Register.R2),
			new BinaryStatement(Instruction.Subtract, Register.R0, Register.R1, Register.R0),
			new JumpIfNotZeroStatement(-3, Register.R0)
		]).Memory.Registers[Register.R2].Value, Is.EqualTo(0 + 5 + 4 + 3 + 2 + 1));

	[TestCase("ArithmeticFunction(10, 5).Calculate(\"add\")", 15)]
	[TestCase("ArithmeticFunction(10, 5).Calculate(\"subtract\")", 5)]
	[TestCase("ArithmeticFunction(10, 5).Calculate(\"multiply\")", 50)]
	[TestCase("ArithmeticFunction(10, 5).Calculate(\"divide\")", 2)]
	public void RunArithmeticFunctionExample(string methodCall, int expectedResult)
	{
		var statements = new ByteCodeGenerator(GenerateMethodCallFromSource("ArithmeticFunction",
			methodCall, ArithmeticFunctionExample)).Generate();
		Assert.That(vm.Execute(statements).Returns?.Value, Is.EqualTo(expectedResult));
	}

	[Test]
	public void AccessListByIndex()
	{
		var statements = new ByteCodeGenerator(GenerateMethodCallFromSource("AccessList",
			"AccessList(1, 2, 3, 4, 5).Get(2)", "has numbers", "Get(index Number) Number",
			"\tconstant element = numbers(index)", "\telement")).Generate();
		Assert.That(vm.Execute(statements).Returns?.Value, Is.EqualTo(3));
	}

	[Test]
	public void AccessListByIndexNonNumberType()
	{
		var statements = new ByteCodeGenerator(GenerateMethodCallFromSource("AccessList",
			"AccessList(\"1\", \"2\", \"3\", \"4\", \"5\").Get(2)", "has texts", "Get(index Number) Text",
			"\tconstant element = texts(index)", "\telement")).Generate();
		Assert.That(vm.Execute(statements).Returns?.Value, Is.EqualTo("3"));
	}

	[Test]
	public void ReduceButGrowLoopExample() =>
		Assert.That(
			vm.Execute([
				new StoreVariableStatement(new Instance(NumberType, 10), "number"),
				new StoreVariableStatement(new Instance(NumberType, 1), "result"),
				new StoreVariableStatement(new Instance(NumberType, 2), "multiplier"),
				new LoadVariableStatement(Register.R0, "number"),
				new LoopBeginStatement(Register.R0), new LoadVariableStatement(Register.R2, "result"),
				new LoadVariableStatement(Register.R3, "multiplier"),
				new BinaryStatement(Instruction.Multiply, Register.R2, Register.R3, Register.R4),
				new StoreFromRegisterStatement(Register.R4, "result"),
				new IterationEndStatement(5),
				new LoadVariableStatement(Register.R5, "result"), new ReturnStatement(Register.R5)
			]).Returns?.Value, Is.EqualTo(1024));

	[TestCase("NumberConvertor", "NumberConvertor(5).ConvertToText", "5", "has number", "ConvertToText Text",
		"\tconstant result = 5 to Text", "\tresult")]
	[TestCase("TextConvertor", "TextConvertor(\"5\").ConvertToNumber", 5, "has text", "ConvertToNumber Number",
		"\tconstant result = text to Number", "\tresult")]
	public void ExecuteToOperator(string programName, string methodCall, object expected, params string[] code)
	{
		var statements = new ByteCodeGenerator(GenerateMethodCallFromSource(programName,
			methodCall, code)).Generate();
		Assert.That(vm.Execute(statements).Returns?.Value, Is.EqualTo(expected));
	}

	//ncrunch: no coverage start
	private static IEnumerable<TestCaseData> MethodCallTests
	{
		get
		{
			yield return new TestCaseData("AddNumbers", "AddNumbers(2, 5).GetSum", SimpleMethodCallCode, 7);
			yield return new TestCaseData("CallWithConstants", "CallWithConstants(2, 5).GetSum", MethodCallWithConstantValues, 6);
			yield return new TestCaseData("CallWithoutArguments", "CallWithoutArguments(2, 5).GetSum", MethodCallWithLocalWithNoArguments, 542);
			yield return new TestCaseData("CurrentlyFailing", "CurrentlyFailing(10).SumEvenNumbers", CurrentlyFailingTest, 20);
		}
	} //ncrunch: no coverage end

	[TestCaseSource(nameof(MethodCallTests))]
	// ReSharper disable TooManyArguments, makes below tests easier
	public void MethodCall(string programName, string methodCall, string[] source, object expected)
	{
		var statements =
			new ByteCodeGenerator(GenerateMethodCallFromSource(programName, methodCall,
				source)).Generate();
		Assert.That(vm.Execute(statements).Returns?.Value, Is.EqualTo(expected));
	}

	[Test]
	public void IfAndElseTest()
	{
		var statements = new ByteCodeGenerator(GenerateMethodCallFromSource("IfAndElseTest",
			"IfAndElseTest(3).IsEven", IfAndElseTestCode)).Generate();
		Assert.That(vm.Execute(statements).Returns?.Value,
			Is.EqualTo("Number is less or equal than 10"));
	}

	[TestCase("EvenSumCalculator(100).IsEven", 2450, "EvenSumCalculator",
		new[]
		{
			"has number", "IsEven Number", "\tmutable sum = 0", "\tfor number",
			"\t\tif (index % 2) is 0", "\t\t\tsum = sum + index", "\tsum"
		})]
	[TestCase("EvenSumCalculatorForList(100, 200, 300).IsEvenList", 2, "EvenSumCalculatorForList",
		new[]
		{
			"has numbers", "IsEvenList Number",
			"\tmutable sum = 0",
			"\tfor numbers",
			"\t\tif (index % 2) is 0",
			"\t\t\tsum = sum + index",
			"\tsum"
		})]
	public void CompileCompositeBinariesInIfCorrectlyWithModulo(string methodCall,
		object expectedResult, string methodName, params string[] code)
	{
		var statements = new ByteCodeGenerator(GenerateMethodCallFromSource(methodName,
			methodCall, code)).Generate();
		Assert.That(vm.Execute(statements).Returns?.Value, Is.EqualTo(expectedResult));
	}

	[TestCase("AddToTheList(5).Add", "100 200 300 400 0 1 2 3", "AddToTheList",
		new[]
		{
			"has number",
			"Add Numbers",
			"\tmutable myList = (100, 200, 300, 400)",
			"\tfor myList",
			"\t\tif (value % 2) is 0",
			"\t\t\tmyList = myList + index",
			"\tmyList"
		})]
	[TestCase("RemoveFromTheList(5).Remove", "100 200 300", "RemoveFromTheList",
		new[]
		{
			"has number",
			"Remove Numbers",
			"\tmutable myList = (100, 200, 300, 400)",
			"\tfor myList",
			"\t\tif value is 400",
			"\t\t\tmyList = myList - 400",
			"\tmyList"
		})]
	[TestCase("RemoveB(\"s\", \"b\", \"s\").Remove", "s s", "RemoveB",
		new[]
		{
			"has texts",
			"Remove Texts",
			"\tmutable textList = texts",
			"\tfor texts",
			"\t\tif value is \"b\"",
			"\t\t\ttextList = textList - value",
			"\ttextList"
		})]
	[TestCase("ListRemove(\"s\", \"b\", \"s\").Remove", "s s", "ListRemove",
		new[]
		{
			"has texts",
			"Remove Texts",
			"\tmutable textList = texts",
			"\ttextList.Remove(\"b\")",
			"\ttextList"
		})]
	[TestCase("ListRemoveMultiple(\"s\", \"b\", \"s\").Remove", "b", "ListRemoveMultiple",
		new[]
		{
			"has texts",
			"Remove Texts",
			"\tmutable textList = texts",
			"\ttextList.Remove(\"s\")",
			"\ttextList"
		})]
	[TestCase("RemoveDuplicates(\"s\", \"b\", \"s\").Remove", "s b", "RemoveDuplicates",
		new[]
		{
			"has texts",
			"Remove Texts",
			"\tmutable textList = (\"\")",
			"\tfor texts",
			"\t\tif textList.Contains(value) is false",
			"\t\t\ttextList = textList + value",
			"\ttextList"
		})]
	public void ExecuteListBinaryOperations(string methodCall,
		object expectedResult, string programName, params string[] code)
	{
		var statements = new ByteCodeGenerator(GenerateMethodCallFromSource(programName,
			methodCall, code)).Generate();
		var values = (List<Expression>)vm.Execute(statements).Returns?.Value!;
		var elements = values.Aggregate("", (current, value) => current + ((Value)value).Data + " ");
		Assert.That(elements.Trim(), Is.EqualTo(expectedResult));
	} //ncrunch: no coverage end

	[TestCase("TestContains(\"s\", \"b\", \"s\").Contains(\"b\")", "True", "TestContains",
		new[]
		{
			"has elements Texts",
			"Contains(other Text) Boolean",
			"\tfor elements",
			"\t\tif value is other",
			"\t\t\treturn true",
			"\tfalse"
		})]
	public void CallCommonMethodCalls(string methodCall, object expectedResult,
		string programName, params string[] code)
	{
		var statements = new ByteCodeGenerator(GenerateMethodCallFromSource(programName,
			methodCall, code)).Generate();
		var result = vm.Execute(statements).Returns?.Value!;
		Assert.That(result.ToString(), Is.EqualTo(expectedResult));
	}

	[TestCase("CollectionAdd(5).AddNumberToList",
		"1 2 3 5",
		"has number",
		"AddNumberToList Numbers",
		"\tmutable numbers = (1, 2, 3)",
		"\tnumbers.Add(number)",
		"\tnumbers")]
	public void CollectionAdd(string methodCall, string expected, params string[] code)
	{
		var statements =
			new ByteCodeGenerator(
				GenerateMethodCallFromSource(nameof(CollectionAdd), methodCall, code)).Generate();
		var result = ExpressionListToSpaceSeparatedString(statements);
		Assert.That(result.TrimEnd(), Is.EqualTo(expected));
	}

	private string ExpressionListToSpaceSeparatedString(IList<Statement> statements) =>
		((IEnumerable<Expression>)vm.Execute(statements).Returns?.Value!).Aggregate("",
			(current, value) => current + ((Value)value).Data + " ");

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
			((Dictionary<Value, Value>)vm.
				Execute(new ByteCodeGenerator(GenerateMethodCallFromSource(nameof(DictionaryAdd),
					"DictionaryAdd(5).RemoveFromDictionary", code)).Generate()).Memory.Variables["values"].
				Value).Count, Is.EqualTo(1));
	}

	[TestCase("CollectionAdd(5).AddToDictionary",
		"5",
		"has number",
		"AddToDictionary Number",
		"\tmutable values = Dictionary(Number, Number)",
		"\tvalues.Add(1, number)",
		"\tvalues.Get(1)")]
	public void DictionaryGet(string methodCall, string expected, params string[] code)
	{
		var statements =
			new ByteCodeGenerator(
				GenerateMethodCallFromSource(nameof(CollectionAdd), methodCall, code)).Generate();
		var result = vm.Execute(statements).Returns?.Value!;
		Assert.That(result.ToString(), Is.EqualTo(expected));
	}

	[TestCase("CollectionAdd(5).AddToDictionary",
		"5",
		"has number",
		"AddToDictionary Number",
		"\tmutable values = Dictionary(Number, Number)",
		"\tvalues.Add(1, number)",
		"\tvalues.Add(2, number + 10)",
		"\tvalues.Remove(1)",
		"\tvalues.Get(1)")]
	public void DictionaryRemove(string methodCall, string expected, params string[] code)
	{
		var statements =
			new ByteCodeGenerator(
				GenerateMethodCallFromSource(nameof(CollectionAdd), methodCall, code)).Generate();
		var result = vm.Execute(statements).Returns?.Value!;
		Assert.That(result.ToString(), Is.EqualTo(expected));
	}

	[Test]
	public void ReturnWithinALoop()
	{
		var source = new[]
		{
			"has number", "GetAll Number", "\tfor number", "\t\tvalue"
		};
		var statements = new ByteCodeGenerator(GenerateMethodCallFromSource(nameof(ReturnWithinALoop),
			"ReturnWithinALoop(5).GetAll", source)).Generate();
		Assert.That(() => vm.Execute(statements).Returns?.Value, Is.EqualTo(5));
	}

	[Test]
	public void ReverseWithRange()
	{
		var source = new[]
		{
			"has numbers", "Reverse Numbers", "\tmutable result = Numbers", "\tconstant len = numbers.Length - 1", "\tfor Range(len, 0)",
			"\t\tresult.Add(numbers(index))", "\tresult"
		};
		var statements = new ByteCodeGenerator(GenerateMethodCallFromSource(nameof(ReverseWithRange),
			"ReverseWithRange(1, 2, 3).Reverse", source)).Generate();
		Assert.That(() => ExpressionListToSpaceSeparatedString(statements), Is.EqualTo("3 2 1 "));
	}

	[Test]
	public void ConditionalJump() =>
		Assert.That(
			vm.Execute([
				new SetStatement(new Instance(NumberType, 5), Register.R0),
				new SetStatement(new Instance(NumberType, 1), Register.R1),
				new SetStatement(new Instance(NumberType, 10), Register.R2),
				new BinaryStatement(Instruction.LessThan, Register.R2, Register.R0),
				new JumpIfStatement(Instruction.JumpIfTrue, 2),
				new BinaryStatement(Instruction.Add, Register.R2, Register.R0, Register.R0)
			]).Memory.Registers[Register.R0].Value, Is.EqualTo(15));

	[TestCase(Instruction.GreaterThan, new[] { 1, 2 }, 2 - 1)]
	[TestCase(Instruction.LessThan, new[] { 1, 2 }, 1 + 2)]
	[TestCase(Instruction.Equal, new[] { 5, 5 }, 5 + 5)]
	[TestCase(Instruction.NotEqual, new[] { 5, 5 }, 5 - 5)]
	public void ConditionalJumpIfAndElse(Instruction conditional, int[] registers, int expected) =>
		Assert.That(
			vm.Execute([
				new SetStatement(new Instance(NumberType, registers[0]), Register.R0),
				new SetStatement(new Instance(NumberType, registers[1]), Register.R1),
				new BinaryStatement(conditional, Register.R0, Register.R1),
				new JumpIfStatement(Instruction.JumpIfTrue, 2),
				new BinaryStatement(Instruction.Subtract, Register.R1, Register.R0, Register.R0),
				new JumpIfStatement(Instruction.JumpIfFalse, 2),
				new BinaryStatement(Instruction.Add, Register.R0, Register.R1, Register.R0)
			]).Memory.Registers[Register.R0].Value, Is.EqualTo(expected));

	[TestCase(Instruction.Add)]
	[TestCase(Instruction.GreaterThan)]
	public void OperandsRequired(Instruction instruction) =>
		Assert.That(
			() => vm.Execute([new BinaryStatement(instruction, Register.R0)]),
			Throws.InstanceOf<VirtualMachine.OperandsRequired>());
}