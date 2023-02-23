using NUnit.Framework;
using Strict.Language;
using Strict.Language.Expressions;

namespace Strict.VirtualMachine.Tests;

public sealed class VirtualMachineTests : BaseVirtualMachineTests
{
	[SetUp]
	public void Setup() => vm = new VirtualMachine();

	private VirtualMachine vm = null!;

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
		new Statement[]
		{
			new SetStatement(new Instance(inputs[0] is int
				? NumberType
				: TextType, inputs[0]), Register.R0),
			new SetStatement(new Instance(inputs[1] is int
				? NumberType
				: TextType, inputs[1]), Register.R1),
			new BinaryStatement(operation, Register.R0, Register.R1)
		};

	[Test]
	public void LoadVariable() =>
		Assert.That(
			vm.Execute(new Statement[]
			{
				new LoadConstantStatement(Register.R0, new Instance(NumberType, 5))
			}).Memory.Registers[Register.R0].Value, Is.EqualTo(5));

	[Test]
	public void SetAndAdd() =>
		Assert.That(
			vm.Execute(new Statement[]
			{
				new LoadConstantStatement(Register.R0, new Instance(NumberType, 10)),
				new LoadConstantStatement(Register.R1, new Instance(NumberType, 5)),
				new BinaryStatement(Instruction.Add, Register.R0, Register.R1, Register.R2)
			}).Memory.Registers[Register.R2].Value, Is.EqualTo(15));

	[Test]
	public void AddFiveTimes() =>
		Assert.That(vm.Execute(new Statement[]
		{
			new SetStatement(new Instance(NumberType, 5), Register.R0),
			new SetStatement(new Instance(NumberType, 1), Register.R1),
			new SetStatement(new Instance(NumberType, 0), Register.R2),
			new BinaryStatement(Instruction.Add, Register.R0, Register.R2, Register.R2), // R2 = R0 + R2
			new BinaryStatement(Instruction.Subtract, Register.R0, Register.R1, Register.R0),
			new JumpIfNotZeroStatement(-3, Register.R0)
		}).Memory.Registers[Register.R2].Value, Is.EqualTo(0 + 5 + 4 + 3 + 2 + 1));

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
	public void ReduceButGrowLoopExample() =>
		Assert.That(
			vm.Execute(new Statement[]
			{
				new StoreVariableStatement(new Instance(NumberType, 10), "number"),
				new StoreVariableStatement(new Instance(NumberType, 1), "result"),
				new StoreVariableStatement(new Instance(NumberType, 2), "multiplier"),
				new LoopBeginStatement("number", Register.R0), new LoadVariableStatement(Register.R2, "result"),
				new LoadVariableStatement(Register.R3, "multiplier"),
				new BinaryStatement(Instruction.Multiply, Register.R2, Register.R3, Register.R4),
				new StoreFromRegisterStatement(Register.R4, "result"),
				new IterationEndStatement(Register.R0),
				new JumpIfNotZeroStatement(-6, Register.R0),
				new LoadVariableStatement(Register.R5, "result"), new ReturnStatement(Register.R5)
			}).Returns?.Value, Is.EqualTo(1024));

	[TestCase("RemoveParentheses(\"some(thing)\").Remove", "some")]
	[TestCase("RemoveParentheses(\"(some)thing\").Remove", "thing")]
	public void RemoveParentheses(string methodCall, string expectedResult)
	{
		var statements = new ByteCodeGenerator(GenerateMethodCallFromSource("RemoveParentheses",
			methodCall, RemoveParenthesesKata)).Generate();
		Assert.That(vm.Execute(statements).Returns?.Value, Is.EqualTo(expectedResult));
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
	}
	//ncrunch: no coverage end

	[TestCaseSource(nameof(MethodCallTests))]
	// ReSharper disable once TooManyArguments
	public void MethodCall(string programName, string methodCall, string[] source, object expected)
	{
		var statements =
			new ByteCodeGenerator(GenerateMethodCallFromSource(programName, methodCall,
				source)).Generate();
		Assert.That(vm.Execute(statements).Returns?.Value, Is.EqualTo(expected));
	}

	[TestCase("Invertor((1, 2, 3, 4, 5)).Invert", "-1-2-3-4-5")]
	public void InvertValues(string methodCall, string expectedResult)
	{
		var statements = new ByteCodeGenerator(GenerateMethodCallFromSource("Invertor",
			methodCall, InvertValueKata)).Generate();
		Assert.That(vm.Execute(statements).Returns?.Value, Is.EqualTo(expectedResult));
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
	[TestCase("EvenSumCalculatorForList((100, 200, 300)).IsEvenList", 2, "EvenSumCalculatorForList",
		new[]
		{
			"has numbers", "IsEvenList Number", "\tmutable sum = 0", "\tfor numbers",
			"\t\tif (index % 2) is 0", "\t\t\tsum = sum + index", "\tsum"
		})]
	// ReSharper disable once TooManyArguments
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
			"Add Number",
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
			"Remove Number",
			"\tmutable myList = (100, 200, 300, 400)",
			"\tfor myList",
			"\t\tif value is 400",
			"\t\t\tmyList = myList - 400",
			"\tmyList"
		})]
	// ReSharper disable once TooManyArguments
	public void CompileListBinaryOperations(string methodCall,
		object expectedResult, string methodName, params string[] code)
	{
		var statements = new ByteCodeGenerator(GenerateMethodCallFromSource(methodName,
			methodCall, code)).Generate();
		var values = (List<Expression>)vm.Execute(statements).Returns?.Value!;
		var elements = values.Aggregate("", (current, value) => current + ((Value)value).Data + " ");
		Assert.That(elements.Trim(), Is.EqualTo(expectedResult));
	}

	[Test]
	public void ConditionalJump() =>
		Assert.That(
			vm.Execute(new Statement[]
			{
				new SetStatement(new Instance(NumberType, 5), Register.R0),
				new SetStatement(new Instance(NumberType, 1), Register.R1),
				new SetStatement(new Instance(NumberType, 10), Register.R2),
				new BinaryStatement(Instruction.LessThan, Register.R2, Register.R0),
				new JumpIfStatement(Instruction.JumpIfTrue, 2),
				new BinaryStatement(Instruction.Add, Register.R2, Register.R0, Register.R0)
			}).Memory.Registers[Register.R0].Value, Is.EqualTo(15));

	[TestCase(Instruction.GreaterThan, new[] { 1, 2 }, 2 - 1)]
	[TestCase(Instruction.LessThan, new[] { 1, 2 }, 1 + 2)]
	[TestCase(Instruction.Equal, new[] { 5, 5 }, 5 + 5)]
	[TestCase(Instruction.NotEqual, new[] { 5, 5 }, 5 - 5)]
	public void ConditionalJumpIfAndElse(Instruction conditional, int[] registers, int expected) =>
		Assert.That(
			vm.Execute(new Statement[]
			{
				new SetStatement(new Instance(NumberType, registers[0]), Register.R0),
				new SetStatement(new Instance(NumberType, registers[1]), Register.R1),
				new BinaryStatement(conditional, Register.R0, Register.R1),
				new JumpIfStatement(Instruction.JumpIfTrue, 2),
				new BinaryStatement(Instruction.Subtract, Register.R1, Register.R0, Register.R0),
				new JumpIfStatement(Instruction.JumpIfFalse, 2),
				new BinaryStatement(Instruction.Add, Register.R0, Register.R1, Register.R0)
			}).Memory.Registers[Register.R0].Value, Is.EqualTo(expected));

	[TestCase(Instruction.Add)]
	[TestCase(Instruction.GreaterThan)]
	public void OperandsRequired(Instruction instruction) =>
		Assert.That(
			() => vm.Execute(new Statement[] { new BinaryStatement(instruction, Register.R0) }),
			Throws.InstanceOf<VirtualMachine.OperandsRequired>());
}