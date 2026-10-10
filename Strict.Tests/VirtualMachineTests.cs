using System.Reflection;
using Strict.Bytecode;
using Strict.Bytecode.Instructions;
using Strict.Bytecode.Tests;
using Strict.Expressions;
using Strict.Language;
using Strict.Language.Tests;
using Strict.Optimizers;
using Type = Strict.Language.Type;

namespace Strict.Tests;

public sealed class VirtualMachineTests : TestBytecode
{
	private static VirtualMachine ExecuteVm(List<Instruction> instructions,
		IReadOnlyDictionary<string, ValueInstance>? initialVariables = null)
	{
		var binary = BinaryExecutable.CreateForEntryInstructions(TestPackage.Instance, instructions);
		return new VirtualMachine(binary).Execute(binary.EntryPoint, initialVariables);
	}

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
		var instructions = new BinaryGenerator(GenerateMethodCallFromSource(nameof(ReturnEnum),
			nameof(ReturnEnum) + "(5).GetMonday", "has dummy Number", "GetMonday Number",
			"\tDays.Monday")).Generate();
		var result = new VirtualMachine(instructions).Execute(initialVariables: null).Returns;
		Assert.That(result!.Value.Number, Is.EqualTo(1));
	}

	[Test]
	public void EnumIfConditionComparison()
	{
		CreateSampleEnum();
		var instructions = new BinaryGenerator(GenerateMethodCallFromSource(
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
		var result = new VirtualMachine(instructions).Execute(initialVariables: null).Returns!;
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
		var result = ExecuteVm(BuildInstructions(inputs, operation)).Memory.Registers[Register.R1];
		var actual = expected is string
			? (object)result.Text
			: result.Number;
		Assert.That(actual, Is.EqualTo(expected));
	}

	private static List<Instruction> BuildInstructions(IReadOnlyList<object> inputs,
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
		Assert.That(ExecuteVm([
			new LoadConstantInstruction(Register.R0, Number(5))
		]).Memory.Registers[Register.R0].Number, Is.EqualTo(5));

	[Test]
	public void SetAndAdd() =>
		Assert.That(ExecuteVm([
			new LoadConstantInstruction(Register.R0, Number(10)),
			new LoadConstantInstruction(Register.R1, Number(5)),
			new BinaryInstruction(InstructionType.Add, Register.R0, Register.R1, Register.R2)
		]).Memory.Registers[Register.R2].Number, Is.EqualTo(15));

	[Test]
	public void AddFiveTimes() =>
		Assert.That(ExecuteVm([
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
		var instructions = new BinaryGenerator(GenerateMethodCallFromSource("ArithmeticFunction",
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
		Assert.That(
			new VirtualMachine(instructions).Execute(initialVariables: null).Returns!.Value.Number,
			Is.EqualTo(expectedResult));
	}

	[Test]
	public void AccessListByIndex()
	{
		var instructions = new BinaryGenerator(GenerateMethodCallFromSource(nameof(AccessListByIndex),
			nameof(AccessListByIndex) + "(1, 2, 3, 4, 5).Get(2)", "has numbers",
			"Get(index Number) Number", "\tnumbers(index)")).Generate();
		Assert.That(
			new VirtualMachine(instructions).Execute(initialVariables: null).Returns!.Value.Number,
			Is.EqualTo(3));
	}

	[Test]
	public void AccessListByIndexNonNumberType()
	{
		var instructions = new BinaryGenerator(GenerateMethodCallFromSource(
			nameof(AccessListByIndexNonNumberType),
			nameof(AccessListByIndexNonNumberType) + "(\"1\", \"2\", \"3\", \"4\", \"5\").Get(2)",
			"has texts", "Get(index Number) Text", "\ttexts(index)")).Generate();
		Assert.That(
			new VirtualMachine(instructions).Execute(initialVariables: null).Returns!.Value.Text,
			Is.EqualTo("3"));
	}

	[Test]
	public void FlatBackedListLengthDoesNotMaterializeItems()
	{
		using var pointType = new Type(TestPackage.Instance,
			new TypeLines(nameof(FlatBackedListLengthDoesNotMaterializeItems), "has xValue Number",
				"has yValue Number")).ParseMembersAndMethods(new MethodExpressionParser());
		var numberType = TestPackage.Instance.GetType(Type.Number);
		var listType = TestPackage.Instance.GetListImplementationType(pointType);
		var point = new ValueInstance(pointType,
			[new ValueInstance(numberType, 1), new ValueInstance(numberType, 2)]);
		var backedList = new ValueInstance(listType, point, 3);
		var itemsField = typeof(ValueArrayInstance).GetField("items",
			BindingFlags.Instance | BindingFlags.NonPublic);
		Assert.That(itemsField, Is.Not.Null);
		Assert.That(itemsField!.GetValue(backedList.List), Is.Null);
		var vm = new VirtualMachine(TestPackage.Instance);
		var tryGetNativeLength = typeof(VirtualMachine).GetMethod("TryGetNativeLength",
			BindingFlags.Instance | BindingFlags.NonPublic);
		Assert.That(tryGetNativeLength, Is.Not.Null);
		object?[] arguments = [backedList, "Length", null];
		Assert.That(tryGetNativeLength!.Invoke(vm, arguments), Is.EqualTo(true));
		Assert.That(((ValueInstance)arguments[2]!).Number, Is.EqualTo(3));
		Assert.That(itemsField.GetValue(backedList.List), Is.Null);
	}

	[Test]
	public void ReduceButGrowLoopExample() =>
		Assert.That(ExecuteVm([
			new StoreVariableInstruction(Number(10), "number"),
			new StoreVariableInstruction(Number(1), "result"),
			new StoreVariableInstruction(Number(2), "multiplier"),
			new LoadVariableToRegister(Register.R0, "number"),
			new LoopBeginInstruction(Register.R0),
			new LoadVariableToRegister(Register.R2, "result"),
			new LoadVariableToRegister(Register.R3, "multiplier"),
			new BinaryInstruction(InstructionType.Multiply, Register.R2, Register.R3, Register.R4),
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
		var instructions = new BinaryGenerator(GenerateMethodCallFromSource(programName,
			methodCall, code)).Generate();
		var result = new VirtualMachine(instructions).Execute(initialVariables: null).Returns!.Value;
		var actual = expected is string
			? (object)result.Text
			: result.Number;
		Assert.That(actual, Is.EqualTo(expected));
	}

	[Test]
	public void AutoTypeTextSkipsDefaultAndConstant()
	{
		var instructions = new BinaryGenerator(GenerateMethodCallFromSource(
				nameof(AutoTypeTextSkipsDefaultAndConstant),
				nameof(AutoTypeTextSkipsDefaultAndConstant) + "(0.25, 0.25, 0.25, 1).Format",
				"has red Number", "has green Number", "has blue Number", "has alpha = 1",
				"constant max = 1",
				"Format Text", "\tAutoTypeTextSkipsDefaultAndConstant(red, green, blue, alpha) to Text")).
			Generate();
		Assert.That(
			new VirtualMachine(instructions).Execute(initialVariables: null).Returns!.Value.Text,
			Is.EqualTo("(0.25, 0.25, 0.25)"));
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
			// @formatter:on
		}
	} //ncrunch: no coverage end

	[TestCaseSource(nameof(MethodCallTests))]
	// ReSharper disable TooManyArguments, makes below tests easier
	public void MethodCall(string programName, string methodCall, string[] source, object expected)
	{
		var instructions =
			new BinaryGenerator(GenerateMethodCallFromSource(programName, methodCall, source)).Generate();
		Assert.That(
			new VirtualMachine(instructions).Execute(initialVariables: null).Returns!.Value.Number,
			Is.EqualTo(expected));
	}

	[TestCase(3, true)]
	[TestCase(5, false)]
	[TestCase(2, false)]
	public void ComparisonsAsValuesInLogicalExpression(int number, bool expected)
	{
		var instructions = new BinaryGenerator(GenerateMethodCallFromSource("InRangeCheck",
			"InRangeCheck(" + number + ").Check", "has number", "Check Boolean",
			"\tnumber >= 3 and number < 5")).Generate();
		Assert.That(new VirtualMachine(instructions).Execute(initialVariables: null).Returns!.Value.Boolean,
			Is.EqualTo(expected));
	}

	[TestCase(4, "big")]
	[TestCase(2, "small")]
	[TestCase(7, "listed")]
	public void IfConditionsWithGreaterOrEqualAndIn(int number, string expected)
	{
		var instructions = new BinaryGenerator(GenerateMethodCallFromSource("IfCompareCheck",
			"IfCompareCheck(" + number + ").Check", "has number", "Check Text", "\tif number is in (7, 8)",
			"\t\treturn \"listed\"", "\tif number >= 3", "\t\treturn \"big\"", "\t\"small\"")).Generate();
		Assert.That(new VirtualMachine(instructions).Execute(initialVariables: null).Returns!.Value.Text,
			Is.EqualTo(expected));
	}

	[TestCase(1, "\"one\"")]
	[TestCase(2, "\"other\"")]
	public void ConditionalTextExpressionHasTextType(int number, string expected)
	{
		var instructions = new BinaryGenerator(GenerateMethodCallFromSource("TextPicker",
			"TextPicker(" + number + ").Pick", "has number", "Pick Text",
			"\tnumber is 1 then \"one\" else \"other\"")).Generate();
		Assert.That(new VirtualMachine(instructions).Execute(initialVariables: null).Returns!.Value.
			ToExpressionCodeString(true), Is.EqualTo(expected));
	}

	[Test]
	public void IfAndElseTest()
	{
		var instructions = new BinaryGenerator(GenerateMethodCallFromSource("IfAndElseTest",
			"IfAndElseTest(3).IsEven", "has number", "IsEven Text", "\tmutable result = \"\"",
			"\tif number > 10", "\t\tresult = \"Number is more than 10\"", "\t\treturn result", "\telse",
			"\t\tresult = \"Number is less or equal than 10\"", "\t\treturn result")).Generate();
		Assert.That(
			new VirtualMachine(instructions).Execute(initialVariables: null).Returns!.Value.Text,
			Is.EqualTo("Number is less or equal than 10"));
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
	[TestCase("RemoveList(\"s\", \"b\", \"s\").Remove", "s s", "RemoveList",
		new[]
		{
			"has texts", "Remove Texts", "\tmutable textList = texts", "\ttextList.Remove(\"b\")",
			"\ttextList"
		})]
	[TestCase("RemoveListMultiple(\"s\", \"b\", \"s\").Remove", "b", "RemoveListMultiple",
		new[]
		{
			"has texts", "Remove Texts", "\tmutable textList = texts", "\ttextList.Remove(\"s\")",
			"\ttextList"
		})]
	public void ExecuteListBinaryOperations(string methodCall, object expectedResult,
		string programName, params string[] code)
	{
		var instructions = new BinaryGenerator(GenerateMethodCallFromSource(programName,
			methodCall, code)).Generate();
		var result = new VirtualMachine(instructions).Execute(initialVariables: null).Returns!.Value;
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
		var instructions = new BinaryGenerator(GenerateMethodCallFromSource(programName,
			methodCall, code)).Generate();
		var result = new VirtualMachine(instructions).Execute(initialVariables: null).Returns!.Value;
		Assert.That(result.ToExpressionCodeString(), Is.EqualTo(expectedResult));
	}

	[Test]
	public void ExecuteOptimizedRunMethodWithProgramArguments()
	{
		var programType = new Type(type.Package,
				new TypeLines(nameof(ExecuteOptimizedRunMethodWithProgramArguments), "has logger",
					"Run(numbers)", "\tlogger.Log(numbers.Sum)")).
			ParseMembersAndMethods(new MethodExpressionParser());
		var runMethod = programType.Methods.Single(m => m.Name == Method.Run);
		var binary = BinaryGenerator.GenerateFromRunMethods(runMethod, [runMethod]);
		new AllInstructionOptimizers().Optimize(binary);
		using var consoleWriter = new StringWriter();
		var rememberConsole = Console.Out;
		Console.SetOut(consoleWriter);
		try
		{
			new VirtualMachine(binary).Execute(initialVariables: new Dictionary<string, ValueInstance>
			{
				[runMethod.Parameters[0].Name] = new(runMethod.Parameters[0].Type,
					[Number(5), Number(10), Number(20)])
			});
		}
		finally
		{
			Console.SetOut(rememberConsole);
		}
		Assert.That(consoleWriter.ToString(), Does.Contain("35"));
	}

	[TestCase("NumbersAdder(5).AddNumberToList", "1 2 3 5", "has number", "AddNumberToList Numbers",
		"\tmutable numbers = (1, 2, 3)", "\tnumbers.Add(number)", "\tnumbers")]
	public void CollectionAdd(string methodCall, string expected, params string[] code)
	{
		var instructions =
			new BinaryGenerator(GenerateMethodCallFromSource("NumbersAdder", methodCall, code)).
				Generate();
		var result = ExpressionListToSpaceSeparatedString(instructions);
		Assert.That(result.TrimEnd(), Is.EqualTo(expected));
	}

	private static string ExpressionListToSpaceSeparatedString(BinaryExecutable binary)
	{
		var result = new VirtualMachine(binary).Execute(initialVariables: null).Returns!.Value;
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
			new VirtualMachine(new BinaryGenerator(GenerateMethodCallFromSource(nameof(DictionaryAdd),
					"DictionaryAdd(5).RemoveFromDictionary", code)).Generate()).
				Execute(initialVariables: null).
				Memory.Variables["values"].GetDictionaryItems().Count, Is.EqualTo(1));
	}

	[Test]
	public void ErrorConstantInMethodBodyRunsOnVirtualMachine() =>
		Assert.That(
			new VirtualMachine(new BinaryGenerator(GenerateMethodCallFromSource(
				nameof(ErrorConstantInMethodBodyRunsOnVirtualMachine),
				nameof(ErrorConstantInMethodBodyRunsOnVirtualMachine) + "(5).Checked", "has number",
				"Checked Number", "\tconstant Invalid = Error",
				"\tnumber is 0 then Invalid else number")).Generate()).Execute().Returns!.Value.Number,
			Is.EqualTo(5));

	[Test]
	public void MethodCallLeftOfNestedBinaryIsInvoked() =>
		Assert.That(
			new VirtualMachine(new BinaryGenerator(GenerateMethodCallFromSource(
					nameof(MethodCallLeftOfNestedBinaryIsInvoked),
					nameof(MethodCallLeftOfNestedBinaryIsInvoked) + "(0).Scaled(8.5)", "has number",
					"Scaled(other Number) Number", "\tother.Floor / 2 ^ 2")).Generate()).
				Execute().Returns!.Value.Number, Is.EqualTo(2));

	[Test]
	public void TextReturningLoopConcatenatesValues() =>
		Assert.That(
			new VirtualMachine(new BinaryGenerator(GenerateMethodCallFromSource(
					nameof(TextReturningLoopConcatenatesValues),
					nameof(TextReturningLoopConcatenatesValues) + "(\"a\").Joined", "has text",
					"Joined Text", "\tfor (text, \"b\")", "\t\tvalue + \",\"")).Generate()).
				Execute().Returns!.Value.Text, Is.EqualTo("a,b,"));

	[Test]
	public void ProfileCountsInvokedMethods()
	{
		var machine = new VirtualMachine(new BinaryGenerator(GenerateMethodCallFromSource(
			nameof(ProfileCountsInvokedMethods), nameof(ProfileCountsInvokedMethods) + "(3).Twice",
			"has number", "Twice Number", "\tDoubled(number) + Doubled(1)",
			"Doubled(amount Number) Number", "\tamount * 2")).Generate()) { Profile = [] };
		machine.Execute();
		Assert.That(machine.Profile!.Single(entry =>
			entry.Key.EndsWith(".Doubled", StringComparison.Ordinal)).Value.Calls, Is.EqualTo(2));
	}

	[Test]
	public void ListContainsRunsNatively()
	{
		var machine = new VirtualMachine(new BinaryGenerator(GenerateMethodCallFromSource(
			nameof(ListContainsRunsNatively), nameof(ListContainsRunsNatively) + "(\"b\").Known",
			"has name Text", "Known Boolean", "\tname is in (\"a\", \"b\", name + \"c\")")).Generate())
		{
			Profile = []
		};
		Assert.That(machine.Execute().Returns!.Value.Boolean, Is.True);
		Assert.That(machine.Profile!.Keys, Has.None.EndsWith(".in"));
	}

	[Test]
	public void ListIndexRunsNatively()
	{
		var machine = new VirtualMachine(new BinaryGenerator(GenerateMethodCallFromSource(
			nameof(ListIndexRunsNatively), nameof(ListIndexRunsNatively) + "(\"b\").Position",
			"has name Text", "Position Number",
			"\t(\"a\", name, name + \"c\").Index(name)")).Generate())
		{
			Profile = []
		};
		Assert.That(machine.Execute().Returns!.Value.Number, Is.EqualTo(1));
		Assert.That(machine.Profile!.Keys, Has.None.EndsWith(".Index"));
	}

	[Test]
	public void ListCountWithArgumentCountsMatches() =>
		Assert.That(new VirtualMachine(new BinaryGenerator(GenerateMethodCallFromSource(
				nameof(ListCountWithArgumentCountsMatches), nameof(ListCountWithArgumentCountsMatches) +
				"(1).Matches", "has number", "Matches Number", "\t(number, 2, number).Count(number)")).
			Generate()).Execute().Returns!.Value.Number, Is.EqualTo(2));

	[Test]
	public void NumberFloorRunsNatively()
	{
		var machine = new VirtualMachine(new BinaryGenerator(GenerateMethodCallFromSource(
			nameof(NumberFloorRunsNatively), nameof(NumberFloorRunsNatively) + "(-2.5).Rounded",
			"has number", "Rounded Number", "\tnumber.Floor")).Generate())
		{
			Profile = []
		};
		Assert.That(machine.Execute().Returns!.Value.Number, Is.EqualTo(-3));
		Assert.That(machine.Profile!.Keys, Has.None.EndsWith(".Floor"));
	}

	[Test]
	public void NumberWrapperValueIsUsedAsNumberInArithmetic() =>
		Assert.That(RunOnNumberWrapper(nameof(NumberWrapperValueIsUsedAsNumberInArithmetic),
			"value / 2").Number, Is.EqualTo(90));

	private ValueInstance RunOnNumberWrapper(string typeName, string calculation) =>
		new VirtualMachine(new BinaryGenerator(GenerateMethodCallFromSource(typeName,
			typeName + "(0).Run", "has number", "Run Number", "\t" + typeName + "(180).Calculate",
			"Calculate Number", "\t" + calculation)).Generate()).Execute().Returns!.Value;

	[TestCase("Plus", "90 + value", "270")]
	[TestCase("Minus", "value - 90", "90")]
	[TestCase("Times", "2 * value", "360")]
	public void ArithmeticWithNumberWrapperGivesNumber(string operation, string calculation,
		string expected) =>
		Assert.That(RunOnNumberWrapper(nameof(ArithmeticWithNumberWrapperGivesNumber) + operation,
			calculation).ToExpressionCodeString(), Is.EqualTo(expected));

	private static Invoke CreateFromInvoke(Type targetType, Register register)
	{
		var fromMethod = targetType.FindMethod(Method.From, []);
		var parameterNames = fromMethod != null
			? fromMethod.Parameters.Select(parameter => parameter.Name).ToArray()
			: Array.Empty<string>();
		return new Invoke(register,
			new InvokeMethodInfo(targetType.FullName, Method.From, parameterNames, targetType.Name, [],
				null));
	}

	[Test]
	public void CreateEmptyDictionaryFromConstructor()
	{
		var dictionaryType = TestPackage.Instance.GetType(Type.Dictionary).
			GetGenericImplementation(NumberType, NumberType);
		var instructions = new List<Instruction> { CreateFromInvoke(dictionaryType, Register.R0) };
		var result = ExecuteVm(instructions).Memory.Registers[Register.R0];
		Assert.That(result.IsDictionary, Is.True);
		Assert.That(result.GetDictionaryItems().Count, Is.EqualTo(0));
	}

	[Test]
	public void DictionaryGet()
	{
		string[] code =
		[
			"has number",
			"AddToDictionary Number",
			"\tmutable values = Dictionary(Number, Number)",
			"\tvalues.Add(1, number)",
			"\tnumber"
		];
		var instructions = new BinaryGenerator(GenerateMethodCallFromSource(nameof(DictionaryGet),
			"DictionaryGet(5).AddToDictionary", code)).Generate();
		var values = new VirtualMachine(instructions).Execute(initialVariables: null).Memory.
			Variables["values"].GetDictionaryItems();
		Assert.That(GetDictionaryValue(values, 1), Is.EqualTo("5"));
	}

	[Test]
	public void DictionaryRemove()
	{
		string[] code =
		[
			"has number",
			"AddToDictionary Number",
			"\tmutable values = Dictionary(Number, Number)",
			"\tvalues.Add(1, number)",
			"\tvalues.Add(2, number + 10)",
			"\tnumber"
		];
		var instructions = new BinaryGenerator(GenerateMethodCallFromSource(nameof(DictionaryRemove),
			"DictionaryRemove(5).AddToDictionary", code)).Generate();
		var values = new VirtualMachine(instructions).Execute(initialVariables: null).Memory.
			Variables["values"].GetDictionaryItems();
		Assert.That(GetDictionaryValue(values, 2), Is.EqualTo("15"));
	}

	private static string GetDictionaryValue(IReadOnlyDictionary<ValueInstance, ValueInstance> values,
		double key) =>
		values.First(entry => entry.Key.Number == key).Value.ToExpressionCodeString();

	[Test]
	public void ReturnWithinALoop()
	{
		var source = new[] { "has number", "GetAll Number", "\tfor number", "\t\tvalue" };
		var instructions = new BinaryGenerator(GenerateMethodCallFromSource(nameof(ReturnWithinALoop),
			"ReturnWithinALoop(5).GetAll", source)).Generate();
		Assert.That(
			() => new VirtualMachine(instructions).Execute(initialVariables: null).Returns!.Value.Number,
			Is.EqualTo(0 + 1 + 2 + 3 + 4));
	}

	[Test]
	public void ReverseWithRange()
	{
		var source = new[]
		{
			"has numbers", "Reverse Numbers", "\tmutable result = Numbers",
			"\tlet len = numbers.Length - 1", "\tfor Range(len, -1)", "\t\tresult.Add(numbers(index))",
			"\tresult"
		};
		var instructions = new BinaryGenerator(GenerateMethodCallFromSource(nameof(ReverseWithRange),
			"ReverseWithRange(1, 2, 3).Reverse", source)).Generate();
		Assert.That(() => ExpressionListToSpaceSeparatedString(instructions), Is.EqualTo("3 2 1 "));
	}

	[Test]
	public void LoopOverRangeReturnedByMethod()
	{
		var source = new[]
		{
			"has number", "Down Numbers", "\tmutable result = Numbers", "\tfor Range(0, number).Reverse",
			"\t\tresult.Add(index)", "\tresult"
		};
		var instructions = new BinaryGenerator(GenerateMethodCallFromSource(
			nameof(LoopOverRangeReturnedByMethod), nameof(LoopOverRangeReturnedByMethod) + "(3).Down",
			source)).Generate();
		Assert.That(() => ExpressionListToSpaceSeparatedString(instructions), Is.EqualTo("2 1 0 "));
	}

	[Test]
	public void PowerOfNumbers()
	{
		var source = new[] { "has number", "Squared Number", "\tnumber ^ 2" };
		var instructions = new BinaryGenerator(GenerateMethodCallFromSource(nameof(PowerOfNumbers),
			nameof(PowerOfNumbers) + "(3).Squared", source)).Generate();
		Assert.That(
			() => new VirtualMachine(instructions).Execute(initialVariables: null).Returns!.Value.Number,
			Is.EqualTo(9));
	}

	[Test]
	public void MemberOfComputedInstance()
	{
		var source = new[] { "has text", "Code Number", "\tCharacter(text.Length + 64).number + 1" };
		var instructions = new BinaryGenerator(GenerateMethodCallFromSource(
			nameof(MemberOfComputedInstance), nameof(MemberOfComputedInstance) + "(\"A\").Code",
			source)).Generate();
		Assert.That(
			() => new VirtualMachine(instructions).Execute(initialVariables: null).Returns!.Value.Number,
			Is.EqualTo(66));
	}

	[Test]
	public void ConstantCreatedByConstructor()
	{
		var source = new[]
		{
			"has number", "constant Window = Range(2, 5)", "WindowStart Number",
			"\tWindow.Start + number"
		};
		var instructions = new BinaryGenerator(GenerateMethodCallFromSource(
			nameof(ConstantCreatedByConstructor), nameof(ConstantCreatedByConstructor) + "(3).WindowStart",
			source)).Generate();
		Assert.That(
			() => new VirtualMachine(instructions).Execute(initialVariables: null).Returns!.Value.Number,
			Is.EqualTo(5));
	}

	[Test]
	public void NegativeListIndexCountsFromEnd()
	{
		var source = new[] { "has numbers", "LastNumber Number", "\tnumbers(-1)" };
		var instructions = new BinaryGenerator(GenerateMethodCallFromSource(
			nameof(NegativeListIndexCountsFromEnd),
			nameof(NegativeListIndexCountsFromEnd) + "(1, 2, 3).LastNumber", source)).Generate();
		Assert.That(
			() => new VirtualMachine(instructions).Execute(initialVariables: null).Returns!.Value.Number,
			Is.EqualTo(3));
	}

	[Test]
	public void ListCallOnComputedList()
	{
		var source = new[]
		{
			"has numbers", "Doubled Number", "\tListCallOnComputedList(numbers + numbers).numbers(4)"
		};
		var instructions = new BinaryGenerator(GenerateMethodCallFromSource(
			nameof(ListCallOnComputedList), nameof(ListCallOnComputedList) + "(1, 2, 3).Doubled",
			source)).Generate();
		Assert.That(
			() => new VirtualMachine(instructions).Execute(initialVariables: null).Returns!.Value.Number,
			Is.EqualTo(2));
	}

	[Test]
	public void OptimizedRangeFromComputedArguments()
	{
		var source = new[]
		{
			"has number", "Bounds Numbers", "\tlet range = Range(number - 1, number - 2)",
			"\t(range.Start, range.ExclusiveEnd)"
		};
		var binary = new BinaryGenerator(GenerateMethodCallFromSource(
			nameof(OptimizedRangeFromComputedArguments),
			nameof(OptimizedRangeFromComputedArguments) + "(5).Bounds", source)).Generate();
		new AllInstructionOptimizers().Optimize(binary);
		Assert.That(() => ExpressionListToSpaceSeparatedString(binary), Is.EqualTo("4 3 "));
	}

	[Test]
	public void ConditionalJump() =>
		Assert.That(ExecuteVm([
			new SetInstruction(Number(5), Register.R0),
			new SetInstruction(Number(1), Register.R1),
			new SetInstruction(Number(10), Register.R2),
			new BinaryInstruction(InstructionType.LessThan, Register.R2, Register.R0),
			new Jump(2, InstructionType.JumpIfTrue),
			new BinaryInstruction(InstructionType.Add, Register.R2, Register.R0, Register.R0)
		]).Memory.Registers[Register.R0].Number, Is.EqualTo(15));

	[Test]
	public void JumpIfTrueSkipsNextInstruction() =>
		Assert.That(ExecuteVm([
			new SetInstruction(Number(1), Register.R0),
			new SetInstruction(Number(1), Register.R1),
			new SetInstruction(Number(0), Register.R2),
			new BinaryInstruction(InstructionType.Equal, Register.R0, Register.R1),
			new Jump(1, InstructionType.JumpIfTrue),
			new BinaryInstruction(InstructionType.Add, Register.R0, Register.R1, Register.R2)
		]).Memory.Registers[Register.R2].Number, Is.EqualTo(0));

	[Test]
	public void JumpIfFalseSkipsNextInstruction() =>
		Assert.That(ExecuteVm([
			new SetInstruction(Number(1), Register.R0),
			new SetInstruction(Number(2), Register.R1),
			new SetInstruction(Number(0), Register.R2),
			new BinaryInstruction(InstructionType.Equal, Register.R0, Register.R1),
			new Jump(1, InstructionType.JumpIfFalse),
			new BinaryInstruction(InstructionType.Add, Register.R0, Register.R1, Register.R2)
		]).Memory.Registers[Register.R2].Number, Is.EqualTo(0));

	[TestCase(InstructionType.GreaterThan, new[] { 1, 2 }, 2 - 1)]
	[TestCase(InstructionType.LessThan, new[] { 1, 2 }, 1 + 2)]
	[TestCase(InstructionType.Equal, new[] { 5, 5 }, 5 + 5)]
	[TestCase(InstructionType.NotEqual, new[] { 5, 5 }, 5 - 5)]
	public void ConditionalJumpIfAndElse(InstructionType conditional, int[] registers,
		int expected) =>
		Assert.That(ExecuteVm([
			new SetInstruction(Number(registers[0]), Register.R0),
			new SetInstruction(Number(registers[1]), Register.R1),
			new BinaryInstruction(conditional, Register.R0, Register.R1),
			new Jump(2, InstructionType.JumpIfTrue),
			new BinaryInstruction(InstructionType.Subtract, Register.R1, Register.R0, Register.R0),
			new Jump(2, InstructionType.JumpIfFalse),
			new BinaryInstruction(InstructionType.Add, Register.R0, Register.R1, Register.R0)
		]).Memory.Registers[Register.R0].Number, Is.EqualTo(expected));

	[TestCase(InstructionType.Add)]
	[TestCase(InstructionType.GreaterThan)]
	public void OperandsRequired(InstructionType instruction) =>
		Assert.That(() => ExecuteVm([new BinaryInstruction(instruction, Register.R0)]),
			Throws.InstanceOf<VirtualMachine.OperandsRequired>());

	[Test]
	public void LoopOverTextStopsWhenIndexExceedsLength()
	{
		var text = Text("Hi");
		var loopBegin = new LoopBeginInstruction(Register.R0);
		var result = ExecuteVm([
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
			"has numbers", "CountItems Number", "\tmutable count = 0", "\tfor numbers",
			"\t\tcount = count + 1", "\tcount"
		};
		var instructions = new BinaryGenerator(GenerateMethodCallFromSource(
			nameof(LoopOverListStopsWhenIndexExceedsCount),
			$"{nameof(LoopOverListStopsWhenIndexExceedsCount)}(1, 2, 3).CountItems", source)).Generate();
		var result = new VirtualMachine(instructions).Execute(initialVariables: null).Returns!.Value.
			Number;
		Assert.That(result, Is.EqualTo(3));
	}

	[Test]
	public void MutableListGrowsInsideLoop()
	{
		var source = new[]
		{
			"has number", "Collect Numbers", "\tmutable result = List(Mutable(Number))", "\tfor number",
			"\t\tresult.Add(value * 2)", "\tresult"
		};
		var instructions = new BinaryGenerator(GenerateMethodCallFromSource(nameof(MutableListGrowsInsideLoop),
			$"{nameof(MutableListGrowsInsideLoop)}(3).Collect", source)).Generate();
		var result = new VirtualMachine(instructions).Execute(initialVariables: null).Returns!.Value;
		Assert.That(result.List.Items.Select(item => item.Number), Is.EqualTo(new[] { 0.0, 2.0, 4.0 }));
	}

	[Test]
	public void GrowingCopyOfListKeepsOriginal() =>
		Assert.That(RunSource(nameof(GrowingCopyOfListKeepsOriginal), "((1, 2)).Grow", "has numbers",
			"Grow Number", "\tmutable result = numbers", "\tresult = result + 3", "\tresult.Add(4)",
			"\tnumbers.Length").Number, Is.EqualTo(2));

	private ValueInstance RunSource(string programName, string call, params string[] source) =>
		new VirtualMachine(new BinaryGenerator(GenerateMethodCallFromSource(programName,
			programName + call, source)).Generate()).Execute(initialVariables: null).Returns!.Value;

	[Test]
	public void ListReassignedFromMemberIsChangedAsCopy() =>
		Assert.That(RunSource(nameof(ListReassignedFromMemberIsChangedAsCopy), "((1, 2)).Grow",
			"has numbers", "Grow Number", "\tmutable result = numbers + 7", "\tresult = numbers",
			"\tresult.Add(4)", "\tresult = numbers", "\tresult = result + 5", "\tnumbers.Length").Number,
			Is.EqualTo(2));

	[Test]
	public void ListTakenFromNestedListIsChangedAsCopy() =>
		Assert.That(RunSource(nameof(ListTakenFromNestedListIsChangedAsCopy), "(((1, 2), (3, 4))).Grow",
			"has rows List(Numbers)", "Grow Number", "\tmutable row = rows(0)", "\trow.Add(5)",
			"\trows(0).Length").Number, Is.EqualTo(2));

	[Test]
	public void SavedListKeepsLengthWhenOriginalGrows() =>
		Assert.That(RunSource(nameof(SavedListKeepsLengthWhenOriginalGrows), "((1, 2)).Grow",
			"has numbers", "Grow Number", "\tmutable result = numbers + 7", "\tlet saved = result",
			"\tresult.Add(4)", "\tmutable other = numbers + 8", "\tmutable firsts = numbers", "\tfor 2",
			"\t\tif index is 1", "\t\t\tother.Add(9)", "\t\tif index is 0", "\t\t\tfirsts = other",
			"\tsaved.Length + firsts.Length * 10").Number, Is.EqualTo(33));

	[Test]
	public void RemoveAsLastLineReturnsList() =>
		Assert.That(RunSource(nameof(RemoveAsLastLineReturnsList), "((1, 2)).Shrink", "has numbers",
			"Shrink Numbers", "\tmutable result = numbers + 7", "\tresult.Remove(1)").List.Items.
			Select(item => item.Number), Is.EqualTo(new[] { 2.0, 7.0 }));

	[Test]
	public void ListMinusListRemovesEachElementOnce() =>
		Assert.That(RunSource(nameof(ListMinusListRemovesEachElementOnce), "((1, 2, 2, 3)).Shrink",
			"has numbers", "Shrink Number", "\t(numbers - (2, 3)).Length").Number, Is.EqualTo(2));

	[Test]
	public void AndSkipsRightSideWhenLeftIsFalse()
	{
		var source = new[]
		{
			"has numbers", "Has(position Number) Boolean",
			"	position < numbers.Length and (numbers(position) is 1)"
		};
		var instructions = new BinaryGenerator(GenerateMethodCallFromSource(
			nameof(AndSkipsRightSideWhenLeftIsFalse),
			$"{nameof(AndSkipsRightSideWhenLeftIsFalse)}((1, 2)).Has(5)", source)).Generate();
		Assert.That(new VirtualMachine(instructions).Execute(initialVariables: null).Returns!.Value.Boolean,
			Is.False);
	}

	[Test]
	public void ListConstantOfConstructedInstanceKeepsItsValues()
	{
		var source = new[]
		{
			"has number", "constant primes = (2, 3, 5)", "Pick Number",
			$"	{nameof(ListConstantOfConstructedInstanceKeepsItsValues)}(number).Prime",
			"Prime Number", "	primes(number)"
		};
		var instructions = new BinaryGenerator(GenerateMethodCallFromSource(
			nameof(ListConstantOfConstructedInstanceKeepsItsValues),
			$"{nameof(ListConstantOfConstructedInstanceKeepsItsValues)}(1).Pick", source)).Generate();
		Assert.That(new VirtualMachine(instructions).Execute(initialVariables: null).Returns!.Value.Number,
			Is.EqualTo(3));
	}

	[Test]
	public void ToTextUsesOwnToTextMethod()
	{
		var source = new[]
		{
			"has number", "Show Text", $"	{nameof(ToTextUsesOwnToTextMethod)}(number + 1) to Text",
			"to Text", "	\"number \" + number"
		};
		var instructions = new BinaryGenerator(GenerateMethodCallFromSource(nameof(ToTextUsesOwnToTextMethod),
			$"{nameof(ToTextUsesOwnToTextMethod)}(2).Show", source)).Generate();
		Assert.That(new VirtualMachine(instructions).Execute(initialVariables: null).Returns!.Value.Text,
			Is.EqualTo("number 3"));
	}

	[Test]
	public void FailureInCalledMethodNamesCaller()
	{
		var instructions = new BinaryGenerator(GenerateMethodCallFromSource(
			nameof(FailureInCalledMethodNamesCaller), $"{nameof(FailureInCalledMethodNamesCaller)}((1, 2)).Outer",
			"has numbers", "Outer Number", "\tInner + 1", "Inner Number", "\tnumbers(5)")).Generate();
		Assert.That(() => new VirtualMachine(instructions).Execute(initialVariables: null),
			Throws.InstanceOf<InstructionExecutionFailed>().With.Message.
				Contains("called from " + nameof(FailureInCalledMethodNamesCaller)));
	}

	[Test]
	public void EndlessRecursionIsAStackOverflowError()
	{
		var source = new[]
		{
			"has number", "Deeper Number", "	if number < 0", "		return 0",
			$"	{nameof(EndlessRecursionIsAStackOverflowError)}(number + 1).Deeper"
		};
		var instructions = new BinaryGenerator(GenerateMethodCallFromSource(
			nameof(EndlessRecursionIsAStackOverflowError),
			$"{nameof(EndlessRecursionIsAStackOverflowError)}(1).Deeper", source)).Generate();
		Assert.That(() => new VirtualMachine(instructions).Execute(initialVariables: null),
			Throws.InstanceOf<VirtualMachine.StackOverflow>());
	}

	[Test]
	public void RecursiveCallInsideLoopKeepsOuterLoop()
	{
		var source = new[]
		{
			"has number", "Leaves Number", "	if number is 0", "		return 1", "	mutable total = 0",
			"	for 3", $"		total = total + {nameof(RecursiveCallInsideLoopKeepsOuterLoop)}(number - 1).Leaves",
			"	total"
		};
		var instructions = new BinaryGenerator(GenerateMethodCallFromSource(
			nameof(RecursiveCallInsideLoopKeepsOuterLoop),
			$"{nameof(RecursiveCallInsideLoopKeepsOuterLoop)}(2).Leaves", source)).Generate();
		Assert.That(new VirtualMachine(instructions).Execute(initialVariables: null).Returns!.Value.Number,
			Is.EqualTo(9));
	}

	[Test]
	public void ReturnedInlineConditionalUsesThenValue()
	{
		var source = new[]
		{
			"has number", "Pick Number", "	if number > 5", "		return number is 6 then Twice(1) else Twice(2)",
			"	0", "Twice(value Number) Number", "	value * 2"
		};
		var instructions = new BinaryGenerator(GenerateMethodCallFromSource(
			nameof(ReturnedInlineConditionalUsesThenValue),
			$"{nameof(ReturnedInlineConditionalUsesThenValue)}(6).Pick", source)).Generate();
		Assert.That(new VirtualMachine(instructions).Execute(initialVariables: null).Returns!.Value.Number,
			Is.EqualTo(2));
	}

	[Test]
	public void TextIndexOfUsesStartIndex()
	{
		var source = new[] { "has letters Text", "SecondQuote Number", "	letters.IndexOf(\"a\", 1)" };
		var instructions = new BinaryGenerator(GenerateMethodCallFromSource(nameof(TextIndexOfUsesStartIndex),
			$"{nameof(TextIndexOfUsesStartIndex)}(\"abca\").SecondQuote", source)).Generate();
		Assert.That(new VirtualMachine(instructions).Execute(initialVariables: null).Returns!.Value.Number,
			Is.EqualTo(3));
	}

	[Test]
	public void LoopOverNumberValueStartsAtZero()
	{
		var source = new[]
		{
			"has number", "Sum Number", "\tmutable sum = 0", "\tfor number", "\t\tsum = sum + value", "\tsum"
		};
		var instructions = new BinaryGenerator(GenerateMethodCallFromSource(nameof(LoopOverNumberValueStartsAtZero),
			$"{nameof(LoopOverNumberValueStartsAtZero)}(3).Sum", source)).Generate();
		Assert.That(new VirtualMachine(instructions).Execute(initialVariables: null).Returns!.Value.Number,
			Is.EqualTo(3));
	}

	[Test]
	public void LoopOverEmptyListSkipsBody()
	{
		var source = new[]
		{
			"has number", "Mapped Texts", "	constant none = List(Text)", "	for none",
			"		value + number"
		};
		var instructions = new BinaryGenerator(GenerateMethodCallFromSource(
			nameof(LoopOverEmptyListSkipsBody), $"{nameof(LoopOverEmptyListSkipsBody)}(1).Mapped",
			source)).Generate();
		var result = new VirtualMachine(instructions).Execute(initialVariables: null).Returns!.Value;
		Assert.That(result.List.Count, Is.EqualTo(0));
	}

	[Test]
	public void ListOfDynamicValuesInDeclaration()
	{
		var source = new[]
		{
			"has number", "Second Number", "\tlet pair = (number + 1, number + 2)", "\tpair(1)"
		};
		var instructions = new BinaryGenerator(GenerateMethodCallFromSource(
			nameof(ListOfDynamicValuesInDeclaration), $"{nameof(ListOfDynamicValuesInDeclaration)}(5).Second",
			source)).Generate();
		Assert.That(new VirtualMachine(instructions).Execute(initialVariables: null).Returns!.Value.
			Number, Is.EqualTo(7));
	}

	[Test]
	public void AddingListsConcatenatesWithoutChangingLeftList()
	{
		var source = new[]
		{
			"has number", "Combined Number", "\tlet first = (number, number + 1)",
			"\tlet both = first + (number + 2, number + 3)", "\tboth(3) + first.Length"
		};
		var instructions = new BinaryGenerator(GenerateMethodCallFromSource(
			nameof(AddingListsConcatenatesWithoutChangingLeftList),
			$"{nameof(AddingListsConcatenatesWithoutChangingLeftList)}(5).Combined", source)).Generate();
		Assert.That(new VirtualMachine(instructions).Execute(initialVariables: null).Returns!.Value.
			Number, Is.EqualTo(10));
	}

	[Test]
	public async Task LoopOverSizeIteratesWidthTimesHeight()
	{
		var parser = new MethodExpressionParser();
		var repositories = new Repositories(parser);
		using var package = await repositories.LoadStrictPackage("Strict/ImageProcessing");
		using var testType = new Type(package, new TypeLines(
			nameof(LoopOverSizeIteratesWidthTimesHeight),
			// @formatter: off
			"has number", "Run Number", "\tconstant width = 16", "\tconstant height = 9",
			"\tmutable image = Image(Size(width, height))", "\tfor image.Size",
			"\t\timage.Colors(index) = ColorValue(0.25, 0.25, 0.25)", "\tmutable count = 0",
			"\tfor image.Size", "\t\tif image.Colors(index) is ColorValue(0.25, 0.25, 0.25)",
			"\t\t\tcount = count + 1", "\tcount")).ParseMembersAndMethods(parser);
		// @formatter: on
		var runMethod = testType.Methods.Single(m => m.Name == Method.Run);
		var executable =
			BinaryGenerator.GenerateFromRunMethods(runMethod, [runMethod]); //TODO: extremely slow
		var result = new VirtualMachine(executable).Execute().Returns!.Value.Number;
		Assert.That(result, Is.EqualTo(16 * 9));
	}

	[Test]
	public void LoopOverSingleCharTextStopsAtEnd()
	{
		var source = new[]
		{
			"has letter Text", "CountChars Number", "\tmutable count = 0", "\tfor letter",
			"\t\tcount = count + 1", "\tcount"
		};
		var instructions = new BinaryGenerator(GenerateMethodCallFromSource(
			nameof(LoopOverSingleCharTextStopsAtEnd),
			$"{nameof(LoopOverSingleCharTextStopsAtEnd)}(\"X\").CountChars", source)).Generate();
		var result = new VirtualMachine(instructions).Execute(initialVariables: null).Returns!.Value.
			Number;
		Assert.That(result, Is.EqualTo(1));
	}

	[Test]
	public void LoopOverSingleItemListStopsAtEnd()
	{
		var numbersListType = ListType.GetGenericImplementation(NumberType);
		var singleItemList = new ValueInstance(numbersListType, [new ValueInstance(NumberType, 42)]);
		var result = ExecuteVm([
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

	[Test]
	public void NestedForLoopCanReturnListInBytecode()
	{
		var instructions = new BinaryGenerator(GenerateMethodCallFromSource(
			nameof(NestedForLoopCanReturnListInBytecode),
			$"{nameof(NestedForLoopCanReturnListInBytecode)}.Coordinates", "has number",
			"Coordinates Numbers", "\tfor 2", "\t\tfor 2", "\t\t\tindex + outer.index * 10")).Generate();
		var result = new VirtualMachine(instructions).Execute(initialVariables: null).Returns!.Value;
		Assert.That(result.ToExpressionCodeString(), Is.EqualTo("(0, 1, 10, 11)"));
	}

	[TestCase("add", 1)]
	[TestCase("subtract", 2)]
	[TestCase("other", 3)]
	public void SelectorIfReturnsCorrectCase(string operation, double expected)
	{
		var instructions = new BinaryGenerator(GenerateMethodCallFromSource(
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
		var result = new VirtualMachine(instructions).Execute(initialVariables: null).Returns!;
		Assert.That(result.Value.Number, Is.EqualTo(expected));
	}

	[Test]
	public void RoundTripInvokeWithDoubleNumberArgument()
	{
		var instructions = new BinaryGenerator(GenerateMethodCallFromSource("DoubleCalc",
			"DoubleCalc(3.14).GetHalf", "has number", "GetHalf Number", "\tnumber / 2")).Generate();
		var result = new VirtualMachine(instructions).Execute(initialVariables: null).Returns!;
		Assert.That(result.Value.Number, Is.EqualTo(3.14 / 2));
	}

	[Test]
	public void CreateInstanceWithLoggerTraitMember()
	{
		if (type.Package.FindDirectType("TypeWithLogger") == null)
			new Type(type.Package,
					new TypeLines("TypeWithLogger", "has logger", "GetZero Number", "\t0")).
				ParseMembersAndMethods(new MethodExpressionParser());
		var typeWithLogger = type.Package.FindDirectType("TypeWithLogger")!;
		var instructions = new List<Instruction> { CreateFromInvoke(typeWithLogger, Register.R0) };
		var result = ExecuteVm(instructions).Memory.Registers[Register.R0];
		Assert.That(result.TryGetValueTypeInstance(), Is.Not.Null);
	}

	[Test]
	public void CreateInstanceWithConcreteListMemberUsesEmptyListDefault()
	{
		if (type.Package.FindDirectType("HolderWithColors") == null)
		{
			new Type(type.Package, new TypeLines("Color", "has red Number", "GetZero Number", "\t0")).
				ParseMembersAndMethods(new MethodExpressionParser());
			new Type(type.Package,
					new TypeLines("HolderWithColors", "mutable colors", "GetFirst Number",
						"\tcolors(0).red")).
				ParseMembersAndMethods(new MethodExpressionParser());
		}
		var holderType = type.Package.FindDirectType("HolderWithColors")!;
		var fromMethod = holderType.FindMethod(Method.From, []);
		Assert.That(fromMethod, Is.Not.Null);
		Assert.That(fromMethod!.Parameters, Has.Count.EqualTo(1));
		Assert.That(fromMethod.Parameters[0].Type.IsList, Is.True);
		Assert.That(fromMethod.Parameters[0].DefaultValue?.ToString(), Is.Null);
		var result = ExecuteVm([
			CreateFromInvoke(holderType, Register.R0)
		]).Memory.Registers[Register.R0];
		var colors = result.TryGetValueTypeInstance()!["colors"];
		Assert.That(colors.IsList, Is.True);
		Assert.That(colors.List.Items, Is.Empty);
	}

	[Test]
	public void AddHundredElementsToMutableList()
	{
		var source = new[]
		{
			"has count Number", "AddMany Numbers", "\tmutable myList = (0)", "\tfor count",
			"\t\tmyList = myList + value", "\tmyList"
		};
		var instructions = new BinaryGenerator(GenerateMethodCallFromSource(
			nameof(AddHundredElementsToMutableList),
			$"{nameof(AddHundredElementsToMutableList)}(100).AddMany", source)).Generate();
		var result = new VirtualMachine(instructions).Execute(initialVariables: null).Returns!.Value;
		Assert.That(result.List.Items.Count, Is.EqualTo(101));
	}

	[Test]
	public void ExecuteRunUsesBinaryEntryPoint()
	{
		var binary = new BinaryGenerator(GenerateMethodCallFromSource("VmExecuteExpressionMethodType",
			"VmExecuteExpressionMethodType(10, 5).Calculate", "has First Number", "has Second Number",
			"Calculate Number", "\tFirst + Second")).Generate();
		Assert.That(new VirtualMachine(binary).Execute().Returns!.Value.Number, Is.EqualTo(15));
	}

	[Test]
	public void ExecuteLoadedBinaryPreservesNestedForLoopBehavior()
	{
		var binary = new BinaryGenerator(GenerateMethodCallFromSource(
			nameof(ExecuteLoadedBinaryPreservesNestedForLoopBehavior),
			$"{nameof(ExecuteLoadedBinaryPreservesNestedForLoopBehavior)}(3, 2).CountAll",
			"has Width Number", "has Height Number", "CountAll Number", "\tmutable total = Width - Width",
			"\tfor Height", "\t\tfor Width", "\t\t\ttotal = total + 1", "\ttotal")).Generate();
		var filePath = Path.Combine(Path.GetTempPath(), Guid.NewGuid() + BinaryExecutable.Extension);
		try
		{
			binary.Serialize(filePath);
			var loadedBinary = new BinaryExecutable(filePath, TestPackage.Instance);
			Assert.That(new VirtualMachine(loadedBinary).Execute().Returns!.Value.Number, Is.EqualTo(6));
		}
		finally
		{
			if (File.Exists(filePath))
				File.Delete(filePath);
		}
	}

	[Test]
	public void ExecuteLoadedBinaryPreservesNestedIteratorAggregation()
	{
		const string ProgramName = "NestedIteratorAggregation";
		var binary = new BinaryGenerator(GenerateMethodCallFromSource(ProgramName,
			$"{ProgramName}(3, 2).All", "has Width Number", "has Height Number", "All Numbers",
			"\tfor Height", "\t\tfor Width", "\t\t\tindex")).Generate();
		var filePath = Path.Combine(Path.GetTempPath(), Guid.NewGuid() + BinaryExecutable.Extension);
		try
		{
			binary.Serialize(filePath);
			var loadedBinary = new BinaryExecutable(filePath, TestPackage.Instance);
			Assert.That(
				new VirtualMachine(loadedBinary).Execute().Returns!.Value.ToExpressionCodeString(),
				Is.EqualTo("(0, 1, 2, 0, 1, 2)"));
		}
		finally
		{
			if (File.Exists(filePath))
				File.Delete(filePath);
		}
	}

	[Test]
	public async Task ExecuteLoadedBinaryPreservesAdjustBrightnessColorComputation()
	{
		var repositories = new Repositories(new MethodExpressionParser());
		using var package = await repositories.LoadStrictPackage(nameof(Strict) +
			Context.ParentSeparator + "ImageProcessing");
		var adjustBrightness = package.GetType("AdjustBrightness");
		var colorValue = package.GetType("ColorValue");
		var zero = new Number(package, 0);
		var brightness = new Number(package, 0.25);
		var colorCall = new MethodCall(colorValue.FindMethod(Method.From, [zero, zero, zero])!, null,
			[zero, zero, zero]);
		var adjustBrightnessCall = new MethodCall(
			adjustBrightness.FindMethod(Method.From, [brightness])!, null, [brightness]);
		var getBrightnessAdjustedColor =
			adjustBrightness.FindMethod("GetBrightnessAdjustedColor", [colorCall])!;
		var methodCall = new MethodCall(getBrightnessAdjustedColor, adjustBrightnessCall, [colorCall]);
		var binary = new BinaryGenerator(methodCall).Generate();
		var filePath = Path.Combine(Path.GetTempPath(), Guid.NewGuid() + BinaryExecutable.Extension);
		try
		{
			binary.Serialize(filePath);
			var loadedBinary = new BinaryExecutable(filePath, package);
			Assert.That(
				new VirtualMachine(loadedBinary).Execute().Returns!.Value.ToExpressionCodeString(),
				Is.EqualTo("(0.25, 0.25, 0.25)"));
		}
		finally
		{
			if (File.Exists(filePath))
				File.Delete(filePath);
		}
	}

	[Test]
	public void ExecuteExpressionRunsProvidedBinaryMethod()
	{
		var binary = new BinaryGenerator(GenerateMethodCallFromSource("VmExecuteExpressionMethodType",
			"VmExecuteExpressionMethodType(10, 5).Calculate", "has First Number", "has Second Number",
			"Calculate Number", "\tFirst + Second")).Generate();
		Assert.That(new VirtualMachine(binary).Execute().Returns!.Value.Number, Is.EqualTo(15));
	}

	[Test]
	public void InvokeUsesPrecompiledMethodInstructionsFromBinaryExecutable()
	{
		var binary = new BinaryGenerator(GenerateMethodCallFromSource("InvokePrecompiledCall",
			"InvokePrecompiledCall(10, 5).Calculate", "has First Number", "has Second Number",
			"Calculate Number", "\tFirst + Second")).Generate();
		var vm = new VirtualMachine(binary);
		Assert.That(vm.Execute().Returns!.Value.Number, Is.EqualTo(15));
	}

	[Test]
	public void RunFibonacciVm()
	{
		var instructions = new BinaryGenerator(GenerateMethodCallFromSource("Fibonacci",
				"Fibonacci(10).GetNthFibonacci", "has number", "GetNthFibonacci Number",
				"\tmutable first = 1", "\tmutable second = 1", "\tfor Range(2, number)",
				"\t\tlet next = first + second", "\t\tfirst = second", "\t\tsecond = next", "\tsecond")).
			Generate();
		Assert.That(
			new VirtualMachine(instructions).Execute(initialVariables: null).Returns!.Value.Number,
			Is.EqualTo(55));
	}

	[Test]
	public void ExecuteFieldLoadInstruction()
	{
		using var pointType = new Type(TestPackage.Instance,
				new TypeLines(nameof(ExecuteFieldLoadInstruction), "has xValue Number",
					"has yValue Number")).
			ParseMembersAndMethods(new MethodExpressionParser());
		var point = new ValueInstance(pointType,
			[new ValueInstance(NumberType, 3), new ValueInstance(NumberType, 7)]);
		var result = ExecuteVm([
			new SetInstruction(point, Register.R0),
			new FieldLoadInstruction(Register.R1, Register.R0, "xValue"),
			new ReturnInstruction(Register.R1)
		]);
		Assert.That(result.Returns!.Value.Number, Is.EqualTo(3));
	}

	[Test]
	public void ExecuteConstructValueTypeInstruction()
	{
		using var pointType = new Type(TestPackage.Instance,
			new TypeLines(nameof(ExecuteConstructValueTypeInstruction), "has xValue Number",
				"has yValue Number")).ParseMembersAndMethods(new MethodExpressionParser());
		var result = ExecuteVm([
			new SetInstruction(Number(3), Register.R0),
			new SetInstruction(Number(7), Register.R1),
			new ConstructValueTypeInstruction(Register.R2, pointType, [Register.R0, Register.R1]),
			new ReturnInstruction(Register.R2)
		]);
		var typeInstance = result.Returns!.Value.TryGetValueTypeInstance()!;
		Assert.That(typeInstance.Values[0].Number, Is.EqualTo(3));
		Assert.That(typeInstance.Values[1].Number, Is.EqualTo(7));
	}

	[Test]
	public void AddScalarToRightListAppendsElement()
	{
		var numbersListType = ListType.GetGenericImplementation(NumberType);
		var right = new ValueInstance(numbersListType,
			[new ValueInstance(NumberType, 2), new ValueInstance(NumberType, 3)]);
		var result = ExecuteVm([
			new SetInstruction(Number(1), Register.R0),
			new SetInstruction(right, Register.R1),
			new BinaryInstruction(InstructionType.Add, Register.R0, Register.R1, Register.R2),
			new ReturnInstruction(Register.R2)
		]);
		Assert.That(result.Returns!.Value.List.Items.Count, Is.EqualTo(3));
	}
}