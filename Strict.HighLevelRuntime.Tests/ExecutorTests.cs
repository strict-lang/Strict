using Strict.Expressions;
using Strict.Language;
using Strict.Language.Tests;
using Type = Strict.Language.Type;

namespace Strict.HighLevelRuntime.Tests;

public sealed class ExecutorTests
{
	[SetUp]
	public void CreateExecutor() =>
		executor = new Executor(TestPackage.Instance, TestBehavior.Disabled);

	private Executor executor = null!;

	[Test]
	public void MissingArgument()
	{
		using var t = CreateCalcType();
		var method = t.Methods.Single(m => m.Name == "Add");
		Assert.That(() => executor.Execute(method),
			Throws.TypeOf<Executor.MissingArgument>().With.Message.StartsWith("first"));
	}

	[Test]
	public void FromConstructorWithExistingInstanceThrows()
	{
		using var t = CreateType(nameof(FromConstructorWithExistingInstanceThrows), "has number",
			"from(number Number)", "\tvalue");
		var method = t.Methods.Single(m => m.Name == Method.From);
		var number = new ValueInstance(executor.numberType, 3);
		var instance = new ValueInstance(t, 1);
		Assert.That(() => executor.Execute(method, instance, [number]),
			Throws.InstanceOf<MethodCall.CannotCallFromConstructorWithExistingInstance>());
	}

	private static Type CreateCalcType() =>
		CreateType("Calc", "mutable last Number", "Add(first Number, second = 1) Number",
			"\tAdd(1) is 2", "\tlast = first + second");

	private static Type CreateType(string name, params string[] lines) =>
		new Type(TestPackage.Instance, new TypeLines(name, lines)).ParseMembersAndMethods(
			new MethodExpressionParser());

	[Test]
	public void UseDefaultValue()
	{
		using var t = CreateCalcType();
		var method = t.Methods.Single(m => m.Name == "Add");
		var result = executor.Execute(method, executor.noneInstance,
			[new ValueInstance(executor.numberType, 5)]);
		Assert.That(result.Number, Is.EqualTo(6));
	}

	[Test]
	public void TooManyArguments()
	{
		using var t = CreateCalcType();
		var method = t.Methods.Single(m => m.Name == "Add");
		Assert.That(() => executor.Execute(method, executor.noneInstance, [
			new ValueInstance(executor.numberType, 1),
			new ValueInstance(executor.numberType, 2),
			new ValueInstance(executor.numberType, 3)
		]), Throws.InstanceOf<Executor.TooManyArguments>().With.Message.StartsWith("Number: 3"));
	}

	[Test]
	public void ArgumentDoesNotMapToMethodParameters()
	{
		using var t = CreateType(nameof(ArgumentDoesNotMapToMethodParameters), "has number",
			"Use(number Number) Number", "\tnumber");
		var method = t.Methods.Single(m => m.Name == "Use");
		Assert.That(() => executor.Execute(method, executor.noneInstance, [executor.trueInstance]),
			Throws.InstanceOf<Executor.ArgumentDoesNotMapToMethodParameters>());
	}

	[Test]
	public void EvaluateValueAndVariableAndParameterCalls()
	{
		using var t = CreateCalcType();
		var method = t.Methods.Single(m => m.Name == "Add");
		var first = new ValueInstance(executor.numberType, 5);
		var second = new ValueInstance(executor.numberType, 7);
		var result = executor.Execute(method, executor.noneInstance, [first, second]);
		Assert.That(result.Number, Is.EqualTo(12));
	}

	[Test]
	public void EvaluateDeclaration()
	{
		using var t = CreateType(nameof(EvaluateDeclaration), "mutable last Number",
			"AddFive(number) Number", "\tconstant five = 5", "\tnumber + five");
		var method = t.Methods.Single(m => m.Name == "AddFive");
		var number = new ValueInstance(executor.numberType, 5);
		var result = executor.Execute(method, executor.noneInstance, [number]);
		Assert.That(result.Number, Is.EqualTo(10));
	}

	[Test]
	public void EvaluateAllArithmeticOperators()
	{
		var number = TestPackage.Instance.GetType(Type.Number);

		Method GetBinaryOperator(string op) =>
			number.Methods.Single(m => m.Name == op && m.Parameters.Count == 1);

		ValueInstance N(double x) => new(number, x);
		Assert.That(executor.Execute(GetBinaryOperator(BinaryOperator.Plus), N(2), [N(3)]).Number,
			Is.EqualTo(5));
		Assert.That(executor.Execute(GetBinaryOperator(BinaryOperator.Minus), N(8), [N(3)]).Number,
			Is.EqualTo(5));
		Assert.That(executor.Execute(GetBinaryOperator(BinaryOperator.Multiply), N(6), [N(7)]).Number,
			Is.EqualTo(42));
		Assert.That(executor.Execute(GetBinaryOperator(BinaryOperator.Divide), N(8), [N(2)]).Number,
			Is.EqualTo(4));
		Assert.That(executor.Execute(GetBinaryOperator(BinaryOperator.Modulate), N(8), [N(3)]).Number,
			Is.EqualTo(2));
		Assert.That(executor.Execute(GetBinaryOperator(BinaryOperator.Power), N(2), [N(3)]).Number,
			Is.EqualTo(8));
	}

	[Test]
	public void AddTwoTexts()
	{
		var text = TestPackage.Instance.GetType(Type.Text);
		var plusText = text.Methods.Single(m =>
			m.Name == BinaryOperator.Plus && m.Parameters is [{ Type.IsText: true }]);
		var result = executor.Execute(plusText, new ValueInstance("hi "), [new ValueInstance("there")]);
		Assert.That(result.Text, Is.EqualTo("hi there"));
	}

	[Test]
	public void EvaluateAllComparisonOperators()
	{
		var number = TestPackage.Instance.GetType(Type.Number);

		Method GetBinaryOperator(string op) =>
			number.Methods.Single(m => m.Name == op && m.Parameters.Count == 1);

		ValueInstance N(double x) => new(number, x);
		Assert.That(executor.Execute(GetBinaryOperator(BinaryOperator.Greater), N(5), [N(3)]),
			Is.EqualTo(executor.trueInstance));
		Assert.That(executor.Execute(GetBinaryOperator(BinaryOperator.Smaller), N(2), [N(3)]),
			Is.EqualTo(executor.trueInstance));
		Assert.That(executor.Execute(GetBinaryOperator(BinaryOperator.GreaterOrEqual), N(5), [N(3)]),
			Is.EqualTo(executor.trueInstance));
		Assert.That(executor.Execute(GetBinaryOperator(BinaryOperator.GreaterOrEqual), N(5), [N(5)]),
			Is.EqualTo(executor.trueInstance));
		Assert.That(executor.Execute(GetBinaryOperator(BinaryOperator.SmallerOrEqual), N(2), [N(3)]),
			Is.EqualTo(executor.trueInstance));
		Assert.That(executor.Execute(GetBinaryOperator(BinaryOperator.SmallerOrEqual), N(3), [N(3)]),
			Is.EqualTo(executor.trueInstance));
		Assert.That(executor.Execute(GetBinaryOperator(BinaryOperator.Is), N(3), [N(3)]),
			Is.EqualTo(executor.trueInstance));
		Assert.That(executor.Execute(GetBinaryOperator(BinaryOperator.Is), N(3), [N(4)]),
			Is.EqualTo(executor.falseInstance));
	}

	[Test]
	public void EvaluateAllLogicalOperators()
	{
		var boolean = TestPackage.Instance.GetType(Type.Boolean);

		Method GetBinaryOperator(string op) =>
			boolean.Methods.Single(m =>
				m.Name == op && m.ReturnType.IsBoolean && m.Parameters is [{ Type.IsBoolean: true }]);

		var not = boolean.Methods.Single(m =>
			m.Name == UnaryOperator.Not && m.ReturnType.IsBoolean && m.Parameters.Count == 0);
		var and = GetBinaryOperator(BinaryOperator.And);
		AssertBooleanOperation(and, true, true, true);
		AssertBooleanOperation(and, true, false, false);
		var or = GetBinaryOperator(BinaryOperator.Or);
		AssertBooleanOperation(or, true, false, false);
		AssertBooleanOperation(or, false, false, true);
		var xor = GetBinaryOperator(BinaryOperator.Xor);
		AssertBooleanOperation(xor, true, false, true);
		AssertBooleanOperation(xor, true, true, false);
		Assert.That(executor.Execute(not, executor.falseInstance, []), Is.EqualTo(executor.trueInstance));
	}

	private void AssertBooleanOperation(Method method, bool first, bool second, bool result) =>
		Assert.That(executor.Execute(method, executor.ToBoolean(first), [executor.ToBoolean(second)]),
			Is.EqualTo(executor.ToBoolean(result)));

	[Test]
	public void EvaluateMemberCallFromStaticConstant()
	{
		using var t = CreateType(nameof(EvaluateMemberCallFromStaticConstant), "mutable last Number",
			"GetTab Character", "\tCharacter.Tab");
		var result = executor.Execute(t.Methods.Single(m => m.Name == "GetTab"));
		Assert.That(result.IsPrimitiveType(executor.characterType), Is.True, result.ToString());
		Assert.That(result.Number, Is.EqualTo(7));
	}

	[Test]
	public void EvaluateBooleanComparisons()
	{
		using var t = CreateType(nameof(EvaluateBooleanComparisons), "mutable last Boolean",
			"IfDifferent Boolean", "\tlast is false");
		Assert.That(
			executor.Execute(t.Methods.Single(m => m.Name == "IfDifferent"),
				new ValueInstance(t,
					new Dictionary<string, ValueInstance> { { "last", executor.falseInstance } }), []),
			Is.EqualTo(executor.trueInstance));
	}

	[Test]
	public void EvaluateRangeEquality()
	{
		using var t = CreateType(nameof(EvaluateRangeEquality), "has number", "Compare Boolean",
			"\tRange(0, 5) is Range(0, 5)");
		Assert.That(executor.Execute(t.Methods.Single(m => m.Name == "Compare"), executor.noneInstance,
			[]), Is.EqualTo(executor.trueInstance));
	}

	[Test]
	public void MultilineMethodRequiresTests()
	{
		using var t = CreateType(nameof(MultilineMethodRequiresTests), "has number", "GetText Text",
			"\tif number is 0", "\t\treturn \"\"", "\tnumber to Text");
		var instance = new ValueInstance(t,
			new Dictionary<string, ValueInstance>
			{
				{ "number", new ValueInstance(executor.numberType, 5.0) }
			});
		Assert.That(executor.Execute(t.Methods.Single(m => m.Name == "GetText"), instance, []).Text,
			Is.EqualTo("5"));
	}

	[Test]
	public void MethodWithoutTestsThrowsMethodRequiresTestDuringValidation()
	{
		using var t = CreateType("NoTestsNeedValidation", "has number", "Compute Number", "\tif true",
			"\t\treturn 1", "\t2");
		var method = t.Methods.Single(m => m.Name == "Compute");
		var validatingExecutor = new Executor(TestPackage.Instance);
		var ex = Assert.Throws<ExecutionFailed>(() => validatingExecutor.Execute(method));
		Assert.That(ex!.InnerException, Is.InstanceOf<Executor.MethodRequiresTest>());
	}

	[Test]
	public void MethodRequiresTestWhenParsingFailsDuringValidation()
	{
		using var t = CreateType(nameof(MethodRequiresTestWhenParsingFailsDuringValidation),
			"has number", "Compute Number", "\tunknown(1)");
		var method = t.Methods.Single(m => m.Name == "Compute");
		var validatingExecutor = new Executor(TestPackage.Instance);
		Assert.That(() => validatingExecutor.Execute(method),
			Throws.InstanceOf<Executor.MethodRequiresTest>());
	}

	[Test]
	public void InvalidTypeForFromConstructor()
	{
		using var t = CreateType(nameof(InvalidTypeForFromConstructor), "has flag Boolean",
			"from(flag Boolean, other Boolean)", "\tvalue");
		var method = t.Methods.Single(m => m.Name == Method.From);
		var number = new ValueInstance(executor.numberType, 1.0);
		var boolean = executor.trueInstance;
		Assert.That(() => executor.Execute(method, executor.noneInstance, [number, boolean]),
			Throws.InstanceOf<Executor.InvalidTypeForArgument>());
	}

	[Test]
	public void FromConstructorConvertsSingleCharText()
	{
		using var t = CreateType(nameof(FromConstructorConvertsSingleCharText), "has number",
			"has text", "from(number Number, text Text)", "\tvalue");
		var method = t.Methods.Single(m => m.Name == Method.From);
		var numberText = new ValueInstance("A");
		var text = new ValueInstance("ok");
		var result = executor.Execute(method, executor.noneInstance, [numberText, text]);
		var members = result.TryGetValueTypeInstance()!.Members;
		Assert.That(members["number"].Number, Is.EqualTo(65));
	}

	[Test]
	public void CompareNumberToText()
	{
		using var t = CreateType(nameof(CompareNumberToText), "has number", "Compare",
			"\t\"5\" is 5");
		Assert.That(
			executor.Execute(t.Methods.Single(m => m.Name == "Compare"), executor.noneInstance, []),
			Is.EqualTo(executor.trueInstance));
	}

	[Test]
	public void CompareTextToCharacterTab()
	{
		using var t = CreateType(nameof(CompareTextToCharacterTab), "has number", "Compare Boolean",
			"\t\"7\" is Character.Tab");
		Assert.That(
			executor.Execute(t.Methods.Single(m => m.Name == "Compare"), executor.noneInstance, []),
			Is.EqualTo(executor.falseInstance));
	}

	[Test]
	public void StackOverflowCallingYourselfWithSameArguments() =>
		Assert.That(() =>
			{
				using var t = CreateType(nameof(StackOverflowCallingYourselfWithSameArguments),
					"has number", "Recursive(other Number)", "\tRecursive(other)");
			}, //ncrunch: no coverage
			Throws.InstanceOf<TypeParser.SelfRecursiveCallWithSameArgumentsDetected>());

	[Test]
	public void StackOverflowCallingYourselfWithSameInstanceMember()
	{
		using var t = CreateType(nameof(StackOverflowCallingYourselfWithSameInstanceMember),
			"has number", "Recursive(other Number)", "\tRecursive(number)");
		Assert.That(() => executor.Execute(t.Methods.Single(m => m.Name == "Recursive"),
				new ValueInstance(t, new Dictionary<string, ValueInstance>
				{
					{ "number", new ValueInstance(executor.numberType, 3.0) }
				}), [new ValueInstance(executor.numberType, 1.0)]),
			Throws.InstanceOf<Executor.StackOverflowCallingItselfWithSameInstanceAndArguments>());
	}

	[Test]
	public void CallNumberPlusOperator()
	{
		using var t = CreateType(nameof(CallNumberPlusOperator), "has number", "+(text) Number",
			"\tnumber + text.Length");
		Assert.That(executor.Execute(t.Methods.Single(m => m.Name == BinaryOperator.Plus),
			new ValueInstance(t, new Dictionary<string, ValueInstance>
			{
				{ "number", new ValueInstance(executor.numberType, 5.0) }
			}),
			[new ValueInstance("abc")]).Number, Is.EqualTo(5 + 3));
	}

	[Test]
	public void InlineTestSkipListDeclarationReferencingMember()
	{
		using var t = CreateType(nameof(InlineTestSkipListDeclarationReferencingMember),
			"has first Number", "has second Number", "GetCount Number",
			"\t(1, 2, 3).Length is 3",
			"\tlet myList = (second, 2, 3)",
			"\tmyList.Length");
		var validatingExecutor = new Executor(TestPackage.Instance);
		Assert.That(
			validatingExecutor.Execute(t.Methods.Single(m => m.Name == "GetCount"),
				executor.noneInstance, [], null, true).Number, Is.EqualTo(3));
	}

	[Test]
	public void InlineDictionaryDeclarationLength()
	{
		using var t = CreateType(nameof(InlineDictionaryDeclarationLength),
			"has number", "GetCount Number",
			"\t(1, 2, 3).Length is 3",
			"\tconstant myDict = Dictionary(Number, Number)",
			"\tmyDict.Length");
		var validatingExecutor = new Executor(TestPackage.Instance);
		Assert.That(
			validatingExecutor.Execute(t.Methods.Single(m => m.Name == "GetCount")).Number, Is.EqualTo(0));
	}

	[Test]
	public void InlineTestDictionaryDeclaration()
	{
		using var t = CreateType(nameof(InlineTestDictionaryDeclaration),
				"has number", "Run Number",
				"\tconstant myDict = Dictionary(Text, Text)",
				"\tmyDict.Length is 0",
				"\tnumber");
		var validatingExecutor = new Executor(TestPackage.Instance);
		Assert.That(
			validatingExecutor.Execute(t.Methods.Single(m => m.Name == "Run")).Number, Is.EqualTo(0));
	}

	[Test]
	public void CannotCallMethodWithWrongInstanceThrows()
	{
		using var t = CreateType(nameof(CannotCallMethodWithWrongInstanceThrows), "has number",
			"Compute Number", "\tnumber");
		var method = t.Methods.Single(m => m.Name == "Compute");
		Assert.That(() => executor.Execute(method, executor.trueInstance, []),
			Throws.InstanceOf<Executor.CannotCallMethodWithWrongInstance>());
	}

	[Test]
	public void MutableDeclarationWithMutableValueTracksStatistics()
	{
		using var t = CreateType(nameof(MutableDeclarationWithMutableValueTracksStatistics),
			"has number", "Run Number",
			"\tmutable vx = number",
			"\tmutable vy = vx",
			"\tvx + vy");
		executor.Execute(t.Methods.Single(m => m.Name == "Run"), executor.noneInstance, []);
		Assert.That(executor.Statistics.MutableDeclarationCount, Is.EqualTo(1));
		Assert.That(executor.Statistics.MutableUsageCount, Is.EqualTo(1));
	}

	[Test]
	public void IsNotOperatorReturnsTrueForDifferentValues()
	{
		using var t = CreateType(nameof(IsNotOperatorReturnsTrueForDifferentValues), "has number",
			"Check Boolean", "\t1 is not 2");
		Assert.That(
			executor.Execute(t.Methods.Single(m => m.Name == "Check"), executor.noneInstance, []),
			Is.EqualTo(executor.trueInstance));
	}

	[Test]
	public void IsNotOperatorReturnsFalseForSameValues()
	{
		using var t = CreateType(nameof(IsNotOperatorReturnsFalseForSameValues), "has number",
			"Check Boolean", "\t1 is not 1");
		Assert.That(
			executor.Execute(t.Methods.Single(m => m.Name == "Check"), executor.noneInstance, []),
			Is.EqualTo(executor.falseInstance));
	}

	[Test]
	public void IsNotErrorReturnsFalseWhenBothAreErrors()
	{
		using var t = CreateType(nameof(IsNotErrorReturnsFalseWhenBothAreErrors), "has number",
			"Check Boolean",
			"\tconstant err = Error(\"test\")",
			"\terr is not err");
		Assert.That(
			executor.Execute(t.Methods.Single(m => m.Name == "Check"), executor.noneInstance, []),
			Is.EqualTo(executor.falseInstance));
	}
}