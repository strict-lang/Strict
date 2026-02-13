using Strict.Expressions;
using Strict.Language;
using Strict.Language.Tests;
using Type = Strict.Language.Type;

namespace Strict.HighLevelRuntime.Tests;

public sealed class ExecutorTests
{
	[SetUp]
	public void CreateExecutor()
	{
		executor = new Executor(TestPackage.Instance, TestBehavior.Disabled);
		one = new ValueInstance(TestPackage.Instance.GetType(Base.Number), 1);
		two = new ValueInstance(TestPackage.Instance.GetType(Base.Number), 2);
	}

	private Executor executor = null!;
	private ValueInstance one = null!;
	private ValueInstance two = null!;

	[Test]
	public void MissingArgument()
	{
		using var t = CreateCalcType();
		var method = t.Methods.Single(m => m.Name == "Add");
		Assert.That(() => executor.Execute(method, null, []),
			Throws.TypeOf<Executor.MissingArgument>().With.Message.StartsWith("first"));
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
		var result = executor.Execute(method, null,
			[new ValueInstance(TestPackage.Instance.FindType(Base.Number)!, 5)]);
		Assert.That(Convert.ToDouble(result.Value), Is.EqualTo(6));
	}

	[Test]
	public void TooManyArguments()
	{
		using var t = CreateCalcType();
		var method = t.Methods.Single(m => m.Name == "Add");
		Assert.That(() => executor.Execute(method, null, [
			new ValueInstance(TestPackage.Instance.FindType(Base.Number)!, 1),
			new ValueInstance(TestPackage.Instance.FindType(Base.Number)!, 2),
			new ValueInstance(TestPackage.Instance.FindType(Base.Number)!, 3)
		]), Throws.InstanceOf<Executor.TooManyArguments>().With.Message.StartsWith("Number:3"));
	}

	[Test]
	public void EvaluateValueAndVariableAndParameterCalls()
	{
		using var t = CreateCalcType();
		var method = t.Methods.Single(m => m.Name == "Add");
		var first = new ValueInstance(TestPackage.Instance.FindType(Base.Number)!, 5);
		var second = new ValueInstance(TestPackage.Instance.FindType(Base.Number)!, 7);
		var result = executor.Execute(method, null, [first, second]);
		Assert.That(result.ReturnType.Name, Is.EqualTo(Base.Number));
		Assert.That(Convert.ToDouble(result.Value), Is.EqualTo(12));
	}

	[Test]
	public void EvaluateDeclaration()
	{
		using var t = CreateType(nameof(EvaluateDeclaration), "mutable last Number",
			"AddFive(number) Number", "\tconstant five = 5", "\tnumber + five");
		var method = t.Methods.Single(m => m.Name == "AddFive");
		var number = new ValueInstance(TestPackage.Instance.FindType(Base.Number)!, 5);
		var result = executor.Execute(method, null, [number]);
		Assert.That(Convert.ToDouble(result.Value), Is.EqualTo(10));
	}

	[Test]
	public void EvaluateAllArithmeticOperators()
	{
		using var t = CreateType(nameof(EvaluateAllArithmeticOperators), "mutable last Number",
			"Plus(first Number, second Number) Number", "\tfirst + second",
			"Minus(first Number, second Number) Number", "\tfirst - second",
			"Mul(first Number, second Number) Number", "\tfirst * second",
			"Div(first Number, second Number) Number", "\tfirst / second",
			"Mod(first Number, second Number) Number", "\tfirst % second",
			"Pow(first Number, second Number) Number", "\tfirst ^ second");
		static ValueInstance N(double x) => new(TestPackage.Instance.FindType(Base.Number)!, x);
		Assert.That(Convert.ToDouble(executor.Execute(t.Methods.Single(m => m.Name == "Plus"), null, [
			N(2), N(3)
		]).Value), Is.EqualTo(5));
		Assert.That(Convert.ToDouble(executor.Execute(t.Methods.Single(m => m.Name == "Minus"), null,
		[
			N(8), N(3)
		]).Value), Is.EqualTo(5));
		Assert.That(Convert.ToDouble(executor.Execute(t.Methods.Single(m => m.Name == "Mul"), null, [
			N(6), N(7)
		]).Value), Is.EqualTo(42));
		Assert.That(Convert.ToDouble(executor.Execute(t.Methods.Single(m => m.Name == "Div"), null, [
			N(8), N(2)
		]).Value), Is.EqualTo(4));
		Assert.That(Convert.ToDouble(executor.Execute(t.Methods.Single(m => m.Name == "Mod"), null, [
			N(8), N(3)
		]).Value), Is.EqualTo(2));
		Assert.That(Convert.ToDouble(executor.Execute(t.Methods.Single(m => m.Name == "Pow"), null, [
			N(2), N(3)
		]).Value), Is.EqualTo(8));
	}

	[Test]
	public void EvaluateAllComparisonOperators()
	{
		using var t = CreateType(nameof(EvaluateAllComparisonOperators), "mutable last Number",
			"Gt(first Number, second Number) Boolean", "\tfirst > second",
			"Lt(first Number, second Number) Boolean", "\tfirst < second",
			"Gte(first Number, second Number) Boolean", "\tfirst >= second",
			"Lte(first Number, second Number) Boolean", "\tfirst <= second",
			"Eq(first Number, second Number) Boolean", "\tfirst is second",
			"Neq(first Number, second Number) Boolean", "\tfirst is not second");
		var num = TestPackage.Instance.FindType(Base.Number)!;
		ValueInstance N(double x) => new(num, x);
		Assert.That(executor.Execute(t.Methods.Single(m => m.Name == "Gt"), null, [N(5), N(3)]).Value,
			Is.EqualTo(true));
		Assert.That(executor.Execute(t.Methods.Single(m => m.Name == "Lt"), null, [N(2), N(3)]).Value,
			Is.EqualTo(true));
		Assert.That(executor.Execute(t.Methods.Single(m => m.Name == "Gte"), null, [N(5), N(3)]).Value,
			Is.EqualTo(true));
		Assert.That(executor.Execute(t.Methods.Single(m => m.Name == "Gte"), null, [N(5), N(5)]).Value,
			Is.EqualTo(true));
		Assert.That(executor.Execute(t.Methods.Single(m => m.Name == "Lte"), null, [N(2), N(3)]).Value,
			Is.EqualTo(true));
		Assert.That(executor.Execute(t.Methods.Single(m => m.Name == "Lte"), null, [N(3), N(3)]).Value,
			Is.EqualTo(true));
		Assert.That(executor.Execute(t.Methods.Single(m => m.Name == "Eq"), null, [N(3), N(3)]).Value,
			Is.EqualTo(true));
		Assert.That(executor.Execute(t.Methods.Single(m => m.Name == "Neq"), null, [N(3), N(4)]).Value,
			Is.EqualTo(true));
	}

	[Test]
	public void EvaluateAllLogicalOperators()
	{
		using var t = CreateType(nameof(EvaluateAllLogicalOperators), "has unused Boolean",
			"And(first Boolean, second Boolean) Boolean", "\tfirst and second",
			"Or(first Boolean, second Boolean) Boolean", "\tfirst or second",
			"Xor(first Boolean, second Boolean) Boolean", "\tfirst xor second",
			"Not(first Boolean) Boolean", "\tnot first");
		var boolType = TestPackage.Instance.FindType(Base.Boolean)!;
		ValueInstance B(bool x) => new(boolType, x);
		Assert.That(executor.Execute(t.Methods.Single(m => m.Name == "And"), null, [B(true), B(true)]).Value,
			Is.EqualTo(true));
		Assert.That(executor.Execute(t.Methods.Single(m => m.Name == "And"), null, [B(true), B(false)]).Value,
			Is.EqualTo(false));
		Assert.That(executor.Execute(t.Methods.Single(m => m.Name == "Or"), null, [B(true), B(false)]).Value,
			Is.EqualTo(true));
		Assert.That(executor.Execute(t.Methods.Single(m => m.Name == "Or"), null, [B(false), B(false)]).Value,
			Is.EqualTo(false));
		Assert.That(executor.Execute(t.Methods.Single(m => m.Name == "Xor"), null, [B(true), B(false)]).Value,
			Is.EqualTo(true));
		Assert.That(executor.Execute(t.Methods.Single(m => m.Name == "Xor"), null, [B(true), B(true)]).Value,
			Is.EqualTo(false));
		Assert.That(executor.Execute(t.Methods.Single(m => m.Name == "Not"), null, [B(false)]).Value,
			Is.EqualTo(true));
	}

	[Test]
	public void EvaluateIfTrueThenReturn()
	{
		using var t = CreateType(nameof(EvaluateIfTrueThenReturn), "mutable last Number",
			"IfTrue Number", "\tif true", "\t\treturn 33", "\t0");
		var method = t.Methods.Single(m => m.Name == "IfTrue");
		var result = executor.Execute(method, null, []);
		Assert.That(Convert.ToDouble(result.Value), Is.EqualTo(33));
	}

	[Test]
	public void EvaluateIsInEnumerableRange()
	{
		using var t = CreateType(nameof(EvaluateIsInEnumerableRange), "has number",
			"IsInRange(range Range) Boolean", "\tnumber is in range");
		var rangeType = TestPackage.Instance.FindType(Base.Range)!;
		var rangeInstance = new ValueInstance(rangeType, new Dictionary<string, object?>
		{
			{ "Start", 1.0 },
			{ "ExclusiveEnd", 10.0 }
		});
		var result = executor.Execute(t.Methods.Single(m => m.Name == "IsInRange"),
			new ValueInstance(t, 7.0), [rangeInstance]);
		Assert.That(result.Value, Is.EqualTo(true));
		result = executor.Execute(t.Methods.Single(m => m.Name == "IsInRange"),
			new ValueInstance(t, 11.0), [rangeInstance]);
		Assert.That(result.Value, Is.EqualTo(false));
	}

	[Test]
	public void EvaluateIsNotInEnumerableRange()
	{
		using var t = CreateType(nameof(EvaluateIsNotInEnumerableRange), "has number",
			"IsNotInRange(range Range) Boolean", "\tnumber is not in range");
		var rangeType = TestPackage.Instance.FindType(Base.Range)!;
		var rangeInstance = new ValueInstance(rangeType, new Dictionary<string, object?>
		{
			{ "Start", 1.0 },
			{ "ExclusiveEnd", 10.0 }
		});
		var result = executor.Execute(t.Methods.Single(m => m.Name == "IsNotInRange"),
			new ValueInstance(t, 11), [rangeInstance]);
		Assert.That(result.Value, Is.EqualTo(true));
		result = executor.Execute(t.Methods.Single(m => m.Name == "IsNotInRange"),
			new ValueInstance(t, 7), [rangeInstance]);
		Assert.That(result.Value, Is.EqualTo(false));
	}

	[Test]
	public void EvaluateIfFalseFallsThrough()
	{
		using var t = CreateType(nameof(EvaluateIfFalseFallsThrough), "mutable last Number",
			"IfFalse Number", "\tif false", "\t\treturn 99", "\t42");
		var method = t.Methods.Single(m => m.Name == "IfFalse");
		var result = executor.Execute(method, null, []);
		Assert.That(Convert.ToDouble(result.Value), Is.EqualTo(42));
	}

	[Test]
	public void EvaluateMemberCallFromStaticConstant()
	{
		using var t = CreateType(nameof(EvaluateMemberCallFromStaticConstant), "mutable last Number",
			"GetTab Character", "\tCharacter.Tab");
		var method = t.Methods.Single(m => m.Name == "GetTab");
		var result = executor.Execute(method, null, []);
		Assert.That(result.ReturnType.Name, Is.EqualTo(Base.Character));
		Assert.That(result.Value, Is.EqualTo(7));
	}

	[Test]
	public void EvaluateBooleanComparisons()
	{
		using var t = CreateType(nameof(EvaluateBooleanComparisons), "mutable last Boolean",
			"IfDifferent Boolean", "\tlast is false");
		var method = t.Methods.Single(m => m.Name == "IfDifferent");
		var result = executor.Execute(method,
			new ValueInstance(t, new Dictionary<string, object?> { { "last", false } }), []);
		Assert.That(Convert.ToBoolean(result.Value), Is.EqualTo(true));
	}

	[Test]
	public void EvaluateRangeEquality()
	{
		using var t = CreateType(nameof(EvaluateRangeEquality), "has number", "Compare Boolean",
			"\tRange(0, 5) is Range(0, 5)");
		var result = executor.Execute(t.Methods.Single(m => m.Name == "Compare"), null, []);
		Assert.That(result.Value, Is.EqualTo(true));
	}

	[Test]
	public void EvaluateToTextAndNumber()
	{
		using var t = CreateType(nameof(EvaluateToTextAndNumber), "has number",
			"GetText Text", "\tnumber to Text",
			"GetNumber Number", "\tnumber to Text to Number");
		var instance = new ValueInstance(t, 5);
		Assert.That(executor.Execute(t.Methods.Single(m => m.Name == "GetText"), instance, []).Value,
			Is.EqualTo("5"));
		Assert.That(
			Convert.ToDouble(executor.
				Execute(t.Methods.Single(m => m.Name == "GetNumber"), instance, []).Value),
			Is.EqualTo(5));
	}

	[Test]
	public void ToCharacterComparison()
	{
		using var t = CreateType(nameof(ToCharacterComparison), "has number",
			"Compare", "\t5 to Character is \"5\"");
		Assert.That(executor.Execute(t.Methods.Single(m => m.Name == "Compare"), null, []).Value,
			Is.EqualTo(true));
	}

	[Test]
	public void MultilineMethodRequiresTests()
	{
		using var t = CreateType(nameof(MultilineMethodRequiresTests), "has number", "GetText Text",
			"\tif number is 0", "\t\treturn \"\"", "\tnumber to Text");
		var instance = new ValueInstance(t, 5);
		Assert.That(executor.Execute(t.Methods.Single(m => m.Name == "GetText"), instance, []).Value,
			Is.EqualTo("5"));
	}

	[Test]
	public void CompareNumberToText()
	{
		using var t = CreateType(nameof(ToCharacterComparison), "has number",
			"Compare", "\t\"5\" is 5");
		Assert.That(executor.Execute(t.Methods.Single(m => m.Name == "Compare"), null, []).Value,
			Is.EqualTo(false));
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
		using var t = CreateType(nameof(StackOverflowCallingYourselfWithSameInstanceMember), "has number",
			"Recursive(other Number)", "\tRecursive(number)");
		Assert.That(
			() => executor.Execute(t.Methods.Single(m => m.Name == "Recursive"), new ValueInstance(t, 3),
				[new ValueInstance(t.GetType(Base.Number), 1)]).Value,
			Throws.InstanceOf<Executor.StackOverflowCallingItselfWithSameInstanceAndArguments>());
	}

	[Test]
	public void CallNumberPlusOperator()
	{
		using var t = CreateType(nameof(CallNumberPlusOperator), "has number",
			"+(text) Number", "\tnumber + text.Length");
		var instance = new ValueInstance(t, 5);
		Assert.That(
			executor.Execute(t.Methods.Single(m => m.Name == BinaryOperator.Plus), instance,
				[new ValueInstance(t.GetType(Base.Text), "abc")]).Value, Is.EqualTo(8));
	}

	[Test]
	public void CallListOperator()
	{
		using var t = CreateType(nameof(CallListOperator), "has numbers",
			"Double Numbers", "\tnumbers + numbers");
		Assert.That(
			executor.Execute(t.Methods.Single(m => m.Name == "Double"), CreateNumbers(t), []).Value,
			Is.EqualTo(new[] { one, two, one, two }));
	}

	private ValueInstance CreateNumbers(Type t) =>
		new(t, new Dictionary<string, object?> { { "numbers", new[] { one, two } } });

	[Test]
	public void AddNumberToList()
	{
		using var t = CreateType(nameof(AddNumberToList), "has numbers",
			"AddOne Numbers", "\tnumbers + 1");
		Assert.That(
			executor.Execute(t.Methods.Single(m => m.Name == "AddOne"), CreateNumbers(t), []).Value,
			Is.EqualTo(new List<object?> { one, two, one }));
	}

	[Test]
	public void RemoveNumberFromList()
	{
		using var t = CreateType(nameof(AddNumberToList), "has numbers",
			"RemoveOne Numbers", "\tnumbers - 1");
		Assert.That(
			executor.Execute(t.Methods.Single(m => m.Name == "RemoveOne"), CreateNumbers(t), []).Value,
			Is.EqualTo(new[] { two }));
	}

	[Test]
	public void MultiplyList()
	{
		using var t = CreateType(nameof(MultiplyList), "has numbers",
			"Multiply Numbers", "\tnumbers * 2");
		Assert.That(
			executor.Execute(t.Methods.Single(m => m.Name == "Multiply"), CreateNumbers(t), []).Value,
			Is.EqualTo(new[] { two, new ValueInstance(t.GetType(Base.Number), 4) }));
	}

	[Test]
	public void DivideList()
	{
		using var t = CreateType(nameof(DivideList), "has numbers",
			"Divide Numbers", "\tnumbers / 10");
		Assert.That(
			executor.Execute(t.Methods.Single(m => m.Name == "Divide"), CreateNumbers(t), []).Value,
			Is.EqualTo(new[]
			{
				new ValueInstance(t.GetType(Base.Number), 0.1),
				new ValueInstance(t.GetType(Base.Number), 0.2)
			}));
	}

	[TestCase("(1, 2, 3) * (1, 2)")]
	[TestCase("(1, 2, 3) * (1, 2, 3, 4)")]
	public void ListsHaveDifferentDimensionsIsNotAllowed(string input)
	{
		using var t = CreateType(nameof(ListsHaveDifferentDimensionsIsNotAllowed), "has number",
			"Run", "\t" + input);
		var error = executor.Execute(t.Methods[0], null, []);
		Assert.That(error.ReturnType.Name, Is.EqualTo(Base.Error));
		Assert.That(error.FindInnerValue(Base.Name),
			Is.EqualTo(Executor.ListsHaveDifferentDimensions));
	}

	[Test]
	public void RunListIn()
	{
		using var type = new Type(TestPackage.Instance,
				new TypeLines(nameof(RunListIn), "has number",
					"Run Boolean",
					"\t\"d\" is not in (\"a\", \"b\", \"c\")",
					"\t\"b\" is in (\"a\", \"b\", \"c\")")).
			ParseMembersAndMethods(new MethodExpressionParser());
		Assert.That(executor.Execute(type.Methods[0], null, []).Value, Is.EqualTo(true));
	}

	[Test]
	public void RunListCount()
	{
		using var type = new Type(TestPackage.Instance,
				new TypeLines(nameof(RunListCount), "has number",
					"GetCount Number",
					"\t(1, 2).Count(1)")).
			ParseMembersAndMethods(new MethodExpressionParser());
		Assert.That(executor.Execute(type.Methods[0], null, []).Value, Is.EqualTo(1));
	}

	[Test]
	public void RunListReverse()
	{
		using var type = new Type(TestPackage.Instance,
				new TypeLines(nameof(RunListCount), "has number",
					"GetReverse List(Number)",
					"\t(1, 2).Reverse")).
			ParseMembersAndMethods(new MethodExpressionParser());
		Assert.That(executor.Execute(type.Methods[0], null, []).Value,
			Is.EqualTo(new[] { 2.0, 1.0 }));
	}
}