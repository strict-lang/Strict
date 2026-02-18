using Strict.Expressions;
using Strict.Language;
using Strict.Language.Tests;
using Type = Strict.Language.Type;

namespace Strict.HighLevelRuntime.Tests;

public sealed class ExecutorTests
{
	[SetUp]
	public void CreateExecutor() => executor = new Executor(TestBehavior.Disabled);

	private Executor executor = null!;

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
		Assert.That(
			executor.Execute(t.Methods.Single(m => m.Name == "Gte"), null, [N(5), N(3)]).Value,
			Is.EqualTo(true));
		Assert.That(
			executor.Execute(t.Methods.Single(m => m.Name == "Gte"), null, [N(5), N(5)]).Value,
			Is.EqualTo(true));
		Assert.That(
			executor.Execute(t.Methods.Single(m => m.Name == "Lte"), null, [N(2), N(3)]).Value,
			Is.EqualTo(true));
		Assert.That(
			executor.Execute(t.Methods.Single(m => m.Name == "Lte"), null, [N(3), N(3)]).Value,
			Is.EqualTo(true));
		Assert.That(executor.Execute(t.Methods.Single(m => m.Name == "Eq"), null, [N(3), N(3)]).Value,
			Is.EqualTo(true));
		Assert.That(
			executor.Execute(t.Methods.Single(m => m.Name == "Neq"), null, [N(3), N(4)]).Value,
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
		Assert.That(
			executor.Execute(t.Methods.Single(m => m.Name == "And"), null, [B(true), B(true)]).Value,
			Is.EqualTo(true));
		Assert.That(
			executor.Execute(t.Methods.Single(m => m.Name == "And"), null, [B(true), B(false)]).Value,
			Is.EqualTo(false));
		Assert.That(
			executor.Execute(t.Methods.Single(m => m.Name == "Or"), null, [B(true), B(false)]).Value,
			Is.EqualTo(true));
		Assert.That(
			executor.Execute(t.Methods.Single(m => m.Name == "Or"), null, [B(false), B(false)]).Value,
			Is.EqualTo(false));
		Assert.That(
			executor.Execute(t.Methods.Single(m => m.Name == "Xor"), null, [B(true), B(false)]).Value,
			Is.EqualTo(true));
		Assert.That(
			executor.Execute(t.Methods.Single(m => m.Name == "Xor"), null, [B(true), B(true)]).Value,
			Is.EqualTo(false));
		Assert.That(executor.Execute(t.Methods.Single(m => m.Name == "Not"), null, [B(false)]).Value,
			Is.EqualTo(true));
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
	public void MultilineMethodRequiresTests()
	{
		using var t = CreateType(nameof(MultilineMethodRequiresTests), "has number", "GetText Text",
			"\tif number is 0", "\t\treturn \"\"", "\tnumber to Text");
		var instance = new ValueInstance(t, 5);
		Assert.That(executor.Execute(t.Methods.Single(m => m.Name == "GetText"), instance, []).Value,
			Is.EqualTo("5"));
	}

	[Test]
	public void MethodWithoutTestsThrowsMethodRequiresTestDuringValidation()
	{
		using var t = CreateType("NoTestsNeedValidation", "has number", "Compute Number", "\tif true",
			"\t\treturn 1", "\t2");
		var method = t.Methods.Single(m => m.Name == "Compute");
		var validatingExecutor = new Executor();
		var ex = Assert.Throws<ExecutionFailed>(() => validatingExecutor.Execute(method, null, []));
		Assert.That(ex!.InnerException, Is.InstanceOf<Executor.MethodRequiresTest>());
	}

	[Test]
	public void CompareNumberToText()
	{
		using var t = CreateType(nameof(CompareNumberToText), "has number", "Compare",
			"\t\"5\" is 5");
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
		using var t = CreateType(nameof(StackOverflowCallingYourselfWithSameInstanceMember),
			"has number", "Recursive(other Number)", "\tRecursive(number)");
		Assert.That(
			() => executor.Execute(t.Methods.Single(m => m.Name == "Recursive"),
				new ValueInstance(t, 3), [new ValueInstance(t.GetType(Base.Number), 1)]).Value,
			Throws.InstanceOf<Executor.StackOverflowCallingItselfWithSameInstanceAndArguments>());
	}

	[Test]
	public void CallNumberPlusOperator()
	{
		using var t = CreateType(nameof(CallNumberPlusOperator), "has number", "+(text) Number",
			"\tnumber + text.Length");
		var instance = new ValueInstance(t, 5);
		Assert.That(
			executor.Execute(t.Methods.Single(m => m.Name == BinaryOperator.Plus), instance,
				[new ValueInstance(t.GetType(Base.Text), "abc")]).Value, Is.EqualTo(8));
	}
}