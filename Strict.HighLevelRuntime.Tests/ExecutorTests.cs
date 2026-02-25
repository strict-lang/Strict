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

	[Test]
	public void FromConstructorWithExistingInstanceThrows()
	{
		using var t = CreateType(nameof(FromConstructorWithExistingInstanceThrows), "has number",
			"from(number Number)", "\tvalue");
		var method = t.Methods.Single(m => m.Name == Method.From);
		var number = ValueInstance.Create(TestPackage.Instance.FindType(Base.Number)!, 3d);
		var instance = ValueInstance.Create(t, 1d);
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
		var result = executor.Execute(method, null,
			[ValueInstance.Create(TestPackage.Instance.FindType(Base.Number)!, 5d)]);
		Assert.That(Convert.ToDouble(result.Value), Is.EqualTo(6));
	}

	[Test]
	public void TooManyArguments()
	{
		using var t = CreateCalcType();
		var method = t.Methods.Single(m => m.Name == "Add");
		var num = TestPackage.Instance.FindType(Base.Number)!;
		Assert.That(() => executor.Execute(method, null, [
			ValueInstance.Create(num, 1d),
			ValueInstance.Create(num, 2d),
			ValueInstance.Create(num, 3d)
		]), Throws.InstanceOf<Executor.TooManyArguments>().With.Message.StartsWith("Number:3"));
	}

	[Test]
	public void ArgumentDoesNotMapToMethodParameters()
	{
		using var t = CreateType(nameof(ArgumentDoesNotMapToMethodParameters), "has number",
			"Use(number Number) Number", "\tnumber");
		var method = t.Methods.Single(m => m.Name == "Use");
		var boolean = ValueInstance.Create(TestPackage.Instance.FindType(Base.Boolean)!, true);
		Assert.That(() => executor.Execute(method, null, [boolean]),
			Throws.InstanceOf<Executor.ArgumentDoesNotMapToMethodParameters>());
	}

	[Test]
	public void EvaluateValueAndVariableAndParameterCalls()
	{
		using var t = CreateCalcType();
		var method = t.Methods.Single(m => m.Name == "Add");
		var num = TestPackage.Instance.FindType(Base.Number)!;
		var first = ValueInstance.Create(num, 5d);
		var second = ValueInstance.Create(num, 7d);
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
		var number = ValueInstance.Create(TestPackage.Instance.FindType(Base.Number)!, 5d);
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
		var num = TestPackage.Instance.FindType(Base.Number)!;
		static ValueInstance N(Type numType, double x) => ValueInstance.Create(numType, x);
		Assert.That(Convert.ToDouble(executor.Execute(t.Methods.Single(m => m.Name == "Plus"), null, [
			N(num, 2), N(num, 3)
		]).Value), Is.EqualTo(5));
		Assert.That(Convert.ToDouble(executor.Execute(t.Methods.Single(m => m.Name == "Minus"), null,
		[
			N(num, 8), N(num, 3)
		]).Value), Is.EqualTo(5));
		Assert.That(Convert.ToDouble(executor.Execute(t.Methods.Single(m => m.Name == "Mul"), null, [
			N(num, 6), N(num, 7)
		]).Value), Is.EqualTo(42));
		Assert.That(Convert.ToDouble(executor.Execute(t.Methods.Single(m => m.Name == "Div"), null, [
			N(num, 8), N(num, 2)
		]).Value), Is.EqualTo(4));
		Assert.That(Convert.ToDouble(executor.Execute(t.Methods.Single(m => m.Name == "Mod"), null, [
			N(num, 8), N(num, 3)
		]).Value), Is.EqualTo(2));
		Assert.That(Convert.ToDouble(executor.Execute(t.Methods.Single(m => m.Name == "Pow"), null, [
			N(num, 2), N(num, 3)
		]).Value), Is.EqualTo(8));
	}

	[Test]
	public void AddTwoTexts()
	{
		using var t = CreateType(nameof(AddTwoTexts), "has text",
			"Concat(text Text, other Text) Text", "\ttext + other");
		var textType = t.GetType(Base.Text);
		var result = executor.Execute(t.Methods.Single(m => m.Name == "Concat"), null,
			[ValueInstance.Create(textType, "hi "), ValueInstance.Create(textType, "there")]);
		Assert.That(result.Value, Is.EqualTo("hi there"));
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
		ValueInstance N(double x) => ValueInstance.Create(num, x);
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
		ValueInstance B(bool x) => ValueInstance.Create(boolType, x);
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
			ValueInstance.Create(t, new Dictionary<string, object?> { { "last", false } }), []);
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
		var instance = ValueInstance.Create(t, 5);
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
	public void MethodRequiresTestWhenParsingFailsDuringValidation()
	{
		using var t = CreateType(nameof(MethodRequiresTestWhenParsingFailsDuringValidation),
			"has number", "Compute Number", "\tunknown(1)");
		var method = t.Methods.Single(m => m.Name == "Compute");
		var validatingExecutor = new Executor();
		Assert.That(() => validatingExecutor.Execute(method, null, []),
			Throws.InstanceOf<Executor.MethodRequiresTest>());
	}

	[Test]
	public void InvalidTypeForFromConstructor()
	{
		using var t = CreateType(nameof(InvalidTypeForFromConstructor), "has flag Boolean",
			"from(flag Boolean, other Boolean)", "\tvalue");
		var method = t.Methods.Single(m => m.Name == Method.From);
		var number = ValueInstance.Create(TestPackage.Instance.FindType(Base.Number)!, 1);
		var boolean = ValueInstance.Create(TestPackage.Instance.FindType(Base.Boolean)!, true);
		Assert.That(() => executor.Execute(method, null, [number, boolean]),
			Throws.InstanceOf<Executor.InvalidTypeForArgument>());
	}

	[Test]
	public void FromConstructorConvertsSingleCharText()
	{
		using var t = CreateType(nameof(FromConstructorConvertsSingleCharText), "has number",
			"has text", "from(number Number, text Text)", "\tvalue");
		var method = t.Methods.Single(m => m.Name == Method.From);
		var numberText = ValueInstance.Create(TestPackage.Instance.FindType(Base.Text)!, "A");
		var text = ValueInstance.Create(TestPackage.Instance.FindType(Base.Text)!, "ok");
		var result = executor.Execute(method, null, [numberText, text]);
		var values = (IDictionary<string, object?>)result.Value!;
		Assert.That(values["number"], Is.EqualTo(65));
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
	public void CompareTextToCharacterTab()
	{
		using var t = CreateType(nameof(CompareTextToCharacterTab), "has number", "Compare Boolean",
			"\t\"7\" is Character.Tab");
		Assert.That(executor.Execute(t.Methods.Single(m => m.Name == "Compare"), null, []).Value,
			Is.EqualTo(true));
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
				ValueInstance.Create(t, 3), [ValueInstance.Create(t.GetType(Base.Number), 1)]).Value,
			Throws.InstanceOf<Executor.StackOverflowCallingItselfWithSameInstanceAndArguments>());
	}

	[Test]
	public void CallNumberPlusOperator()
	{
		using var t = CreateType(nameof(CallNumberPlusOperator), "has number", "+(text) Number",
			"\tnumber + text.Length");
		Assert.That(
			executor.Execute(t.Methods.Single(m => m.Name == BinaryOperator.Plus),
				ValueInstance.Create(t, 5),
				[ValueInstance.Create(t.GetType(Base.Text), "abc")]).Value, Is.EqualTo(5 + 3));
	}
}