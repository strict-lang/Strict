using Strict.Language;
using Strict.Expressions;
using Strict.Language.Tests;
using Type = Strict.Language.Type;

namespace Strict.HighLevelRuntime.Tests;

public sealed class ExecutorTests
{
	private static Type CreateType(string name, params string[] lines) =>
		new Type(TestPackage.Instance, new TypeLines(name, lines)).ParseMembersAndMethods(
			new MethodExpressionParser());

	[Test]
	public void EvaluateValueAndVariableAndParameterCalls()
	{
		using var t = CreateType("Calc",
			"mutable last Number",
			"Add(first Number, second Number) Number",
			"\tlast = first + second");
		var method = t.Methods.Single(m => m.Name == "Add");
		var exec = new Executor(TestPackage.Instance);
		var a = new ValueInstance(TestPackage.Instance.FindType(Base.Number)!, 5);
		var b = new ValueInstance(TestPackage.Instance.FindType(Base.Number)!, 7);
		var result = exec.Execute(method, null, [a, b]);
		Assert.That(result.ReturnType.Name, Is.EqualTo(Base.Number));
		Assert.That(Convert.ToDouble(result.Value), Is.EqualTo(12));
	}

	[Test]
	public void EvaluateAllArithmeticOperators()
	{
		using var t = CreateType("Ops",
			"mutable last Number",
			"Plus(first Number, second Number) Number",
			"\tfirst + second",
			"Minus(first Number, second Number) Number",
			"\tfirst - second",
			"Mul(first Number, second Number) Number",
			"\tfirst * second",
			"Div(first Number, second Number) Number",
			"\tfirst / second",
			"Mod(first Number, second Number) Number",
			"\tfirst % second");
		var exec = new Executor(TestPackage.Instance);
		static ValueInstance N(double x) => new(TestPackage.Instance.FindType(Base.Number)!, x);
		Assert.That(Convert.ToDouble(exec.Execute(t.Methods.Single(m => m.Name == "Plus"), null, [
			N(2), N(3)
		]).Value), Is.EqualTo(5));
		Assert.That(Convert.ToDouble(exec.Execute(t.Methods.Single(m => m.Name == "Minus"), null, [
			N(8), N(3)
		]).Value), Is.EqualTo(5));
		Assert.That(Convert.ToDouble(exec.Execute(t.Methods.Single(m => m.Name == "Mul"), null, [
			N(6), N(7)
		]).Value), Is.EqualTo(42));
		Assert.That(Convert.ToDouble(exec.Execute(t.Methods.Single(m => m.Name == "Div"), null, [
			N(8), N(2)
		]).Value), Is.EqualTo(4));
		Assert.That(Convert.ToDouble(exec.Execute(t.Methods.Single(m => m.Name == "Mod"), null, [
			N(8), N(3)
		]).Value), Is.EqualTo(2));
	}

	[Test]
	public void EvaluateAllComparisonOperators()
	{
		using var t = CreateType("Cmp",
			"mutable last Number",
			"Gt(first Number, second Number) Boolean",
			"\tfirst > second",
			"Lt(first Number, second Number) Boolean",
			"\tfirst < second",
			"Eq(first Number, second Number) Boolean",
			"\tfirst is second"
		);
		var exec = new Executor(TestPackage.Instance);
		var num = TestPackage.Instance.FindType(Base.Number)!;
		ValueInstance N(double x) => new(num, x);
		Assert.That(exec.Execute(t.Methods.Single(m => m.Name == "Gt"), null, [N(5), N(3)]).Value, Is.EqualTo(true));
		Assert.That(exec.Execute(t.Methods.Single(m => m.Name == "Lt"), null, [N(2), N(3)]).Value, Is.EqualTo(true));
		Assert.That(exec.Execute(t.Methods.Single(m => m.Name == "Eq"), null, [N(3), N(3)]).Value, Is.EqualTo(true));
	}

	[Test]
	public void EvaluateIfTrueThenReturn()
	{
		using var t = CreateType("Cond",
			"mutable last Number",
			"IfTrue Number",
			"\tif true",
			"\t\treturn 33",
			"\t0"
		);
		var method = t.Methods.Single(m => m.Name == "IfTrue");
		var exec = new Executor(TestPackage.Instance);
		var result = exec.Execute(method, null, []);
		Assert.That(Convert.ToDouble(result.Value), Is.EqualTo(33));
	}

	[Test]
	public void EvaluateIfFalseFallsThrough()
	{
		using var t = CreateType("Cond",
			"mutable last Number",
			"IfFalse Number",
			"\tif false",
			"\t\treturn 99",
			"\t42"
		);
		var method = t.Methods.Single(m => m.Name == "IfFalse");
		var exec = new Executor(TestPackage.Instance);
		var result = exec.Execute(method, null, []);
		Assert.That(Convert.ToDouble(result.Value), Is.EqualTo(42));
	}

	[Test]
	public void EvaluateMemberCallFromStaticConstant()
	{
		using var t = CreateType("UseTab",
			"mutable last Number",
			"GetTab Character",
			"\tCharacter.Tab"
		);
		var method = t.Methods.Single(m => m.Name == "GetTab");
		var exec = new Executor(TestPackage.Instance);
		var result = exec.Execute(method, null, []);
		Assert.That(result.ReturnType.Name, Is.EqualTo(Base.Character));
		// Ensure value came from initial value of the member (Character(7))
		Assert.That(result.Value, Is.Not.Null);
	}
}