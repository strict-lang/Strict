using Strict.Expressions;
using Strict.Language;
using Strict.Language.Tests;
using Type = Strict.Language.Type;

namespace Strict.HighLevelRuntime.Tests;

public sealed class IfTests
{
	[SetUp]
	public void CreateExecutor() => executor = new Executor(TestBehavior.Disabled);

	private Executor executor = null!;

	private static Type CreateType(string name, params string[] lines) =>
		new Type(TestPackage.Instance, new TypeLines(name, lines)).ParseMembersAndMethods(
			new MethodExpressionParser());

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
	public void EvaluateIfFalseFallsThrough()
	{
		using var t = CreateType(nameof(EvaluateIfFalseFallsThrough), "mutable last Number",
			"IfFalse Number", "\tif false", "\t\treturn 99", "\t42");
		var method = t.Methods.Single(m => m.Name == "IfFalse");
		var result = executor.Execute(method, null, []);
		Assert.That(Convert.ToDouble(result.Value), Is.EqualTo(42));
	}

	[Test]
	public void EvaluateIsInEnumerableRange()
	{
		using var t = CreateType(nameof(EvaluateIsInEnumerableRange), "has number",
			"IsInRange(range Range) Boolean", "\tnumber is in range");
		var rangeType = TestPackage.Instance.FindType(Base.Range)!;
		var numberType = TestPackage.Instance.FindType(Base.Number)!;
		var rangeInstance = new ValueInstance(rangeType,
			new Dictionary<string, ValueInstance> {
				{ "Start", new ValueInstance(numberType, 1.0) },
				{ "ExclusiveEnd", new ValueInstance(numberType, 10.0) } });
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
		var numberType = TestPackage.Instance.FindType(Base.Number)!;
		var rangeInstance = new ValueInstance(rangeType,
			new Dictionary<string, ValueInstance> {
				{ "Start", new ValueInstance(numberType, 1.0) },
				{ "ExclusiveEnd", new ValueInstance(numberType, 10.0) } });
		var result = executor.Execute(t.Methods.Single(m => m.Name == "IsNotInRange"),
			new ValueInstance(t, 11), [rangeInstance]);
		Assert.That(result.Value, Is.EqualTo(true));
		result = executor.Execute(t.Methods.Single(m => m.Name == "IsNotInRange"),
			new ValueInstance(t, 7), [rangeInstance]);
		Assert.That(result.Value, Is.EqualTo(false));
	}

	[Test]
	public void ReturnTypeMustMatchMethod()
	{
		using var t = CreateType(nameof(ReturnTypeMustMatchMethod), "has unused Boolean",
			"Run(condition Boolean) Number", "\t5 is 5", "\tif condition", "\t\t6");
		var method = t.Methods.Single(m => m.Name == "Run");
		Assert.That(
			() => executor.Execute(method, null,
				[new ValueInstance(TestPackage.Instance.FindType(Base.Boolean)!, false)]),
			Throws.TypeOf<ExecutionFailed>().With.InnerException.
				TypeOf<Executor.ReturnTypeMustMatchMethod>());
	}

	[Test]
	public void SelectorIfUsesElseWhenNoCaseMatches()
	{
		using var t = CreateType(nameof(SelectorIfUsesElseWhenNoCaseMatches), "has value Number",
			"Run Boolean", "\tif value is", "\t\t2 then true", "\t\telse false");
		var instance = new ValueInstance(t, 3.0);
		var result = executor.Execute(t.Methods.Single(m => m.Name == "Run"), instance, []);
		Assert.That(result.Value, Is.EqualTo(false));
	}
}
