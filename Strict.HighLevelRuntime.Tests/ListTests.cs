using Strict.Expressions;
using Strict.Language;
using Strict.Language.Tests;
using Type = Strict.Language.Type;

namespace Strict.HighLevelRuntime.Tests;

public sealed class ListTests
{
	[SetUp]
	public void CreateExecutor()
	{
		executor = new Executor(TestPackage.Instance, TestBehavior.Disabled);
		one = new ValueInstance(executor.numberType, 1);
		two = new ValueInstance(executor.numberType, 2);
	}

	private Executor executor = null!;
	private ValueInstance one;
	private ValueInstance two;

	private static Type CreateType(string name, params string[] lines) =>
		new Type(TestPackage.Instance, new TypeLines(name, lines)).ParseMembersAndMethods(
			new MethodExpressionParser());

	private ValueInstance CreateNumbers(Type t) =>
		new(t,
			new Dictionary<string, ValueInstance>(StringComparer.OrdinalIgnoreCase)
			{
				{
					"numbers", new ValueInstance(t.Members[0].Type, new List<ValueInstance> { one, two })
				}
			});

	[Test]
	public void CallListOperator()
	{
		using var t = CreateType(nameof(CallListOperator), "has numbers", "Double Numbers",
			"\tnumbers + numbers");
		Assert.That(
			executor.Execute(t.Methods.Single(m => m.Name == "Double"), CreateNumbers(t), []).List.Items,
			Is.EqualTo(new[] { one, two, one, two }));
	}

	[Test]
	public void AddNumberToList()
	{
		using var t = CreateType(nameof(AddNumberToList), "has numbers", "AddOne Numbers",
			"\tnumbers + 1");
		Assert.That(
			executor.Execute(t.Methods.Single(m => m.Name == "AddOne"), CreateNumbers(t), []).List.Items,
			Is.EqualTo(new List<ValueInstance> { one, two, one }));
	}

	[Test]
	public void RemoveNumberFromList()
	{
		using var t = CreateType(nameof(AddNumberToList), "has numbers", "RemoveOne Numbers",
			"\tnumbers - 1");
		Assert.That(
			executor.Execute(t.Methods.Single(m => m.Name == "RemoveOne"), CreateNumbers(t), []).List.Items,
			Is.EqualTo(new[] { two }));
	}

	[Test]
	public void MultiplyList()
	{
		using var t = CreateType(nameof(MultiplyList), "has numbers", "Multiply Numbers",
			"\tnumbers * 2");
		Assert.That(
			executor.Execute(t.Methods.Single(m => m.Name == "Multiply"), CreateNumbers(t), []).List.Items,
			Is.EqualTo(new[] { two, new ValueInstance(t.GetType(Type.Number), 4d) }));
	}

	[Test]
	public void DivideList()
	{
		using var t = CreateType(nameof(DivideList), "has numbers", "Divide Numbers",
			"\tnumbers / 10");
		Assert.That(
			executor.Execute(t.Methods.Single(m => m.Name == "Divide"), CreateNumbers(t), []).List.Items,
			Is.EqualTo(new[]
			{
				new ValueInstance(t.GetType(Type.Number), 0.1),
				new ValueInstance(t.GetType(Type.Number), 0.2)
			}));
	}

	[Test]
	public void DivideLists()
	{
		using var t = CreateType(nameof(DivideLists), "has numbers", "Divide Numbers",
			"\tnumbers / numbers");
		Assert.That(
			executor.Execute(t.Methods.Single(m => m.Name == "Divide"), CreateNumbers(t), []).List.Items,
			Is.EqualTo(new[]
			{
				new ValueInstance(t.GetType(Type.Number), 1.0),
				new ValueInstance(t.GetType(Type.Number), 1.0)
			}));
	}

	[TestCase("(1, 2, 3) * (1, 2)")]
	[TestCase("(1, 2, 3) * (1, 2, 3, 4)")]
	public void ListsHaveDifferentDimensionsIsNotAllowed(string input)
	{
		using var t = CreateType(nameof(ListsHaveDifferentDimensionsIsNotAllowed), "has number",
			"Run", "\t" + input);
		var error = executor.Execute(t.Methods[0], executor.noneInstance, []);
		Assert.That(error.GetTypeExceptText().Name, Is.EqualTo(Type.Error));
		Assert.That(error.TryGetValueTypeInstance()!.Members["name"].Text,
			Is.EqualTo(MethodCallEvaluator.ListsHaveDifferentDimensions));
	}

	[Test]
	public void RunListIn()
	{
		using var type = new Type(TestPackage.Instance,
				new TypeLines(nameof(RunListIn), "has number", "Run Boolean",
					"\t\"d\" is not in (\"a\", \"b\", \"c\")", "\t\"b\" is in (\"a\", \"b\", \"c\")")).
			ParseMembersAndMethods(new MethodExpressionParser());
		Assert.That(executor.Execute(type.Methods[0], executor.noneInstance, []).Boolean, Is.EqualTo(true));
	}

	[Test]
	public void RunListCount()
	{
		using var type = new Type(TestPackage.Instance,
				new TypeLines(nameof(RunListCount), "has number", "GetCount Number",
					"\t(1, 2).Count(1)")).
			ParseMembersAndMethods(new MethodExpressionParser());
		Assert.That(executor.Execute(type.Methods[0], executor.noneInstance, []).Number, Is.EqualTo(1));
	}

	[Test]
	public void RunListReverse()
	{
		using var type = new Type(TestPackage.Instance,
			new TypeLines(nameof(RunListCount), "has number", "GetReverse List(Number)",
				"\t(1, 2).Reverse")).ParseMembersAndMethods(new MethodExpressionParser());
		Assert.That(executor.Execute(type.Methods[0], executor.noneInstance, []).List.Items,
			Is.EqualTo(new List<ValueInstance>
			{
				new(type.GetType(Type.Number), 2.0),
				new(type.GetType(Type.Number), 1.0)
			}));
	}

	/// <summary>
	/// For EqualsExtensions.AreEqual it is important that both sides of List values are the same.
	/// </summary>
	[Test]
	public void ListExpressionIsBecomesListOfValueInstances()
	{
		using var type = new Type(TestPackage.Instance,
				new TypeLines(nameof(ListExpressionIsBecomesListOfValueInstances), "has number",
					"CompareLists", "\t(1, 2).Reverse is (2, 1)")).
			ParseMembersAndMethods(new MethodExpressionParser());
		executor.Execute(type.Methods[0], executor.noneInstance, []);
	}
}