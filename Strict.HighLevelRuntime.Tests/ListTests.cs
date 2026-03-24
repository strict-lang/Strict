using Strict.Expressions;
using Strict.Language;
using Strict.Language.Tests;
using System.Reflection;
using Type = Strict.Language.Type;

namespace Strict.HighLevelRuntime.Tests;

public sealed class ListTests
{
	[SetUp]
	public void CreateExecutor()
	{
		interpreter = new Interpreter(TestPackage.Instance, TestBehavior.Disabled);
		one = new ValueInstance(interpreter.numberType, 1);
		two = new ValueInstance(interpreter.numberType, 2);
	}

	private Interpreter interpreter = null!;
	private ValueInstance one;
	private ValueInstance two;

	private static Type CreateType(string name, params string[] lines) =>
		new Type(TestPackage.Instance, new TypeLines(name, lines)).ParseMembersAndMethods(
			new MethodExpressionParser());

	private ValueInstance CreateNumbers(Type t) =>
		new(t, [new ValueInstance(t.Members[0].Type, [one, two])]);

	[Test]
	public void CallListOperator()
	{
		using var t = CreateType(nameof(CallListOperator), "has numbers", "Double Numbers",
			"\tnumbers + numbers");
		Assert.That(
			interpreter.Execute(t.Methods.Single(m => m.Name == "Double"), CreateNumbers(t), []).List.
				Items, Is.EqualTo(new[] { one, two, one, two }));
	}

	[Test]
	public void AddNumberToList()
	{
		using var t = CreateType(nameof(AddNumberToList), "has numbers", "AddOne Numbers",
			"\tnumbers + 1");
		Assert.That(
			interpreter.Execute(t.Methods.Single(m => m.Name == "AddOne"), CreateNumbers(t), []).List.
				Items, Is.EqualTo(new List<ValueInstance> { one, two, one }));
	}

	[Test]
	public void RemoveNumberFromList()
	{
		using var t = CreateType(nameof(AddNumberToList), "has numbers", "RemoveOne Numbers",
			"\tnumbers - 1");
		Assert.That(
			interpreter.Execute(t.Methods.Single(m => m.Name == "RemoveOne"), CreateNumbers(t), []).List.
				Items, Is.EqualTo(new[] { two }));
	}

	[Test]
	public void MultiplyList()
	{
		using var t = CreateType(nameof(MultiplyList), "has numbers", "Multiply Numbers",
			"\tnumbers * 2");
		Assert.That(
			interpreter.Execute(t.Methods.Single(m => m.Name == "Multiply"), CreateNumbers(t), []).List.
				Items, Is.EqualTo(new[] { two, new ValueInstance(t.GetType(Type.Number), 4) }));
	}

	[Test]
	public void DivideList()
	{
		using var t = CreateType(nameof(DivideList), "has numbers", "Divide Numbers",
			"\tnumbers / 10");
		Assert.That(
			interpreter.Execute(t.Methods.Single(m => m.Name == "Divide"), CreateNumbers(t), []).List.Items,
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
			interpreter.Execute(t.Methods.Single(m => m.Name == "Divide"), CreateNumbers(t), []).List.Items,
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
		var error = interpreter.Execute(t.Methods[0], interpreter.noneInstance, []);
		Assert.That(error.GetType().Name, Is.EqualTo(Type.Error));
		Assert.That(error.TryGetValueTypeInstance()!["name"].Text,
			Is.EqualTo(MethodCallEvaluator.ListsHaveDifferentDimensions));
	}

	[Test]
	public void AddTextListToNumberListReturnsDowncastError()
	{
		using var type = CreateType(nameof(AddTextListToNumberListReturnsDowncastError), "has number",
			"Run Error", "\t(1) + (\"Hi\")");
		Assert.That(interpreter.Execute(type.Methods[0], interpreter.noneInstance, []).ToString(),
			Contains.Substring("Cannot downcast Text to Number for list: Text: \\\"Hi\\\""));
	}

	[Test]
	public void AddTextToNumberListIsError()
	{
		using var type = CreateType(nameof(AddTextListToNumberListReturnsDowncastError), "has number",
			"Run", "\t(1) + (\"Hi\") is Error(\"Cannot downcast Text to Number for list: Text: \\\"Hi\\\"\")");
		interpreter.Execute(type.Methods[0], interpreter.noneInstance, []);
	}

	[Test]
	public void RunListIn()
	{
		using var type = new Type(TestPackage.Instance,
				new TypeLines(nameof(RunListIn), "has number", "Run Boolean",
					"\t\"d\" is not in (\"a\", \"b\", \"c\")", "\t\"b\" is in (\"a\", \"b\", \"c\")")).
			ParseMembersAndMethods(new MethodExpressionParser());
		Assert.That(interpreter.Execute(type.Methods[0], interpreter.noneInstance, []).Boolean, Is.EqualTo(true));
	}

	[Test]
	public void RunListCount()
	{
		using var type = new Type(TestPackage.Instance,
				new TypeLines(nameof(RunListCount), "has number", "GetCount Number",
					"\t(1, 2).Count(1)")).
			ParseMembersAndMethods(new MethodExpressionParser());
		Assert.That(interpreter.Execute(type.Methods[0], interpreter.noneInstance, []).Number, Is.EqualTo(1));
	}

	[Test]
	public void RunListReverse()
	{
		using var type = new Type(TestPackage.Instance,
			new TypeLines(nameof(RunListCount), "has number", "GetReverse List(Number)",
				"\t(1, 2).Reverse")).ParseMembersAndMethods(new MethodExpressionParser());
		Assert.That(interpreter.Execute(type.Methods[0], interpreter.noneInstance, []).List.Items,
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
		interpreter.Execute(type.Methods[0], interpreter.noneInstance, []);
	}

	[Test]
	public void ConstantListExpressionCachesDataBetweenCalls()
	{
		using var type = new Type(TestPackage.Instance,
				new TypeLines("ConstListCache",
					"has number",
					"TestReverse Boolean",
					"\t(1, 2).Reverse is (2, 1)")).
			ParseMembersAndMethods(new MethodExpressionParser());
		Assert.That(interpreter.Execute(type.Methods[0], interpreter.noneInstance, []).Boolean, Is.True);
		Assert.That(interpreter.Execute(type.Methods[0], interpreter.noneInstance, []).Boolean, Is.True);
	}

	[Test]
	public void AddToMutableListReusesExistingListInstance()
	{
		var mutableList = CreateMutableNumberList(one, two);
		var result = InvokePrivateListMethod("AddToList", mutableList,
			new ValueInstance(interpreter.numberType, 3));
		Assert.That(ReferenceEquals(mutableList.List.Items, result.List.Items), Is.True);
		Assert.That(result.List.Items.Count, Is.EqualTo(3));
	}

	[Test]
	public void CombineMutableListsReusesLeftListInstance()
	{
		var leftList = CreateMutableNumberList(one, two);
		var rightList = new List<ValueInstance>
		{
			new(interpreter.numberType, 3), new(interpreter.numberType, 4)
		};
		var method = typeof(MethodCallEvaluator).GetMethod("CombineLists",
			System.Reflection.BindingFlags.NonPublic | BindingFlags.Instance)!;
		var result = (ValueInstance)(method.Invoke(new MethodCallEvaluator(null!),
			new object[] { leftList, rightList, null!, null! })!);
		Assert.That(ReferenceEquals(leftList.List.Items, result.List.Items), Is.True);
		Assert.That(result.List.Items.Count, Is.EqualTo(4));
	}

	[Test]
	public void SubtractMutableListsReusesLeftListInstance()
	{
		var leftList = CreateMutableNumberList(one, two, one);
		var rightList = new List<ValueInstance> { one, two };
		var result = InvokePrivateListMethod("SubtractLists", leftList, rightList);
		Assert.That(ReferenceEquals(leftList.List.Items, result.List.Items), Is.True);
		Assert.That(result.List.Items, Is.EqualTo(new[] { one }));
	}

	[Test]
	public void RemoveFromMutableListReusesLeftListInstance()
	{
		var leftList = CreateMutableNumberList(one, two, one);
		var result = InvokePrivateListMethod("RemoveFromList", leftList, one);
		Assert.That(ReferenceEquals(leftList.List.Items, result.List.Items), Is.True);
		Assert.That(result.List.Items, Is.EqualTo(new[] { two }));
	}

	[Test]
	public void AddToImmutableListCreatesNewListAndKeepsOriginalUnchanged()
	{
		using var type = CreateType("ImmutableListCopyType",
			"has numbers",
			"Append Numbers",
			"\tnumbers + 3");
		var typeInstance = CreateNumbers(type);
		var originalNumbers = typeInstance.TryGetValueTypeInstance()!["numbers"];
		var result = interpreter.Execute(type.Methods.Single(method => method.Name == "Append"),
			typeInstance, []);
		Assert.That(ReferenceEquals(originalNumbers.List.Items, result.List.Items), Is.False);
		Assert.That(originalNumbers.List.Items.Count, Is.EqualTo(2));
	}

	private ValueInstance CreateMutableNumberList(params ValueInstance[] items)
	{
		var numberType = interpreter.numberType;
		var listType = numberType.GetListImplementationType(numberType);
		var mutableListType = numberType.GetType(Type.Mutable).GetGenericImplementation(listType);
		return new ValueInstance(mutableListType, items);
	}

	private static ValueInstance InvokePrivateListMethod(string methodName, params object[] args)
	{
		var method = typeof(MethodCallEvaluator).GetMethod(methodName,
				System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Static) ??
			throw new InvalidOperationException(methodName + " method not found");
		return (ValueInstance)(method.Invoke(null, args) ??
			throw new InvalidOperationException(methodName + " returned null"));
	}
}