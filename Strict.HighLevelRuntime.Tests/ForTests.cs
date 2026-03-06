using Strict.Expressions;
using Strict.Language;
using Strict.Language.Tests;
using Type = Strict.Language.Type;

namespace Strict.HighLevelRuntime.Tests;

public sealed class ForTests
{
	[SetUp]
	public void CreateExecutor() => executor = new Executor(TestPackage.Instance, TestBehavior.Disabled);

	private Executor executor = null!;

	private static Type CreateType(string name, params string[] lines) =>
		new Type(TestPackage.Instance, new TypeLines(name, lines)).ParseMembersAndMethods(
			new MethodExpressionParser());

	[Test]
	public void CustomVariableInForLoopIsUsed()
	{
		using var t = CreateType(nameof(CustomVariableInForLoopIsUsed), "has number", "Sum Number",
			"\tfor item in (1, 2, 3)", "\t\titem");
		var result = executor.Execute(t.Methods.Single(m => m.Name == "Sum"), executor.noneInstance, []);
		Assert.That(result.Number, Is.EqualTo(6));
	}

	[TestCase("add", 6)]
	[TestCase("subtract", -6)]
	public void SelectorIfInForUsesOperation(string operation, double expected)
	{
		using var t = CreateType(nameof(SelectorIfInForUsesOperation), "has operation Text",
			"Run Number", "\tfor (1, 2, 3)", "\t\tif operation is", "\t\t\t\"add\" then value",
			"\t\t\t\"subtract\" then 0 - value");
		var instance = new ValueInstance(t,
			new Dictionary<string, ValueInstance>(StringComparer.OrdinalIgnoreCase)
			{
				{ "operation", new ValueInstance(operation) }
			});
		var result = executor.Execute(t.Methods.Single(m => m.Name == "Run"), instance, []);
		Assert.That(result.Number, Is.EqualTo(expected));
	}

	[Test]
	public void TextReturnTypeConsolidatesNumbers()
	{
		using var t = CreateType(nameof(TextReturnTypeConsolidatesNumbers), "has number", "Join Text",
			"\tfor (\"a\", \"abc\")", "\t\tvalue.Length");
		var result = executor.Execute(t.Methods.Single(m => m.Name == "Join"), executor.noneInstance, []);
		Assert.That(result.Text, Is.EqualTo("13"));
	}

	[Test]
	public void TextReturnTypeConsolidatesTexts()
	{
		using var t = CreateType(nameof(TextReturnTypeConsolidatesTexts), "has number", "Join Text",
			"\tfor (\"hello \", \"world\")", "\t\tvalue");
		var result = executor.Execute(t.Methods.Single(m => m.Name == "Join"), executor.noneInstance, []);
		Assert.That(result.Text, Is.EqualTo("hello world"));
	}

	[Test]
	public void TextReturnTypeConsolidatesCharacters()
	{
		using var t = CreateType(nameof(TextReturnTypeConsolidatesCharacters), "has number",
			"Join(character) Text", "\tfor (character, character)", "\t\tvalue");
		var result = executor.Execute(t.Methods.Single(m => m.Name == "Join"), executor.noneInstance,
			[new ValueInstance(t.GetType(Type.Character), 'b')]);
		Assert.That(result.Text, Is.EqualTo("bb"));
	}

	[Test]
	public void TextReturnTypeThrowsWhenUnsupportedValueIsUsed()
	{
		using var t = CreateType(nameof(TextReturnTypeThrowsWhenUnsupportedValueIsUsed), "has number",
			"Join Text", "\tfor (1, 2)", "\t\tError(\"boom\")");
		Assert.That(() => executor.Execute(t.Methods.Single(m => m.Name == "Join"), executor.noneInstance, []),
			Throws.InstanceOf<NotSupportedException>());
	}

	[Test]
	public void ForLoopUsesNumberIteratorLength()
	{
		using var t = CreateType(nameof(ForLoopUsesNumberIteratorLength), "has number", "Run Number",
			"\tfor 3", "\t\t1");
		var result = executor.Execute(t.Methods.Single(m => m.Name == "Run"), executor.noneInstance, []);
		Assert.That(result.Number, Is.EqualTo(3));
	}

	[Test]
	public void ForLoopUsesFloatingNumberIteratorLength()
	{
		using var t = CreateType(nameof(ForLoopUsesFloatingNumberIteratorLength), "has number",
			"Run Number", "\tfor 2.5", "\t\t1");
		var result = executor.Execute(t.Methods.Single(m => m.Name == "Run"), executor.noneInstance, []);
		Assert.That(result.Number, Is.EqualTo(2));
	}

	[Test]
	public void ForLoopThrowsWhenIteratorLengthIsUnsupported()
	{
		const string TypeName = nameof(ForLoopThrowsWhenIteratorLengthIsUnsupported);
		using var t = CreateType(TypeName, "has numbers", $"Run(container {TypeName}) Number",
			"\tfor container", "\t\t1");
		var container = new ValueInstance(t,
			new Dictionary<string, ValueInstance>(StringComparer.OrdinalIgnoreCase)
			{
				{ "numbers", new ValueInstance(t.Members[0].Type, new List<ValueInstance>()) }
			});
		Assert.That(() => executor.Execute(t.Methods.Single(m => m.Name == "Run"), executor.noneInstance, [container]),
			Throws.InstanceOf<ValueInstance.IteratorNotSupported>());
	}

	[Test]
	public void GetElementsTextWithCompactConditionalThen()
	{
		using var t = CreateType(nameof(GetElementsTextWithCompactConditionalThen), "has number",
			"GetElementsText(elements Numbers) Text", "\tfor elements",
			"\t\t(index is 0 then \"\" else \", \") + value");
		var nums = new List<ValueInstance>
		{
			new(executor.numberType, 1.0),
			new(executor.numberType, 3.0)
		};
		var listType = executor.listType.GetGenericImplementation(executor.numberType);
		var result = executor.Execute(t.Methods.Single(m => m.Name == "GetElementsText"), executor.noneInstance,
			[new ValueInstance(listType, nums)]);
		Assert.That(result.Text, Is.EqualTo("1, 3"));
	}

	[Test]
	public void ForLoopWithAscendingRangeAndEarlyReturn()
	{
		using var t = CreateType(nameof(ForLoopWithAscendingRangeAndEarlyReturn), "has number",
			"FindFirst Number",
			"\tfor Range(1, 5)",
			"\t\tif value is 3",
			"\t\t\treturn value",
			"\t\t0");
		var result = executor.Execute(t.Methods.Single(m => m.Name == "FindFirst"),
			executor.noneInstance, []);
		Assert.That(result.Number, Is.EqualTo(3));
	}

	[Test]
	public void ForLoopWithDescendingRangeAndEarlyReturn()
	{
		using var t = CreateType(nameof(ForLoopWithDescendingRangeAndEarlyReturn), "has number",
			"FindFirst Number",
			"\tfor Range(5, 1)",
			"\t\tif value is 3",
			"\t\t\treturn value",
			"\t\t0");
		var result = executor.Execute(t.Methods.Single(m => m.Name == "FindFirst"),
			executor.noneInstance, []);
		Assert.That(result.Number, Is.EqualTo(3));
	}

	[Test]
	public void TextReturnTypeWithNoResultsReturnsEmpty()
	{
		using var t = CreateType(nameof(TextReturnTypeWithNoResultsReturnsEmpty), "has number",
			"GetText Text",
			"\tmutable sum = 0",
			"\tfor (1, 2, 3)",
			"\t\tif value > 0",
			"\t\t\tsum = sum + value");
		var result = executor.Execute(t.Methods.Single(m => m.Name == "GetText"),
			executor.noneInstance, []);
		Assert.That(result.Text, Is.EqualTo(""));
	}

	[Test]
	public void TextReturnTypeConsolidatesListValues()
	{
		using var t = CreateType(nameof(TextReturnTypeConsolidatesListValues), "has number",
			"Merge Text", "\tfor (\"a\", \"b\")", "\t\t(value, value)");
		var result = executor.Execute(t.Methods.Single(m => m.Name == "Merge"),
			executor.noneInstance, []);
		Assert.That(result.Text, Is.EqualTo("(a, a), (b, b)"));
	}

	[Test]
	public void RemoveParenthesesWithElseIfChain()
	{
		using var t = CreateType(nameof(RemoveParenthesesWithElseIfChain),
			// @formatter:off
			"has text",
			"Remove Text",
			"\tmutable parentheses = 0",
			"\tfor text",
			"\t\tif value is \"(\"",
			"\t\t\tparentheses.Increment",
			"\t\tif value is \")\"",
			"\t\t\tparentheses.Decrement",
			"\t\telse if parentheses is 0",
			"\t\t\tvalue");
		var result = executor.Execute(t.Methods.Single(m => m.Name == "Remove"),
			new ValueInstance(t, new Dictionary<string, ValueInstance>(StringComparer.OrdinalIgnoreCase)
				{ { "text", new ValueInstance("example(unwanted)example") } }), []);
		Assert.That(result.Text, Is.EqualTo("exampleexample"));
	}
}