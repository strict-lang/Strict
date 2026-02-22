using Strict.Expressions;
using Strict.Language;
using Strict.Language.Tests;
using Type = Strict.Language.Type;

namespace Strict.HighLevelRuntime.Tests;

public sealed class ForTests
{
	[SetUp]
	public void CreateExecutor() => executor = new Executor(TestBehavior.Disabled);

	private Executor executor = null!;

	private static Type CreateType(string name, params string[] lines) =>
		new Type(TestPackage.Instance, new TypeLines(name, lines)).ParseMembersAndMethods(
			new MethodExpressionParser());

	[Test]
	public void CustomVariableInForLoopIsUsed()
	{
		using var t = CreateType(nameof(CustomVariableInForLoopIsUsed), "has number", "Sum Number",
			"\tfor item in (1, 2, 3)", "\t\titem");
		var result = executor.Execute(t.Methods.Single(m => m.Name == "Sum"), null, []);
		Assert.That(Convert.ToDouble(result.Value), Is.EqualTo(6));
	}

	[Test]
	public void TextReturnTypeConsolidatesNumbers()
	{
		using var t = CreateType(nameof(TextReturnTypeConsolidatesNumbers), "has number",
			"Join Text", "\tfor (\"a\", \"abc\")", "\t\tvalue.Length");
		var result = executor.Execute(t.Methods.Single(m => m.Name == "Join"), null, []);
		Assert.That(result.Value, Is.EqualTo("13"));
	}

	[Test]
	public void TextReturnTypeConsolidatesTexts()
	{
		using var t = CreateType(nameof(TextReturnTypeConsolidatesTexts), "has number",
			"Join Text", "\tfor (\"hello \", \"world\")", "\t\tvalue");
		var result = executor.Execute(t.Methods.Single(m => m.Name == "Join"), null, []);
		Assert.That(result.Value, Is.EqualTo("hello world"));
	}

	[Test]
	public void TextReturnTypeConsolidatesCharacters()
	{
		using var t = CreateType(nameof(TextReturnTypeConsolidatesCharacters), "has number",
			"Join(character) Text", "\tfor (character, character)", "\t\tvalue");
		var result = executor.Execute(t.Methods.Single(m => m.Name == "Join"), null,
			[new ValueInstance(t.GetType(Base.Character), 'b')]);
		Assert.That(result.Value, Is.EqualTo("bb"));
	}

	[Test]
	public void TextReturnTypeThrowsWhenUnsupportedValueIsUsed()
	{
		using var t = CreateType(nameof(TextReturnTypeThrowsWhenUnsupportedValueIsUsed), "has number",
			"Join Text", "\tfor (1, 2)", "\t\tError(\"boom\")");
		Assert.That(() => executor.Execute(t.Methods.Single(m => m.Name == "Join"), null, []),
			Throws.InstanceOf<NotSupportedException>());
	}

	[Test]
	public void ForLoopUsesNumberIteratorLength()
	{
		using var t = CreateType(nameof(ForLoopUsesNumberIteratorLength), "has number",
			"Run Number", "\tfor 3", "\t\t1");
		var result = executor.Execute(t.Methods.Single(m => m.Name == "Run"), null, []);
		Assert.That(Convert.ToDouble(result.Value), Is.EqualTo(3));
	}

	[Test]
	public void ForLoopUsesFloatingNumberIteratorLength()
	{
		using var t = CreateType(nameof(ForLoopUsesFloatingNumberIteratorLength), "has number",
			"Run Number", "\tfor 2.5", "\t\t1");
		var result = executor.Execute(t.Methods.Single(m => m.Name == "Run"), null, []);
		Assert.That(Convert.ToDouble(result.Value), Is.EqualTo(2));
	}

	[Test]
	public void ForLoopThrowsWhenIteratorLengthIsUnsupported()
	{
		const string TypeName = nameof(ForLoopThrowsWhenIteratorLengthIsUnsupported);
		using var t = CreateType(TypeName, "has numbers", $"Run(container {TypeName}) Number",
			"\tfor container", "\t\t1");
		var container = new ValueInstance(t,
			new Dictionary<string, object?> { { "numbers", new List<ValueInstance>() } });
		Assert.That(() => executor.Execute(t.Methods.Single(m => m.Name == "Run"), null, [container]),
			Throws.InstanceOf<ValueInstance.IteratorNotSupported>());
	}

	[Test]
	public void GetElementsTextWithCompactConditionalThen()
	{
		using var t = CreateType(nameof(GetElementsTextWithCompactConditionalThen), "has number",
			"GetElementsText(elements Numbers) Text",
			"\tfor elements",
			"\t\t(index is 0 then \"\" else \", \") + value");
		var numberType = TestPackage.Instance.FindType(Base.Number)!;
		var result = executor.Execute(t.Methods.Single(m => m.Name == "GetElementsText"), null,
			[new ValueInstance(TestPackage.Instance.FindType(Base.List)!.GetGenericImplementation(numberType),
				new List<ValueInstance>
				{
					new(numberType, 1.0),
					new(numberType, 3.0)
				})]);
		Assert.That(result.Value, Is.EqualTo("1, 3"));
	}

	[Test]
	public void RemoveParenthesesWithElseIfChain()
	{
		using var t = CreateType(nameof(RemoveParenthesesWithElseIfChain), "has text",
			"Remove Text",
			"\tmutable parentheses = 0",
			"\tfor text",
			"\t\tif value is \"(\"",
			"\t\t\tparentheses = parentheses + 1",
			"\t\telse if value is \")\"",
			"\t\t\tparentheses = parentheses - 1",
			"\t\telse if parentheses is 0",
			"\t\t\tvalue");
		var textType = TestPackage.Instance.FindType(Base.Text)!;
		var result = executor.Execute(t.Methods.Single(m => m.Name == "Remove"),
			new ValueInstance(t, new Dictionary<string, object?> { { "text", "example(unwanted)example" } }),
			[]);
		Assert.That(result.Value, Is.EqualTo("exampleexample"));
	}
}
