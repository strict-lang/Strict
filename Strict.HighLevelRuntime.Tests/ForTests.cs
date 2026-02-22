using Strict.Expressions;
using Strict.Language;
using Strict.Language.Tests;
using System.Collections.Generic;
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

	[TestCase("add", 6)]
	[TestCase("subtract", -6)]
	public void SelectorIfInForUsesOperation(string operation, double expected)
	{
    using var t = CreateType(nameof(SelectorIfInForUsesOperation), "has operation Text",
			"Run Number", "\tfor (1, 2, 3)", "\t\tif operation is",
			"\t\t\t\"add\" then value", "\t\t\t\"subtract\" then 0 - value");
		var instance = new ValueInstance(t,
			new Dictionary<string, object?> { { "operation", operation } });
		var result = executor.Execute(t.Methods.Single(m => m.Name == "Run"), instance, []);
		Assert.That(Convert.ToDouble(result.Value), Is.EqualTo(expected));
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
}
