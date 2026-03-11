using Strict.Expressions;
using Strict.Language;
using Strict.Language.Tests;
using Type = Strict.Language.Type;

namespace Strict.HighLevelRuntime.Tests;

public sealed class DictionaryTests
{
	[SetUp]
	public void CreateExecutor() =>
		interpreter = new Interpreter(TestPackage.Instance, TestBehavior.Disabled);

	private Interpreter interpreter = null!;

	private static Type CreateType(string name, params string[] lines) =>
		new Type(TestPackage.Instance, new TypeLines(name, lines)).ParseMembersAndMethods(
			new MethodExpressionParser());

	[Test]
	public void DictionaryExpressionEvaluatesToEmptyDictionary()
	{
		using var t = CreateType(nameof(DictionaryExpressionEvaluatesToEmptyDictionary), "has number",
			"Run Dictionary(Number, Number)", "\tDictionary(Number, Number)");
		var method = t.Methods.Single(m => m.Name == "Run");
		var result = interpreter.Execute(method);
		Assert.That(result.IsDictionary, Is.True);
		Assert.That(result.GetDictionaryItems().Count, Is.EqualTo(0));
	}

	[Test]
	public void DictionaryTypeExpressionHasLengthZero()
	{
		using var t = CreateType(nameof(DictionaryTypeExpressionHasLengthZero), "has number",
			"Run Number", "\tDictionary(Number, Number).Length");
		var method = t.Methods.Single(m => m.Name == "Run");
		var result = interpreter.Execute(method);
		Assert.That(result.Number, Is.EqualTo(0));
	}

	[Test]
	public void DictionaryAddReturnsDictionaryInstance()
	{
		using var t = CreateType(nameof(DictionaryAddReturnsDictionaryInstance), "has number",
			"Run Dictionary(Number, Number)", "\tDictionary((2, 4)).Add(4, 8)");
		var method = t.Methods.Single(m => m.Name == Method.Run);
		var result = interpreter.Execute(method);
		Assert.That(result.GetType().Name, Is.EqualTo("Dictionary(Number, Number)"));
		var values = result.GetDictionaryItems();
		Assert.That(values.Count, Is.EqualTo(2));
		Assert.That(values.Keys.Select(k => k.Number), Does.Contain(2).And.Contain(4));
	}

	[Test]
	public void DictionaryValuesAreStoredInARealDictionary()
	{
		using var t = CreateType(nameof(DictionaryValuesAreStoredInARealDictionary), "has number",
			"Run Dictionary(Number, Number)", "\tDictionary((1, 2)).Add(3, 4)");
		var method = t.Methods.Single(m => m.Name == "Run");
		var result = interpreter.Execute(method);
		var values = result.GetDictionaryItems();
		Assert.That(values.Keys.Select(k => k.Number), Does.Contain(1).And.Contain(3));
		Assert.That(values[values.Keys.First(key => key.Number == 1)].Number, Is.EqualTo(2));
		Assert.That(values[values.Keys.First(key => key.Number == 3)].Number, Is.EqualTo(4));
	}

	[Test]
	public void DictionaryAddUsesKeysAndValuesPairs()
	{
		using var t = CreateType(nameof(DictionaryAddUsesKeysAndValuesPairs), "has number",
			"Run Dictionary(Number, Number)", "\tDictionary((1, 2)).Add(3, 4).Add(5, 6)");
		var method = t.Methods.Single(m => m.Name == "Run");
		var result = interpreter.Execute(method);
		Assert.That(result.GetDictionaryItems().Count, Is.EqualTo(3));
	}
}