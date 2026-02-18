using Strict.Expressions;
using Strict.Language;
using Strict.Language.Tests;
using Type = Strict.Language.Type;

namespace Strict.HighLevelRuntime.Tests;

public sealed class DictionaryTests
{
	[SetUp]
	public void CreateExecutor() => executor = new Executor(TestBehavior.Disabled);

	private Executor executor = null!;

	private static Type CreateType(string name, params string[] lines) =>
		new Type(TestPackage.Instance, new TypeLines(name, lines)).ParseMembersAndMethods(
			new MethodExpressionParser());

	[Test]
	public void DictionaryTypeExpressionHasLengthZero()
	{
		using var t = CreateType(nameof(DictionaryTypeExpressionHasLengthZero), "has number",
			"Run Number", "\tDictionary(Number, Number).Length");
		var method = t.Methods.Single(m => m.Name == "Run");
		var result = executor.Execute(method, null, []);
		Assert.That(Convert.ToDouble(result.Value), Is.EqualTo(0));
	}

	[Test]
	public void DictionaryAddReturnsDictionaryInstance()
	{
		using var t = CreateType(nameof(DictionaryAddReturnsDictionaryInstance), "has number",
			"Run Dictionary(Number, Number)", "\tDictionary((2, 4)).Add(4, 8)");
		var method = t.Methods.Single(m => m.Name == "Run");
		var result = executor.Execute(method, null, []);
		Assert.That(result.ReturnType.Name, Is.EqualTo("Dictionary(Number, Number)"));
		var values = (System.Collections.IDictionary)result.Value!;
		Assert.That(values.Count, Is.EqualTo(2));
		var keys = values.Keys.Cast<object?>().ToList();
		Assert.That(keys.Select(EqualsExtensions.NumberToDouble), Does.Contain(2));
		Assert.That(keys.Select(EqualsExtensions.NumberToDouble), Does.Contain(4));
	}

	[Test]
	public void DictionaryValuesAreStoredInARealDictionary()
	{
		using var t = CreateType(nameof(DictionaryValuesAreStoredInARealDictionary), "has number",
			"Run Dictionary(Number, Number)", "\tDictionary((1, 2)).Add(3, 4)");
		var method = t.Methods.Single(m => m.Name == "Run");
		var result = executor.Execute(method, null, []);
		var values = (System.Collections.IDictionary)result.Value!;
		var keys = values.Keys.Cast<object?>().ToList();
		Assert.That(keys.Select(EqualsExtensions.NumberToDouble), Does.Contain(1));
		Assert.That(keys.Select(EqualsExtensions.NumberToDouble), Does.Contain(3));
		var key1 = keys.First(key => EqualsExtensions.NumberToDouble(key) == 1);
		var key3 = keys.First(key => EqualsExtensions.NumberToDouble(key) == 3);
		Assert.That(Convert.ToDouble(values[key1!]), Is.EqualTo(2));
		Assert.That(Convert.ToDouble(values[key3!]), Is.EqualTo(4));
	}
}
