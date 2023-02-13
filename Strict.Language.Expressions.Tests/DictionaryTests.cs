using NUnit.Framework;

namespace Strict.Language.Expressions.Tests;

public sealed class DictionaryTests : TestExpressions
{
	[Test]
	public void ParseListInsideAListType()
	{
		var listInListType = new Type(type.Package,
			new TypeLines(nameof(ParseListInsideAListType),
				"has keysAndValues List((key Generic, mappedValue Generic))", "UseDictionary",
				"\tconstant result = 5")).ParseMembersAndMethods(new MethodExpressionParser());
		Assert.That(listInListType.Members[0].Type.IsIterator, Is.True);
		Assert.That(listInListType.Members[0].Type.Name, Is.EqualTo("KeyMappedValues"));
	}

	[Ignore("Make dictionary file parsing work first before using")]
	[Test]
	public void ParseDictionary() =>
		Assert.That(ParseExpression("Dictionary(Number, Number)"),
			Is.EqualTo("Dictionary(Number, Number)"));

	[Ignore("Make dictionary file parsing work first before using")]
	[Test]
	public void UseDictionaryType() =>
		Assert.That(
			() => new Type(type.Package,
					new TypeLines(nameof(UseDictionaryType), "has log", "UseDictionary",
						"\tconstant result = Dictionary(Number, Number)")).
				ParseMembersAndMethods(new MethodExpressionParser()).Methods[0].GetBodyAndParseIfNeeded(),
			Throws.InstanceOf<Type.GenericTypesCannotBeUsedDirectlyUseImplementation>());
}