using NUnit.Framework;

namespace Strict.Language.Expressions.Tests;

public sealed class DictionaryTests : TestExpressions
{
	[Test]
	public void ParseMultipleTypesInsideAListType()
	{
		var listInListType = new Type(type.Package,
			new TypeLines(nameof(ParseMultipleTypesInsideAListType),
				"has keysAndValues List((key Generic, value Generic))", "UseDictionary",
				"\tconstant result = 5")).ParseMembersAndMethods(new MethodExpressionParser());
		Assert.That(listInListType.Members[0].Type.IsIterator, Is.True);
		Assert.That(listInListType.Members[0].Type.Name, Is.EqualTo("KeyValues"));
	}

	[Test]
	public void ParseMultipleTypesInsideAListTypeAsParameter()
	{
		var listInListType = new Type(type.Package,
			new TypeLines(nameof(ParseMultipleTypesInsideAListTypeAsParameter),
				"has keysAndValues Generic", "UseDictionary(keyValues List((firstType Generic, mappedSecondType Generic)))",
				"\tconstant result = 5")).ParseMembersAndMethods(new MethodExpressionParser());
		Assert.That(listInListType.Methods[0].Parameters[0].Type.IsIterator, Is.True);
		Assert.That(listInListType.Methods[0].Parameters[0].Type.Name,
			Is.EqualTo("FirstTypeMappedSecondTypes"));
	}

	[Test]
	public void ParseDictionaryType()
	{
		var dictionary = new Type(type.Package,
			new TypeLines(nameof(ParseDictionaryType),
				"has inputMap Dictionary(Number, Number)", "UseDictionary",
				"\tinputMap.Add(4, 6)",
				"\tinputMap")).ParseMembersAndMethods(new MethodExpressionParser());
		Assert.That(dictionary.Members[0].Type,
			Is.InstanceOf<GenericTypeImplementation>());
		Assert.That(dictionary.Members[0].Type.ToString(),
			Is.EqualTo("TestPackage.Dictionary(TestPackage.Number, TestPackage.Number)"));
		Assert.That(((GenericTypeImplementation)dictionary.Members[0].Type).ImplementationTypes[1],
			Is.EqualTo(type.GetType(Base.Number)));
	}

	[Ignore("Make dictionary file parsing work first before using")]
	[Test]
	public void UseDictionaryType()
	{
		var body = new Type(type.Package,
				new TypeLines(nameof(UseDictionaryType), "has log", "UseDictionary",
					"\tconstant result = Dictionary(Number, Number)")).
			ParseMembersAndMethods(new MethodExpressionParser()).Methods[0].GetBodyAndParseIfNeeded();
	}
}