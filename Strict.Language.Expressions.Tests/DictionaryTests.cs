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
		Assert.That(dictionary.Members[0].Type, Is.InstanceOf<GenericTypeImplementation>()!);
		Assert.That(dictionary.Members[0].Type.ToString(),
			Is.EqualTo("TestPackage.Dictionary(TestPackage.Number, TestPackage.Number)"));
		Assert.That(((GenericTypeImplementation)dictionary.Members[0].Type).ImplementationTypes[1],
			Is.EqualTo(type.GetType(Base.Number)));
	}

	[Test]
	public void ParseDictionaryWithMixedInputTypes()
	{
		var dictionary = new Type(type.Package,
			new TypeLines(nameof(ParseDictionaryWithMixedInputTypes),
				"has input Dictionary(Text, Boolean)",
				"AddToDictionaryAndGetLength Number",
				"\tinput.Add(\"10\", true)",
				"\tinput.Length")).ParseMembersAndMethods(new MethodExpressionParser());
		dictionary.Methods[0].GetBodyAndParseIfNeeded();
		Assert.That(dictionary.Members[0].Type.ToString(),
			Is.EqualTo("TestPackage.Dictionary(TestPackage.Text, TestPackage.Boolean)"));
	}

	[Test]
	public void AddingIncorrectInputTypesToDictionaryShouldError()
	{
		var dictionary = new Type(type.Package,
			new TypeLines(nameof(AddingIncorrectInputTypesToDictionaryShouldError),
				"has input Dictionary(Text, Boolean)", "UseDictionary",
				"\tinput.Add(4, \"10\")",
				"\tinput")).ParseMembersAndMethods(new MethodExpressionParser());
		Assert.That(() => dictionary.Methods[0].GetBodyAndParseIfNeeded(),
			Throws.InstanceOf<Type.ArgumentsDoNotMatchMethodParameters>()!.With.Message.Contains(
				"Arguments: 4 TestPackage.Number, \"10\" TestPackage.Text do not match these method(s):" +
				"\nAdd(key TestPackage.Text, mappedValue TestPackage.Boolean) Mutable(TestPackage.Dictionary)")!);
	}

	[Ignore("Make dictionary file parsing work first before using")]
	[Test]
	public void CreateDictionaryTypeInstance()
	{
		var body = new Type(type.Package,
				new TypeLines("SchoolRegister", "has log", "LogStudentsDetails",
					"\tmutable studentsRegister = Dictionary(Number, Text)",
					"\tstudentsRegister.Add(1, \"AK\"", "\tlog.Write(studentsRegister)")).
			ParseMembersAndMethods(new MethodExpressionParser()).Methods[0].GetBodyAndParseIfNeeded();
	}
}