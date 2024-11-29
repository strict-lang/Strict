using NUnit.Framework;

namespace Strict.Language.Expressions.Tests;

public sealed class DictionaryTests : TestExpressions
{
	[Test]
	public void ParseMultipleTypesInsideAListType()
	{
		var listInListType = new Type(type.Package,
			new TypeLines(nameof(ParseMultipleTypesInsideAListType),
				"has keysAndValues List(key Generic, value Generic)", "UseDictionary",
				"\tconstant result = 5")).ParseMembersAndMethods(new MethodExpressionParser());
		Assert.That(listInListType.Members[0].Type.IsIterator, Is.True);
		Assert.That(listInListType.Members[0].Type.Name, Is.EqualTo("List(key TestPackage.Generic, value TestPackage.Generic)"));
	}

	[Test]
	public void ParseListWithGenericKeyAndValue() =>
		Assert.That(() => new Type(type.Package,
				new TypeLines(nameof(ParseListWithGenericKeyAndValue),
					"has keysAndValues List(key Generic, value Generic)", "Get(key Generic) Generic",
					"\tfor keysAndValues", "\t\tif value is key", "\t\t\treturn value(1)))")).
			ParseMembersAndMethods(new MethodExpressionParser()), Throws.Nothing);

	[Test]
	public void ParseMultipleTypesInsideAListTypeAsParameter()
	{
		var listInListType = new Type(type.Package,
			new TypeLines(nameof(ParseMultipleTypesInsideAListTypeAsParameter),
				"has keysAndValues Generic", "UseDictionary(keyValues List(firstType Generic, mappedSecondType Generic))",
				"\tconstant result = 5")).ParseMembersAndMethods(new MethodExpressionParser());
		Assert.That(listInListType.Methods[0].Parameters[0].Type.IsIterator, Is.True);
		Assert.That(listInListType.Methods[0].Parameters[0].Type.Name,
			Is.EqualTo("List(firstType TestPackage.Generic, mappedSecondType TestPackage.Generic)"));
	}

	[Test]
	public void ParseDictionaryType()
	{
		var dictionary = new Type(type.Package,
				new TypeLines(nameof(ParseDictionaryType), "has inputMap Dictionary(Number, Number)",
					"UseDictionary", "\tinputMap.Add(4, 6)", "\tinputMap")).
			ParseMembersAndMethods(new MethodExpressionParser());
		Assert.That(dictionary.Members[0].Type, Is.InstanceOf<GenericTypeImplementation>());
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
					"has input Dictionary(Text, Boolean)", "AddToDictionaryAndGetLength Number",
					"\tinput.Add(\"10\", true)", "\tinput.Length")).
			ParseMembersAndMethods(new MethodExpressionParser());
		dictionary.Methods[0].GetBodyAndParseIfNeeded();
		Assert.That(dictionary.Members[0].Type.ToString(),
			Is.EqualTo("TestPackage.Dictionary(TestPackage.Text, TestPackage.Boolean)"));
	}

	[Test]
	public void AddingIncorrectInputTypesToDictionaryShouldError()
	{
		var dictionary = new Type(type.Package,
			new TypeLines(nameof(AddingIncorrectInputTypesToDictionaryShouldError),
				"has input Dictionary(Text, Boolean)", "UseDictionary", "\tinput.Add(4, \"10\")",
				"\tinput")).ParseMembersAndMethods(new MethodExpressionParser());
		Assert.That(() => dictionary.Methods[0].GetBodyAndParseIfNeeded(),
			Throws.InnerException.InstanceOf<Type.ArgumentsDoNotMatchMethodParameters>().With.InnerException.Message.Contains(
				"Arguments: 4 TestPackage.Number, \"10\" TestPackage.Text do not match these TestPackage.Dictionary(TestPackage.Text, TestPackage.Boolean) method(s):" +
				"\nAdd(key TestPackage.Text, mappedValue TestPackage.Boolean) Mutable(TestPackage.Dictionary)"));
	}

	[Test]
	public void CreateAndValidateDictionaryTypeInstance()
	{
		var body = (Body)new Type(type.Package,
				new TypeLines("SchoolRegister", "has log", "LogStudentsDetails",
					"\tmutable studentsRegister = Dictionary(Number, Text)",
					"\tstudentsRegister.Add(1, \"AK\")", "\tlog.Write(studentsRegister)")).
			ParseMembersAndMethods(new MethodExpressionParser()).Methods[0].GetBodyAndParseIfNeeded();
		Assert.That(((MutableDeclaration)body.Expressions[0]).Value, Is.InstanceOf<Dictionary>());
		var dictionaryExpression = (Dictionary)((MutableDeclaration)body.Expressions[0]).Value;
		Assert.That(dictionaryExpression.KeyType, Is.EqualTo(type.GetType(Base.Number)));
		Assert.That(dictionaryExpression.MappedValueType, Is.EqualTo(type.GetType(Base.Text)));
	}

	[Test]
	public void DictionaryMustBeInitializedWithTwoTypeParameters() =>
		Assert.That(
			() => new Type(type.Package,
					new TypeLines(nameof(DictionaryMustBeInitializedWithTwoTypeParameters), "has log",
						"DummyInitialization",
						"\tmutable studentsRegister = Dictionary(Number, Text, Number)")).
				ParseMembersAndMethods(new MethodExpressionParser()).Methods[0].GetBodyAndParseIfNeeded(),
			Throws.InstanceOf<Dictionary.DictionaryMustBeInitializedWithTwoTypeParameters>().With.
				Message.StartsWith("Dictionary(Number, Text, Number)"));

	[Test]
	public void CannotAddMismatchingInputTypesToDictionaryInstance() =>
		Assert.That(
			() => new Type(type.Package,
					new TypeLines(nameof(CannotAddMismatchingInputTypesToDictionaryInstance), "has log",
						"DummyInitialization", "\tconstant studentsRegister = Dictionary(Number, Text)",
						"\tstudentsRegister.Add(5, true)", "\tlog.Write(studentsRegister)")).
				ParseMembersAndMethods(new MethodExpressionParser()).Methods[0].GetBodyAndParseIfNeeded(),
			Throws.InstanceOf<ParsingFailed>().With.InnerException.
				InstanceOf<Type.ArgumentsDoNotMatchMethodParameters>());

	[Test]
	public void CannotCreateDictionaryExpressionWithThreeTypeParameters() =>
		Assert.That(
			() => new Dictionary(
				new List<Type>
				{
					type.GetType(Base.Number), type.GetType(Base.Text), type.GetType(Base.Boolean)
				}, type),
			Throws.InstanceOf<Dictionary.DictionaryMustBeInitializedWithTwoTypeParameters>().With.
				Message.StartsWith(
					"Expected Type Parameters: 2, Given type parameters: 3 and they are TestPackage.Number, TestPackage.Text, TestPackage.Boolean"));
}