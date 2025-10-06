namespace Strict.Language.Tests;

public sealed class GenericTypeImplementationTests
{
	[SetUp]
	public void CreateParserAndComparerType()
	{
		parser = new MethodExpressionParser();
		comparerType = CreateType("Comparer", [
			"has FirstTypes Generics",
			"has SecondType Generic",
			"Compare",
			"\tfirstType is secondType"
		]);
		customType = CreateType("CustomType", ["from(first Generic, second Generic)"]);
	}

	private ExpressionParser parser = null!;
	private Type comparerType = null!;
	private Type customType = null!;

	private Type CreateType(string name, string[] lines) =>
		new Type(TestPackage.Instance, new TypeLines(name, lines)).ParseMembersAndMethods(parser);

	[TearDown]
	public void TearDown()
	{
		comparerType.Dispose();
		customType.Dispose();
	}

	[Test]
	public void TypeArgumentsDoNotMatchGenericTypeConstructor() =>
		Assert.That(
			() =>
			{
				using var type = new Type(TestPackage.Instance,
					new TypeLines("SimpleProgram", "has something Comparer(Text)", "Invoke",
						"\tconstant result = something.Compare"));
				return type.ParseMembersAndMethods(parser);
			}, //ncrunch: no coverage
			Throws.InstanceOf<ParsingFailed>().With.InnerException.
				InstanceOf<Context.TypeArgumentsCountDoesNotMatchGenericType>().With.Message.Contains(
					"The generic type TestPackage.Comparer needs these type arguments: (Generic TestPackage." +
					"Generic, SecondType TestPackage.Generic), this does not match provided types: (TestPackage." +
					"Text)"));

	[Test]
	public void GenericTypeWithMultipleImplementations()
	{
		using var usingGenericType = new Type(TestPackage.Instance,
			new TypeLines("SimpleProgram",
				"has something Comparer(Text, Number)",
				"Invoke",
				"\tconstant result = something.Compare")).ParseMembersAndMethods(new MethodExpressionParser());
		Assert.That(usingGenericType.Members[0].Type.Name,
			Is.EqualTo("Comparer(TestPackage.Text, TestPackage.Number)"));
		var genericComparer = ((GenericTypeImplementation)usingGenericType.Members[0].Type).Generic;
		Assert.That(genericComparer.Name, Is.EqualTo("Comparer"));
	}

	[Test]
	public void CannotGetGenericImplementationOnNonGeneric() =>
		Assert.That(
			() =>
			{
				using var type = new Type(TestPackage.Instance,
					new TypeLines(nameof(CannotGetGenericImplementationOnNonGeneric),
						"has custom Boolean(Number, Text)"));
				return type.ParseMembersAndMethods(new MethodExpressionParser());
			}, //ncrunch: no coverage
			Throws.InstanceOf<ParsingFailed>().With.InnerException.
				InstanceOf<Type.CannotGetGenericImplementationOnNonGeneric>());

	[Test]
	public void TypeArgumentsDoNotMatchGenericType() =>
		Assert.That(
			() =>
			{
				using var type = new Type(TestPackage.Instance,
					new TypeLines(nameof(TypeArgumentsDoNotMatchGenericType), "has custom Comparer(Number)",
						"UnusedMethod Number", "\t5"));
				return type.ParseMembersAndMethods(new MethodExpressionParser());
			}, //ncrunch: no coverage
			Throws.InstanceOf<ParsingFailed>().With.InnerException.
				InstanceOf<Context.TypeArgumentsCountDoesNotMatchGenericType>());
}