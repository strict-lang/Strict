using NUnit.Framework;
using Strict.Language.Expressions;

namespace Strict.Language.Tests;

public sealed class GenericTypeTests
{
	[SetUp]
	public void CreateParserAndComparerType()
	{
		parser = new MethodExpressionParser();
		package = new TestPackage();
		CreateType("Comparer",
			new[]
			{
				"has FirstTypes Generics",
				"has SecondType Generic",
				"Compare",
				"\tfirstType is secondType"
			});
		CreateType("CustomType", new[] { "from(first Generic, second Generic)" });
	}

	private ExpressionParser parser = null!;
	public Package package = null!;

	private void CreateType(string name, string[] lines) =>
		new Type(package,
			new TypeLines(name, lines)).ParseMembersAndMethods(parser);

	[Test]
	public void TypeArgumentsDoNotMatchGenericTypeConstructor() =>
		Assert.That(
			() => new Type(package,
				new TypeLines("SimpleProgram", "has something Comparer(Text)", "Invoke",
					"\tconstant result = something.Compare")).ParseMembersAndMethods(parser),
			Throws.InstanceOf<ParsingFailed>().With.InnerException.
				InstanceOf<Context.TypeArgumentsCountDoesNotMatchGenericType>().With.Message.Contains(
					"The generic type Comparer needs these type arguments: (FirstTypes TestPackage.List, SecondType TestPackage.Generic), does not match provided types: (TestPackage.Text)"));

	[Test]
	public void GenericTypeWithMultipleImplementations()
	{
		var usingGenericType = new Type(package,
			new TypeLines("SimpleProgram",
				"has something Comparer(Text, Number)",
				"Invoke",
				"\tconstant result = something.Compare")).ParseMembersAndMethods(new MethodExpressionParser());
		Assert.That(usingGenericType.Members[0].Type.Name, Is.EqualTo("Comparer(TestPackage.Text, TestPackage.Number)"));
	}

	[Test]
	public void CannotGetGenericImplementationOnNonGeneric() =>
		Assert.That(
			() => new Type(package,
					new TypeLines(nameof(CannotGetGenericImplementationOnNonGeneric),
						"has custom Boolean(Number, Text)")).
				ParseMembersAndMethods(new MethodExpressionParser()),
			Throws.InstanceOf<ParsingFailed>().With.InnerException.
				InstanceOf<Type.CannotGetGenericImplementationOnNonGeneric>());

	[Test]
	public void TypeArgumentsDoNotMatchGenericType() =>
		Assert.That(
			() => new Type(package,
					new TypeLines(nameof(TypeArgumentsDoNotMatchGenericType),
						"has custom Comparer(Number)", "UnusedMethod Number", "\t5")).
				ParseMembersAndMethods(new MethodExpressionParser()),
			Throws.InstanceOf<ParsingFailed>().With.InnerException.
				InstanceOf<Context.TypeArgumentsCountDoesNotMatchGenericType>());
}