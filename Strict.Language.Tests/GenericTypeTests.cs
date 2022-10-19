using NUnit.Framework;
using Strict.Language.Expressions;

namespace Strict.Language.Tests;

public sealed class GenericTypeTests : TypeTests
{
	[SetUp]
	public void CreateParserAndComparerType()
	{
		parser = new MethodExpressionParser();
		CreateType("Comparer",
			new[]
			{
				"has FirstType Generic", "has SecondType Generic", "Compare",
				"\tfirstType is secondType"
			});
	}

	private ExpressionParser parser = null!;

	private void CreateType(string name, string[] lines) =>
		new Type(package,
			new TypeLines(name, lines)).ParseMembersAndMethods(parser);

	[Test]
	public void TypeArgumentsDoNotMatchWithMainTypeConstructor() =>
		Assert.That(
			() => new Type(package,
				new TypeLines("SimpleProgram", "has something Comparer(Text)", "Invoke",
					"\tlet result = something.Compare")).ParseMembersAndMethods(parser),
			Throws.InstanceOf<ParsingFailed>().With.InnerException.
				InstanceOf<Context.TypeArgumentsDoNotMatchWithMainType>().With.Message.Contains(
					"Argument(s) (TestPackage.Text) does not match type Comparer with constructor Comparer(FirstType TestPackage.Generic, SecondType TestPackage.Generic)"));

	[Test]
	public void GenericTypeWithMultipleImplementations()
	{
		var usingGenericType = new Type(package,
			new TypeLines("SimpleProgram",
				"has something Comparer(Text, Number)",
				"Invoke",
				"\tlet result = something.Compare")).ParseMembersAndMethods(new MethodExpressionParser());
		Assert.That(usingGenericType.Members[0].Type.Name, Is.EqualTo("Comparer(TestPackage.Text, TestPackage.Number)"));
	}
}