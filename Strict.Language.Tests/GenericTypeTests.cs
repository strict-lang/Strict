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
				"has FirstType Generic", "has SecondType Generic", "Compare",
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
	public void TypeArgumentsDoNotMatchWithMainTypeConstructor() =>
		Assert.That(
			() => new Type(package,
				new TypeLines("SimpleProgram", "has something Comparer(Text)", "Invoke",
					"\tconstant result = something.Compare")).ParseMembersAndMethods(parser),
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
				"\tconstant result = something.Compare")).ParseMembersAndMethods(new MethodExpressionParser());
		Assert.That(usingGenericType.Members[0].Type.Name, Is.EqualTo("Comparer(TestPackage.Text, TestPackage.Number)"));
	}

	[Test]
	public void ConsumeCustomGenericTypeWithCustomConstructor()
	{
		var consumeCustomType = new Type(package,
			new TypeLines(nameof(ConsumeCustomGenericTypeWithCustomConstructor),
				"has custom CustomType(Number, Text)",
				"UnusedMethod Number",
				"\t5")).ParseMembersAndMethods(new MethodExpressionParser());
		Assert.That(consumeCustomType.Members[0].Type.Name, Is.EqualTo("CustomType"));
	}

	[Test]
	public void CustomGenericTypeWithInvalidTypeArguments() =>
		Assert.That(
			() => new Type(package,
					new TypeLines(nameof(CustomGenericTypeWithInvalidTypeArguments),
						"has custom CustomType(Number)", "UnusedMethod Number", "\t5")).
				ParseMembersAndMethods(new MethodExpressionParser()),
			Throws.InstanceOf<ParsingFailed>().With.InnerException.
				InstanceOf<Type.ArgumentsDoNotMatchMethodParameters>());
}