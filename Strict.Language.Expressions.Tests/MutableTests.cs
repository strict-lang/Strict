using NUnit.Framework;

namespace Strict.Language.Expressions.Tests;

public sealed class MutableTests : TestExpressions
{
	[SetUp]
	public void CreateParser() => parser = new MethodExpressionParser();

	private MethodExpressionParser parser = null!;

	[Test]
	public void MutableMemberConstructorWithType()
	{
		var program = new Type(type.Package,
				new TypeLines(nameof(MutableMemberConstructorWithType), "has something Mutable(Number)",
					"Add(input Count) Number",
					"\tlet result = something + input")).
			ParseMembersAndMethods(parser);
		Assert.That(program.Members[0].IsMutable, Is.True);
		Assert.That(program.Methods[0].GetBodyAndParseIfNeeded().ReturnType,
			Is.EqualTo(type.GetType(Base.Number)));
	}

	[Test]
	public void MutableMethodParameterWithType()
	{
		var program = new Type(type.Package,
				new TypeLines(nameof(MutableMethodParameterWithType), "has something Character",
					"Add(input Mutable(Number)) Number",
					"\tlet result = input + something")).
			ParseMembersAndMethods(parser);
		Assert.That(program.Methods[0].Parameters[0].IsMutable,
			Is.True);
		Assert.That(program.Methods[0].GetBodyAndParseIfNeeded().ReturnType,
			Is.EqualTo(type.GetType(Base.Number)));
	}

	[Test]
	public void MutableMemberWithTextType()
	{
		var program = new Type(type.Package,
				new TypeLines(nameof(MutableMemberWithTextType), "has something Mutable(Text)",
					"Add(input Count) Text",
					"\tlet result = input + something")).
			ParseMembersAndMethods(parser);
		Assert.That(() => program.Methods[0].GetBodyAndParseIfNeeded(), Throws.InstanceOf<Type.ArgumentsDoNotMatchMethodParameters>());
	}

	[Test]
	public void MutableVariablesWithSameImplementationTypeShouldUseSameType()
	{
		var program = new Type(type.Package,
			new TypeLines(nameof(MutableVariablesWithSameImplementationTypeShouldUseSameType), "has unused Number",
				"UnusedMethod Number",
				"\tlet first = Mutable 5",
				"\tlet second = Mutable 6",
				"\tfirst + second")).ParseMembersAndMethods(parser);
		var body = (Body)program.Methods[0].GetBodyAndParseIfNeeded();
		Assert.That(body.Expressions[0].ReturnType.Name, Is.EqualTo(Base.Mutable + "(TestPackage." + Base.Number + ")"));
		Assert.That(body.Expressions[0].ReturnType, Is.EqualTo(body.Expressions[1].ReturnType));
	}

	[TestCase("AssignNumberToTextType", "has something Mutable(Text)",
		"TryChangeMutableDataType Text", "\tsomething = 5")]
	[TestCase("AssignNumbersToTexts", "has something = Mutable Texts",
		"TryChangeMutableDataType Text", "\tsomething = (5, 4, 3)")]
	public void InvalidDataAssignment(string testName, params string[] code) =>
		Assert.That(
			() => new Type(type.Package, new TypeLines(testName, code)).ParseMembersAndMethods(parser).
				Methods[0].GetBodyAndParseIfNeeded(), Throws.InstanceOf<Mutable.InvalidDataAssignment>());

	[Test]
	public void MutableVariableInstanceUsingSpace()
	{
		var program = new Type(type.Package,
				new TypeLines(nameof(MutableVariableInstanceUsingSpace), "has log",
					"Add(input Count) Number",
					"\tlet result = Mutable 5",
					"\tresult = result + input")).
			ParseMembersAndMethods(parser);
		var body = (Body)program.Methods[0].GetBodyAndParseIfNeeded();
		Assert.That(((Assignment)body.Expressions[0]).Value.ToString(), Is.EqualTo("Mutable 5"));
	}

	[Test]
	public void MissingMutableArgument() =>
		Assert.That(
			() => new Type(type.Package,
					new TypeLines(nameof(MissingMutableArgument), "has log", "Add(input Count) Number",
						"\tlet result = Mutable", "\tresult = result + input")).
				ParseMembersAndMethods(parser).
				Methods[0].GetBodyAndParseIfNeeded(),
			Throws.InstanceOf<Mutable.MissingMutableArgument>());

	[TestCase("Mutable(5)", "NumberArgument")]
	[TestCase("Mutable(log)", "MutableLog")]
	public void BracketsNotAllowedForSingleArgumentsUseSpaceSyntax(string code, string testName) =>
		Assert.That(
			() => new Type(type.Package,
					new TypeLines(testName, "has log", "Add(input Count) Number",
						$"\tlet result = {code}", "\tresult = result + input")).
				ParseMembersAndMethods(parser).
				Methods[0].GetBodyAndParseIfNeeded(),
			Throws.InstanceOf<Mutable.BracketsNotAllowedForSingleArgumentsUseSpaceSyntax>());

	[TestCase("Mutable(1, 2, 3)", "Numbers", "MutableTypeWithListArgumentIsAllowed")]
	[TestCase("Mutable(Range(1, 10).Start)", "Number", "MutableTypeWithNestedCallShouldUseBrackets")]
	public void MutableTypeWithListArgumentIsAllowed(string code, string returnType, string testName)
	{
		var program = new Type(type.Package,
				new TypeLines(testName, "has log",
					$"Add(input Count) {returnType}",
					$"\tlet result = {code}",
					"\tresult = result + input",
					"\tresult")).
			ParseMembersAndMethods(parser);
		var body = (Body)program.Methods[0].GetBodyAndParseIfNeeded();
		Assert.That(((Assignment)body.Expressions[0]).Value.ToString(),
			Is.EqualTo(code));
	}
}