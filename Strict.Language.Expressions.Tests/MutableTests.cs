using NUnit.Framework;
using static Strict.Language.Expressions.Assignment;

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
				new TypeLines(nameof(MutableMemberConstructorWithType), "mutable something Number",
					"Add(input Count) Number",
					"\tconstant result = something + input")).
			ParseMembersAndMethods(parser);
		Assert.That(program.Members[0].IsMutable, Is.True);
		Assert.That(program.Methods[0].GetBodyAndParseIfNeeded().ReturnType,
			Is.EqualTo(type.GetType(Base.Number)));
	}

	[Test]
	public void MutableMethodParameterWithType()
	{
		var program = new Type(type.Package,
				new TypeLines(nameof(MutableMethodParameterWithType), "has something Number",
					"Add(input Mutable(Number)) Number",
					"\tconstant result = something + input")).
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
				new TypeLines(nameof(MutableMemberWithTextType), "mutable something Text",
					"Add(input Count) Text",
					"\tconstant result = input + something")).
			ParseMembersAndMethods(parser);
		Assert.That(() => program.Methods[0].GetBodyAndParseIfNeeded(), Throws.InstanceOf<Type.ArgumentsDoNotMatchMethodParameters>());
	}

	[Test]
	public void MutableVariablesWithSameImplementationTypeShouldUseSameType()
	{
		var program = new Type(type.Package,
			new TypeLines(nameof(MutableVariablesWithSameImplementationTypeShouldUseSameType), "has unused Number",
				"UnusedMethod Number",
				"\tmutable first = 5",
				"\tmutable second = 6",
				"\tfirst + second")).ParseMembersAndMethods(parser);
		var body = (Body)program.Methods[0].GetBodyAndParseIfNeeded();
		Assert.That(body.Expressions[0].ReturnType.Name, Is.EqualTo(Base.Mutable + "(TestPackage." + Base.Number + ")"));
		Assert.That(body.Expressions[0].ReturnType, Is.EqualTo(body.Expressions[1].ReturnType));
	}

	[TestCase("AssignNumberToTextType", "mutable something Text",
		"TryChangeMutableDataType Text", "\tsomething = 5")]
	[TestCase("AssignNumbersToTexts", "mutable something Texts",
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
					"\tmutable result = 5",
					"\tresult = result + input")).
			ParseMembersAndMethods(parser);
		var body = (Body)program.Methods[0].GetBodyAndParseIfNeeded();
		Assert.That(((Assignment)body.Expressions[0]).Value.ToString(), Is.EqualTo("5"));
	}

	[Test]
	public void MissingMutableArgument() =>
		Assert.That(
			() => new Type(type.Package,
					new TypeLines(nameof(MissingMutableArgument), "has log", "Add(input Count) Number",
						"\tconstant result =", "\tresult = result + input")).
				ParseMembersAndMethods(parser).
				Methods[0].GetBodyAndParseIfNeeded(),
			Throws.InstanceOf<MissingAssignmentValueExpression>());

	[TestCase("(1, 2, 3)", "Numbers", "MutableTypeWithListArgumentIsAllowed")]
	[TestCase("Range(1, 10).Start", "Number", "MutableTypeWithNestedCallShouldUseBrackets")]
	public void MutableTypeWithListArgumentIsAllowed(string code, string returnType, string testName)
	{
		var program = new Type(type.Package,
				new TypeLines(testName, "has log",
					$"Add(input Count) {returnType}",
					$"\tmutable result = {code}",
					"\tresult = result + input",
					"\tresult")).
			ParseMembersAndMethods(parser);
		var body = (Body)program.Methods[0].GetBodyAndParseIfNeeded();
		Assert.That(((Assignment)body.Expressions[0]).Value.ToString(),
			Is.EqualTo(code));
	}

	[Test]
	public void AssignmentWithMutableKeyword()
	{
		var program = new Type(type.Package,
				new TypeLines(nameof(AssignmentWithMutableKeyword), "has something Character",
					"CountEvenNumbers(limit Number) Number",
					"\tmutable counter = 0",
					"\tfor Range(0, limit)",
					"\t\tcounter = counter + 1",
					"\tcounter")).
			ParseMembersAndMethods(parser);
		var body = (Body)program.Methods[0].GetBodyAndParseIfNeeded();
		Assert.That(body.ReturnType,
			Is.EqualTo(type.GetType(Base.Number)));
		Assert.That(body.Expressions[0].ReturnType.Name,
			Is.EqualTo("Mutable(TestPackage.Number)"));
	}

	[Test]
	public void MissingAssignmentValueExpression()
	{
		var program = new Type(type.Package,
				new TypeLines(nameof(MissingAssignmentValueExpression), "has something Character",
					"CountEvenNumbers(limit Number) Number",
					"\tmutable counter =",
					"\tcounter")).
			ParseMembersAndMethods(parser);
		Assert.That(() => program.Methods[0].GetBodyAndParseIfNeeded(),
			Throws.InstanceOf<MissingAssignmentValueExpression>());
	}

	[Test]
	public void DirectUsageOfMutableTypesOrImplementsAreForbidden()
	{
		var program = new Type(type.Package,
				new TypeLines(nameof(DirectUsageOfMutableTypesOrImplementsAreForbidden), "has unused Character",
					"DummyCount(limit Number) Number",
					"\tconstant result = Count(5)",
					"\tresult")).
			ParseMembersAndMethods(parser);
		Assert.That(() => program.Methods[0].GetBodyAndParseIfNeeded(),
			Throws.InstanceOf<DirectUsageOfMutableTypesOrImplementsAreForbidden>()!);
	}

	[Test]
	public void GenericTypesCannotBeUsedDirectlyUseImplementation()
	{
		var program = new Type(type.Package,
				new TypeLines(nameof(GenericTypesCannotBeUsedDirectlyUseImplementation), "has unused Character",
					"DummyCount Number",
					"\tconstant result = Mutable(5)",
					"\tresult")).
			ParseMembersAndMethods(parser);
		Assert.That(() => program.Methods[0].GetBodyAndParseIfNeeded(),
			Throws.InstanceOf<Type.GenericTypesCannotBeUsedDirectlyUseImplementation>()!);
	}

	[Test]
	public void MemberCannotBeAssignedWithMutableType() =>
		Assert.That(
			() => new Type(type.Package,
					new TypeLines(nameof(MemberCannotBeAssignedWithMutableType), "has input = Mutable(0)",
						"DummyMethod Number", "\tconstant result = Count(5)", "\tresult")).
				ParseMembersAndMethods(parser),
			Throws.InstanceOf<ParsingFailed>().With.InnerException.
				InstanceOf<Type.GenericTypesCannotBeUsedDirectlyUseImplementation>());

	[Test]
	public void MemberDeclarationUsingMutableKeyword()
	{
		var program = new Type(type.Package,
				new TypeLines(nameof(MemberDeclarationUsingMutableKeyword), "mutable input = 0",
					"DummyAssignment(limit Number) Number",
					"\tif limit > 5",
					"\t\tinput = 5",
					"\telse",
					"\t\tinput = 10",
					"\tinput")).
			ParseMembersAndMethods(parser);
		program.Methods[0].GetBodyAndParseIfNeeded();
		Assert.That(program.Members[0].IsMutable, Is.True);
		Assert.That(program.Members[0].Value?.ToString(), Is.EqualTo("10"));
	}

	[TestCase("Mutable", "Mutable(Number)")]
	[TestCase("Count", "Count")]
	[TestCase("CountWithValue", "= Count(5)")]
	public void MutableTypesOrImplementsUsageInMembersAreForbidden(string testName, string code) =>
		Assert.That(
			() => new Type(type.Package,
				new TypeLines(testName + nameof(MutableTypesOrImplementsUsageInMembersAreForbidden),
					$"mutable something {code}", "Add(input Count) Number",
					"\tconstant result = something + input")).ParseMembersAndMethods(parser),
			Throws.InstanceOf<Type.UsingMutableTypesOrImplementsAreNotAllowed>());
}