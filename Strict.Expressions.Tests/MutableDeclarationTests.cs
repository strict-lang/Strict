using NUnit.Framework;
using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict.Expressions.Tests;

public sealed class MutableDeclarationTests : TestExpressions
{
	[SetUp]
	public void CreateParser() => parser = new MethodExpressionParser();

	private MethodExpressionParser parser = null!;

	[Test]
	public void MutableMemberConstructorWithType()
	{
		var program = new Type(type.Package,
				new TypeLines(nameof(MutableMemberConstructorWithType), "mutable something Number",
					"Add(input Number) Number",
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
					"Add(mutable input Number, mutable text) Number",
					"\tinput = something + input")).
			ParseMembersAndMethods(parser);
		Assert.That(program.Methods[0].Parameters[0].IsMutable,
			Is.True);
		Assert.That(program.Methods[0].Parameters[1].IsMutable,
			Is.True);
		Assert.That(program.Methods[0].GetBodyAndParseIfNeeded().ReturnType,
			Is.EqualTo(type.GetType(Base.Number)));
	}

	[Test]
	public void EnsureMutableMethodParameterValueIsUpdated()
	{
		var program = new Type(type.Package,
				new TypeLines(nameof(EnsureMutableMethodParameterValueIsUpdated), "has something Number",
					"Add(mutable input Number) Number",
					"\tinput = something + input")).
			ParseMembersAndMethods(parser);
		Assert.That(program.Methods[0].Parameters[0].IsMutable,
			Is.True);
		program.Methods[0].GetBodyAndParseIfNeeded();
		Assert.That(program.Methods[0].Parameters[0].DefaultValue, Is.Null);
	}

	[Test]
	public void IncompleteMutableMethodParameter() =>
		Assert.That(
			() => new Type(type.Package,
					new TypeLines(nameof(IncompleteMutableMethodParameter), "has something Number",
						"Add(mutable input) Number", "\tinput = something + input")).
				ParseMembersAndMethods(parser),
			Throws.InstanceOf<ParsingFailed>().With.InnerException.InstanceOf<Context.TypeNotFound>());

	[Test]
	public void MutableMemberWithTextType()
	{
		var program = new Type(type.Package,
				new TypeLines(nameof(MutableMemberWithTextType), "mutable something Text",
					"Add(input Number) Text",
					"\tconstant result = input + something")).
			ParseMembersAndMethods(parser);
		Assert.That(() => program.Methods[0].GetBodyAndParseIfNeeded(),
			Throws.InstanceOf<ParsingFailed>().With.InnerException.
				InstanceOf<Type.ArgumentsDoNotMatchMethodParameters>());
	}

	[Test]
	public void MutableVariablesUsingSameValueTypeMustBeEqual()
	{
		var program = new Type(type.Package,
			new TypeLines(nameof(MutableVariablesUsingSameValueTypeMustBeEqual), "has unused Number",
				"UnusedMethod Number",
				"\tmutable first = 5",
				"\tmutable second = 6",
				"\tfirst + second")).ParseMembersAndMethods(parser);
		var body = (Body)program.Methods[0].GetBodyAndParseIfNeeded();
		Assert.That(body.Expressions[0].IsMutable, Is.True);
		Assert.That(body.Expressions[0].ReturnType, Is.EqualTo(body.Expressions[1].ReturnType));
	}

	[TestCase("AssignNumberToTextType", "mutable something Number", "TryChangeMutableDataType Text",
		"\tsomething = \"5\"")]
	[TestCase("AssignNumbersToTexts", "mutable something Numbers", "TryChangeMutableDataType Text",
		"\tsomething = (\"5\", \"4\", \"3\")")]
	public void ValueTypeNotMatchingWithAssignmentType(string testName, params string[] code) =>
		Assert.That(
			() => new Type(type.Package, new TypeLines(testName, code)).ParseMembersAndMethods(parser).
				Methods[0].GetBodyAndParseIfNeeded(),
			Throws.InstanceOf<MutableReassignment.ValueTypeNotMatchingWithAssignmentType>());

	[Test]
	public void MutableVariableInstanceUsingSpace()
	{
		var program = new Type(type.Package,
				new TypeLines(nameof(MutableVariableInstanceUsingSpace), "has logger",
					"Add(input Number) Number",
					"\tmutable result = 5",
					"\tresult = result + input")).
			ParseMembersAndMethods(parser);
		var body = (Body)program.Methods[0].GetBodyAndParseIfNeeded();
		Assert.That(((ConstantDeclaration)body.Expressions[0]).Value.ToString(), Is.EqualTo("5"));
	}

	[Test]
	public void MissingMutableArgument() =>
		Assert.That(
			() => new Type(type.Package,
					new TypeLines(nameof(MissingMutableArgument), "has logger", "Add(input Number) Number",
						"\tconstant result =", "\tresult = result + input")).
				ParseMembersAndMethods(parser).
				Methods[0].GetBodyAndParseIfNeeded(),
			Throws.InstanceOf<ConstantDeclaration.MissingAssignmentValueExpression>());

	[TestCase("(1, 2, 3)", "Numbers", "MutableTypeWithListArgumentIsAllowed")]
	[TestCase("Range(1, 10).Start", "Number", "MutableTypeWithNestedCallShouldUseBrackets")]
	public void MutableTypeWithListArgumentIsAllowed(string code, string returnType, string testName)
	{
		var program = new Type(type.Package,
				new TypeLines(testName, "has logger",
					$"Add(input Number) {returnType}",
					$"\tmutable result = {code}",
					"\tresult = result + input",
					"\tresult")).
			ParseMembersAndMethods(parser);
		var body = (Body)program.Methods[0].GetBodyAndParseIfNeeded();
		Assert.That(((ConstantDeclaration)body.Expressions[0]).Value.ToString(),
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
			Is.EqualTo("Number"));
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
			Throws.InstanceOf<ConstantDeclaration.MissingAssignmentValueExpression>());
	}

	[Test]
	public void DirectUsageOfMutableTypesOrImplementsAreForbidden()
	{
		var program = new Type(type.Package,
				new TypeLines(nameof(DirectUsageOfMutableTypesOrImplementsAreForbidden), "has unused Character",
					"DummyCount(limit Number) Number",
					"\tconstant result = Mutable(5)",
					"\tresult")).
			ParseMembersAndMethods(parser);
		Assert.That(() => program.Methods[0].GetBodyAndParseIfNeeded(),
			Throws.InstanceOf<ParsingFailed>().With.InnerException.InstanceOf<Type.GenericTypesCannotBeUsedDirectlyUseImplementation>());
	}

	[Test]
	public void GenericTypesCannotBeUsedDirectlyUseImplementation()
	{
		var program = new Type(type.Package,
				new TypeLines(nameof(GenericTypesCannotBeUsedDirectlyUseImplementation), "has unused Character",
					"DummyCount Number",
					"\tconstant result = List(5, 5)",
					"\tresult")).
			ParseMembersAndMethods(parser);
		Assert.That(() => program.Methods[0].GetBodyAndParseIfNeeded(),
			Throws.InstanceOf<ParsingFailed>().With.InnerException.
				InstanceOf<Type.GenericTypesCannotBeUsedDirectlyUseImplementation>());
	}

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
		Assert.That(program.Members[0].InitialValue?.ToString(), Is.EqualTo("0"));
	}

	[TestCase("Mutable", "Mutable(Number)")]
	[TestCase("Count", "Count")]
	public void MutableTypesUsageInMembersAreForbidden(string testName, string code) =>
		Assert.That(
			() => new Type(type.Package,
				new TypeLines(testName + nameof(MutableTypesUsageInMembersAreForbidden),
					$"mutable something {code}", "Add(input Count) Number",
					"\tconstant result = something + input")).ParseMembersAndMethods(parser),
			Throws.InstanceOf<ParsingFailed>().With.InnerException.InstanceOf<Context.TypeNotFound>());

	[Test]
	public void CannotReassignValuesToImmutableMember()
	{
		new Type(type.Package,
				new TypeLines("BaseClever", "mutable Number", "Compute Number", "\t5 + Number")).
			ParseMembersAndMethods(parser);
		Assert.That(
			() => new Type(type.Package,
					new TypeLines(nameof(CannotReassignValuesToImmutableMember), "has input = BaseClever(3)",
						"Run", "\tinput.Compute", "\tinput = BaseClever(5)")).ParseMembersAndMethods(parser).
				Methods[0].GetBodyAndParseIfNeeded(),
			Throws.InstanceOf<Body.ValueIsNotMutableAndCannotBeChanged>());
	}

	[Test]
	public void ModifyMutableMemberValueUsingTypeInstance()
	{
		new Type(type.Package,
				new TypeLines("Clever", "mutable Number", "Compute Number", "\t5 + Number")).
			ParseMembersAndMethods(parser);
		var cleverConsumerType = new Type(type.Package,
			new TypeLines(nameof(ModifyMutableMemberValueUsingTypeInstance), "has clever = Clever(3)",
				"Run", "\tclever.Compute is 8", "\tclever.Number = 5")).ParseMembersAndMethods(parser);
		cleverConsumerType.Methods[0].GetBodyAndParseIfNeeded();
		Assert.That(type.GetType("Clever").Members[0].InitialValue, Is.Null);
	}

	[Test]
	public void ModifyMutableMembersMultipleTimes()
	{
		new Type(type.Package,
				new TypeLines("Computer", "mutable Number", "Compute Number", "\t5 + Number")).
			ParseMembersAndMethods(parser);
		var cleverConsumerType = new Type(type.Package,
			new TypeLines(nameof(ModifyMutableMembersMultipleTimes), "has computer = Computer(3)",
				"Run",
				"\tconstant bla = 5",
				"\tmutable blub = Compute",
				"\tconstant number = bla + 1",
				"\tmutable swappedBlub = blub",
				"\tblub = 49",
				"\tmutable temporary = swappedBlub",
				"\tswappedBlub = 50",
				"\ttemporary is 9",
				"\ttemporary is 10",
				"Compute Number",
				"\tcomputer.Number.Increment",
				"\tcomputer.Compute")).ParseMembersAndMethods(parser);
		var body = (Body)cleverConsumerType.
			Methods[0].GetBodyAndParseIfNeeded();
		Assert.That(body.FindVariable("blub")?.InitialValue.ToString(), Is.EqualTo("Compute"));
		Assert.That(body.FindVariable("swappedBlub")?.InitialValue.ToString(), Is.EqualTo("blub"));
		Assert.That(body.FindVariable("temporary")?.InitialValue.ToString(), Is.EqualTo("swappedBlub"));
	}
}