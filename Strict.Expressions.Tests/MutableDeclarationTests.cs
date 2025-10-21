namespace Strict.Expressions.Tests;

public sealed class MutableDeclarationTests : TestExpressions
{
	[SetUp]
	public void CreateParser() => parser = new MethodExpressionParser();

	private MethodExpressionParser parser = null!;

	[Test]
	public void MutableMemberConstructorWithType()
	{
		using var program = new Type(type.Package,
			new TypeLines(nameof(MutableMemberConstructorWithType), "mutable something Number",
				"Add(input Number) Number", "\tlet result = something + input"));
		program.ParseMembersAndMethods(parser);
		Assert.That(program.Members[0].IsMutable, Is.True);
		Assert.That(program.Methods[0].GetBodyAndParseIfNeeded().ReturnType,
			Is.EqualTo(type.GetType(Base.Number)));
	}

	[Test]
	public void MutableMethodParameterWithType()
	{
		using var program = new Type(type.Package,
			new TypeLines(nameof(MutableMethodParameterWithType), "has something Number",
				"Add(mutable input Number, mutable text) Number", "\tinput = something + input"));
		program.ParseMembersAndMethods(parser);
		Assert.That(program.Methods[0].Parameters[0].IsMutable, Is.True);
		Assert.That(program.Methods[0].Parameters[1].IsMutable, Is.True);
		Assert.That(program.Methods[0].GetBodyAndParseIfNeeded().ReturnType,
			Is.EqualTo(type.GetType(Base.Number)));
	}

	[Test]
	public void EnsureMutableMethodParameterValueIsUpdated()
	{
		using var program = new Type(type.Package,
			new TypeLines(nameof(EnsureMutableMethodParameterValueIsUpdated), "has something Number",
				"Add(mutable input Number) Number", "\tinput = something + input"));
		program.ParseMembersAndMethods(parser);
		Assert.That(program.Methods[0].Parameters[0].IsMutable, Is.True);
		program.Methods[0].GetBodyAndParseIfNeeded();
		Assert.That(program.Methods[0].Parameters[0].DefaultValue, Is.Null);
	}

	[Test]
	public void IncompleteMutableMethodParameter() =>
		Assert.That(
			() =>
			{
				using var dummy = new Type(type.Package,
					new TypeLines(nameof(IncompleteMutableMethodParameter), "has something Number",
						"Add(mutable input) Number", "\tinput = something + input"));
				dummy.ParseMembersAndMethods(parser);
			},
			Throws.InstanceOf<ParsingFailed>().With.InnerException.InstanceOf<Context.TypeNotFound>());

	[Test]
	public void MutableMemberWithTextType()
	{
		using var program = new Type(type.Package,
			new TypeLines(nameof(MutableMemberWithTextType), "mutable something Text",
				"Add(input Number) Text", "\tconstant result = input + something"));
		program.ParseMembersAndMethods(parser);
		Assert.That(() => program.Methods[0].GetBodyAndParseIfNeeded(),
			Throws.InstanceOf<ParsingFailed>().With.InnerException.
				InstanceOf<Type.ArgumentsDoNotMatchMethodParameters>());
	}

	[Test]
	public void MutableVariablesUsingSameValueTypeMustBeEqual()
	{
		using var program = new Type(type.Package,
			new TypeLines(nameof(MutableVariablesUsingSameValueTypeMustBeEqual), "has unused Number",
				"UnusedMethod Number",
				"\tmutable first = 5",
				"\tconstant second = 6",
				"\tfirst = first + 1",
				"\tfirst + second"));
		program.ParseMembersAndMethods(parser);
		var body = (Body)program.Methods[0].GetBodyAndParseIfNeeded();
		Assert.That(body.Expressions[0].IsMutable, Is.True);
		Assert.That(body.Expressions[0].ReturnType, Is.EqualTo(body.Expressions[1].ReturnType));
		Assert.That(body.ContainsAnythingMutable, Is.True);
	}

	[TestCase("AssignNumberToTextType", "mutable something Number", "TryChangeMutableDataType Text",
		"\tsomething = \"5\"")]
	[TestCase("AssignNumbersToTexts", "mutable something Numbers", "TryChangeMutableDataType Text",
		"\tsomething = (\"5\", \"4\", \"3\")")]
	public void ValueTypeNotMatchingWithAssignmentType(string testName, params string[] code) =>
		Assert.That(
			() =>
			{
				using var dummyType = new Type(type.Package, new TypeLines(testName, code));
				dummyType.ParseMembersAndMethods(parser).Methods[0].GetBodyAndParseIfNeeded();
			},
			Throws.InstanceOf<MutableReassignment.ValueTypeNotMatchingWithAssignmentType>());

	[Test]
	public void MutableVariableInstanceUsingSpace()
	{
		using var program = new Type(type.Package,
			new TypeLines(nameof(MutableVariableInstanceUsingSpace), "has logger",
				"Add(input Number) Number", "\tmutable result = 5", "\tresult = result + input"));
		program.ParseMembersAndMethods(parser);
		var body = (Body)program.Methods[0].GetBodyAndParseIfNeeded();
		Assert.That(((Declaration)body.Expressions[0]).Value.ToString(), Is.EqualTo("5"));
	}

	[Test]
	public void MissingMutableArgument() =>
		Assert.That(
			() =>
			{
				using var dummy = new Type(type.Package,
					new TypeLines(nameof(MissingMutableArgument), "has logger", "Add(input Number) Number",
						"\tconstant result =", "\tresult = result + input"));
				dummy.ParseMembersAndMethods(parser).Methods[0].GetBodyAndParseIfNeeded();
			},
			Throws.InstanceOf<Declaration.MissingAssignmentValueExpression>());

	[TestCase("(1, 2, 3)", "Numbers", "MutableTypeWithListArgumentIsAllowed")]
	[TestCase("Range(1, 10).Start", "Number", "MutableTypeWithNestedCallShouldUseBrackets")]
	public void MutableTypeWithListArgumentIsAllowed(string code, string returnType, string testName)
	{
		using var program = new Type(type.Package,
			new TypeLines(testName, "has logger", $"Add(input Number) {returnType}",
				$"\tmutable result = {code}", "\tresult = result + input", "\tresult"));
		program.ParseMembersAndMethods(parser);
		var body = (Body)program.Methods[0].GetBodyAndParseIfNeeded();
		Assert.That(((Declaration)body.Expressions[0]).Value.ToString(),
			Is.EqualTo(code));
	}

	[Test]
	public void AssignmentWithMutableKeyword()
	{
		using var program = new Type(type.Package,
			new TypeLines(nameof(AssignmentWithMutableKeyword), "has something Character",
				"CountEvenNumbers(limit Number) Number", "\tmutable counter = 0", "\tfor Range(0, limit)",
				"\t\tcounter = counter + 1", "\tcounter"));
		program.ParseMembersAndMethods(parser);
		var body = (Body)program.Methods[0].GetBodyAndParseIfNeeded();
		Assert.That(body.ReturnType,
			Is.EqualTo(type.GetType(Base.Number)));
		Assert.That(body.Expressions[0].ReturnType.Name,
			Is.EqualTo("Number"));
	}

	[Test]
	public void MissingAssignmentValueExpression()
	{
		using var program = new Type(type.Package,
			new TypeLines(nameof(MissingAssignmentValueExpression), "has something Character",
				"CountEvenNumbers(limit Number) Number", "\tmutable counter =", "\tcounter"));
		program.ParseMembersAndMethods(parser);
		Assert.That(() => program.Methods[0].GetBodyAndParseIfNeeded(),
			Throws.InstanceOf<Declaration.MissingAssignmentValueExpression>());
	}

	[Test]
	public void DirectUsageOfMutableTypesOrImplementsAreForbidden()
	{
		using var program = new Type(type.Package,
			new TypeLines(nameof(DirectUsageOfMutableTypesOrImplementsAreForbidden),
				"has unused Character", "DummyCount(limit Number) Number",
				"\tconstant result = Mutable(5)", "\tresult"));
		program.ParseMembersAndMethods(parser);
		Assert.That(() => program.Methods[0].GetBodyAndParseIfNeeded(),
			Throws.InstanceOf<ParsingFailed>().With.InnerException.InstanceOf<Type.GenericTypesCannotBeUsedDirectlyUseImplementation>());
	}

	[Test]
	public void GenericTypesCannotBeUsedDirectlyUseImplementation()
	{
		using var program = new Type(type.Package,
			new TypeLines(nameof(GenericTypesCannotBeUsedDirectlyUseImplementation),
				"has unused Character", "DummyCount Number", "\tconstant result = List(5, 5)",
				"\tresult"));
		program.ParseMembersAndMethods(parser);
		Assert.That(() => program.Methods[0].GetBodyAndParseIfNeeded(),
			Throws.InstanceOf<ParsingFailed>().With.InnerException.
				InstanceOf<Type.GenericTypesCannotBeUsedDirectlyUseImplementation>());
	}

	[Test]
	public void MemberDeclarationUsingMutableKeyword()
	{
		using var program = new Type(type.Package,
			new TypeLines(nameof(MemberDeclarationUsingMutableKeyword), "mutable input = 0",
				"DummyAssignment(limit Number) Number", "\tif limit > 5", "\t\tinput = 5", "\telse",
				"\t\tinput = 10", "\tinput"));
		program.ParseMembersAndMethods(parser);
		program.Methods[0].GetBodyAndParseIfNeeded();
		Assert.That(program.Members[0].IsMutable, Is.True);
		Assert.That(program.Members[0].InitialValue?.ToString(), Is.EqualTo("0"));
		Assert.That(program.Methods[0].GetBodyAndParseIfNeeded().ContainsAnythingMutable, Is.True);
	}

	[TestCase("Mutable", "Mutable(Number)")]
	[TestCase("Count", "Count")]
	public void MutableTypesUsageInMembersAreForbidden(string testName, string code) =>
		Assert.That(
			() =>
			{
				using var dummy = new Type(type.Package,
					new TypeLines(testName + nameof(MutableTypesUsageInMembersAreForbidden),
						$"mutable something {code}", "Add(input Count) Number",
						"\tconstant result = something + input"));
				dummy.ParseMembersAndMethods(parser);
			},
			Throws.InstanceOf<ParsingFailed>().With.InnerException.InstanceOf<Context.TypeNotFound>());

	[Test]
	public void CannotReassignValuesToImmutableMember()
	{
		using var dummy = new Type(type.Package,
			new TypeLines("BaseClever", "mutable Number", "Compute Number", "\t5 + Number"));
		dummy.ParseMembersAndMethods(parser);
		Assert.That(
			() =>
			{
				using var innerDummy = new Type(type.Package,
					new TypeLines(nameof(CannotReassignValuesToImmutableMember),
						"has input = BaseClever(3)", "Run", "\tinput.Compute", "\tinput = BaseClever(5)"));
				innerDummy.ParseMembersAndMethods(parser).Methods[0].GetBodyAndParseIfNeeded();
			},
			Throws.InstanceOf<Body.ValueIsNotMutableAndCannotBeChanged>());
	}

	[Test]
	public void ModifyMutableMemberValueUsingTypeInstance()
	{
		using var dummy = new Type(type.Package,
			new TypeLines("Clever", "mutable Number", "Compute Number", "\t5 + Number"));
		dummy.ParseMembersAndMethods(parser);
		var cleverConsumerType = new Type(type.Package,
			new TypeLines(nameof(ModifyMutableMemberValueUsingTypeInstance), "has clever = Clever(3)",
				"Run", "\tclever.Compute is 8", "\tclever.Number = 5"));
		cleverConsumerType.ParseMembersAndMethods(parser).Methods[0].GetBodyAndParseIfNeeded();
		Assert.That(type.GetType("Clever").Members[0].InitialValue, Is.Null);
	}

	[Test]
	public void ModifyMutableMembersMultipleTimes()
	{
		using var computer = new Type(type.Package,
			new TypeLines("Computer", "mutable Number", "Compute Number", "\t5 + Number"));
		computer.ParseMembersAndMethods(parser);
		using var cleverConsumerType = new Type(type.Package,
			new TypeLines(nameof(ModifyMutableMembersMultipleTimes), "has computer = Computer(3)",
				"Run", "\tconstant bla = 5", "\tmutable blub = Compute", "\tconstant number = bla + 1",
				"\tmutable swappedBlub = blub", "\tblub = 49", "\tmutable temporary = swappedBlub",
				"\tswappedBlub = 50", "\ttemporary is 9", "\ttemporary is 10", "Compute Number",
				"\tcomputer.Number.Increment", "\tcomputer.Compute"));
		cleverConsumerType.ParseMembersAndMethods(parser);
		var body = (Body)cleverConsumerType.Methods[0].GetBodyAndParseIfNeeded();
		Assert.That(body.FindVariable("blub")?.InitialValue.ToString(), Is.EqualTo("Compute"));
		Assert.That(body.FindVariable("swappedBlub")?.InitialValue.ToString(), Is.EqualTo("blub"));
		Assert.That(body.FindVariable("temporary")?.InitialValue.ToString(),
			Is.EqualTo("swappedBlub"));
	}

	[Test]
	public void NewExpressionDoesNotMatchMemberType()
	{
		using var dummyType = new Type(type.Package,
			new TypeLines("Dummy", "mutable Number", "Run", "\tNumber = 3", "\tNumber = 5"));
		dummyType.ParseMembersAndMethods(parser);
		using var badType = new Type(type.Package,
			new TypeLines(nameof(NewExpressionDoesNotMatchMemberType), "mutable Number",
				"Compute Number", "\tNumber = \"Hi\""));
		badType.ParseMembersAndMethods(parser);
		Assert.That(() => badType.Methods[0].GetBodyAndParseIfNeeded(),
			Throws.InstanceOf<MutableReassignment.ValueTypeNotMatchingWithAssignmentType>());
		Assert.That(
			() => badType.Members[0].CheckIfWeCouldUpdateValue(new Text(badType, "Hi"),
				(Body)dummyType.Methods[0].GetBodyAndParseIfNeeded()),
			Throws.InstanceOf<Member.NewExpressionDoesNotMatchMemberType>());
	}

	[Test]
	public void CannotReassignNonMutableMember()
	{
		using var dummyType =
			new Type(type.Package,
				new TypeLines("DummyAgain", "mutable Number", "Run",
					"\tNumber = 3", "\tNumber = 5"));
		dummyType.ParseMembersAndMethods(parser);
		using var badType = new Type(type.Package,
			new TypeLines(nameof(CannotReassignNonMutableMember), "constant something = 7",
				"Compute Number", "\tsomething = 3"));
		badType.ParseMembersAndMethods(parser);
		Assert.That(
			() => badType.Members[0].CheckIfWeCouldUpdateValue(new Number(badType, 9),
				(Body)dummyType.Methods[0].GetBodyAndParseIfNeeded()),
			Throws.InstanceOf<Body.ValueIsNotMutableAndCannotBeChanged>());
	}

	[Test]
	public void NotAllowedToReassignMethodCall() =>
		Assert.That(
			() =>
			{
				using var dummy = new Type(type.Package,
					new TypeLines(nameof(NotAllowedToReassignMethodCall), "mutable Number",
						"MutableCall Mutable(Number)", "\tMutableCall = Number", "\tNumber = 5"));
				dummy.ParseMembersAndMethods(parser).Methods[0].GetBodyAndParseIfNeeded();
			},
			Throws.InstanceOf<Body.IdentifierNotFound>());
}