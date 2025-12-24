namespace Strict.Expressions.Tests;

public sealed class MemberCallTests : TestExpressions
{
	[Test]
	public void UseKnownMember() =>
		Assert.That(ParseExpression("Type(\"Hello\").Name").ToString(),
			Is.EqualTo("Type(\"Hello\", \"TestPackage\").Name"));

	[Test]
	public void UnknownMember() =>
		Assert.That(() => ParseExpression("unknown"), Throws.InstanceOf<Body.IdentifierNotFound>());

	[Test]
	public void MembersMustBeWords() =>
		Assert.That(() => ParseExpression("0g9y53"), Throws.InstanceOf<UnknownExpression>());

	[Test]
	public void NestedMemberNotFound() =>
		Assert.That(() => ParseExpression("logger.unknown"),
			Throws.InstanceOf<MemberOrMethodNotFound>().With.Message.
				StartsWith("unknown in TestPackage.Log"));

	[Test]
	public void NumbersCanNotStartNestedCall() =>
		Assert.That(() => ParseExpression("1.logger"),
			Throws.InstanceOf<NumbersCanNotBeInNestedCalls>());

	[Test]
	public void OperatorsCannotBeInNestedCalls() =>
		Assert.That(() => ParseExpression("+.logger"), Throws.InstanceOf<InvalidOperatorHere>());

	[Test]
	public void MultipleWordsMemberNotFound() =>
		Assert.That(() => ParseExpression("directory.GetFiles"),
			Throws.InstanceOf<MemberOrMethodNotFound>());

	[Test]
	public void NestedMemberIsNotAWord() =>
		Assert.That(() => ParseExpression("logger.5"),
			Throws.InstanceOf<NumbersCanNotBeInNestedCalls>());

	[Test]
	public void ValidMemberCall() =>
		Assert.That(ParseExpression("Range(0, 5).Start").ToString(), Is.EqualTo("Range(0, 5).Start"));

	[Test]
	public void MemberWithArgumentsInitializerShouldNotHaveType() =>
		Assert.That(
			() =>
			{
				using var _ =
					new Type(type.Package, new TypeLines("Declaration", "has input Text = Text(5)")).
						ParseMembersAndMethods(parser);
			}, //ncrunch: no coverage
			Throws.InstanceOf<ParsingFailed>().With.InnerException.
				InstanceOf<NamedType.AssignmentWithInitializerTypeShouldNotHaveNameWithType>());

	private readonly ExpressionParser parser = new MethodExpressionParser();

	[Test]
	public void UnknownExpressionInMemberInitializer() =>
		Assert.That(
			() =>
			{
				using var _ = new Type(type.Package,
						new TypeLines(nameof(UnknownExpressionInMemberInitializer),
							"has input Text = random")).
					ParseMembersAndMethods(parser);
			}, //ncrunch: no coverage
			Throws.InstanceOf<ParsingFailed>());

	[Test]
	public void NameMustBeAWordWithoutAnySpecialCharacterOrNumber() =>
		Assert.That(
			() =>
			{
				using var _ = new Type(type.Package,
					new TypeLines(nameof(NameMustBeAWordWithoutAnySpecialCharacterOrNumber),
						"has input1$ = Text(5)")).ParseMembersAndMethods(parser);
			}, //ncrunch: no coverage
			Throws.InstanceOf<ParsingFailed>().With.InnerException.
				InstanceOf<Context.NameMustBeAWordWithoutAnySpecialCharactersOrNumbers>());

	[Test]
	public void MemberWithArgumentsInitializer()
	{
		using var assignmentType = new Type(type.Package,
			new TypeLines(nameof(MemberWithArgumentsInitializer), "has input = Character(5)",
				"GetInput Text", "\tinput")).ParseMembersAndMethods(parser);
		Assert.That(assignmentType.Members[0].Name, Is.EqualTo("input"));
		Assert.That(assignmentType.Members[0].IsPublic, Is.False);
		Assert.That(assignmentType.Members[0].Type, Is.EqualTo(type.GetType(Base.Character)));
		Assert.That(assignmentType.Members[0].InitialValue, Is.InstanceOf<MethodCall>());
		var methodCall = (MethodCall)assignmentType.Members[0].InitialValue!;
		Assert.That(methodCall.Method.ReturnType.Name, Is.EqualTo(Base.Character));
		Assert.That(methodCall.Arguments[0], Is.EqualTo(new Number(type, 5)));
	}

	[Test]
	public void MemberGetHashCodeAndEquals()
	{
		using var memberCall =
			new Type(type.Package,
				new TypeLines(nameof(MemberGetHashCodeAndEquals), "has input = Text(5)", "GetInput Text",
					"\tinput")).ParseMembersAndMethods(parser);
		Assert.That(memberCall.Members[0].GetHashCode(),
			Is.EqualTo(memberCall.Members[0].Name.GetHashCode()));
		Assert.That(
			memberCall.Members[0].
				Equals(new Member(memberCall, "input", memberCall)
				{
					InitialValue = new Text(memberCall, "5")
				}), Is.True);
	}

	[Test]
	public void MemberWithBinaryExpression()
	{
		// @formatter:off
		using var assignmentType =
			new Type(type.Package,
					new TypeLines(nameof(MemberWithBinaryExpression),
						"has combinedNumber = 3 + 5",
						"GetCombined Number",
						"\tcombinedNumber")).
				ParseMembersAndMethods(parser);
		Assert.That(assignmentType.Members[0].Name, Is.EqualTo("combinedNumber"));
		Assert.That(assignmentType.Members[0].Type, Is.EqualTo(type.GetType(Base.Number)));
		var binary = (Binary)assignmentType.Members[0].InitialValue!;
		Assert.That(binary.Instance, Is.EqualTo(new Number(type, 3)));
		Assert.That(binary.Arguments[0], Is.EqualTo(new Number(type, 5)));
	}

	[Test]
	public void FromConstructorCall()
	{
		using var program = new Type(type.Package,
			new TypeLines(nameof(FromConstructorCall),
				"has file = File(\"test.txt\")",
				"Run",
				"\tconstant a = 5")).ParseMembersAndMethods(parser);
		Assert.That(program.Members[0].InitialValue?.ToString(), Is.EqualTo("File(\"test.txt\")"));
		Assert.That(program.Members[0].InitialValue?.ReturnType.Name, Is.EqualTo("File"));
	}

	[Test]
	public void FromConstructorCallUsingMemberName()
	{
		using var program = new Type(type.Package,
			new TypeLines(nameof(FromConstructorCallUsingMemberName),
				"has file = \"test.txt\"",
				"Run",
				"\tconstant a = 5")).ParseMembersAndMethods(parser);
		Assert.That(program.Members[0].InitialValue?.ReturnType.Name, Is.EqualTo("File"));
	}

	[Test]
	public void MemberCallUsingAnotherMemberIsForbidden() =>
		Assert.That(
			() =>
			{
				using var _ = new Type(type.Package,
					new TypeLines(nameof(MemberCallUsingAnotherMemberIsForbidden),
					"has file = File(\"test.txt\")",
					"has fileDescription = file.Length > 1000 ? \"big file\" else \"small file\"",
					"Run",
					"\tconstant a = 5")).ParseMembersAndMethods(parser);
			}, //ncrunch: no coverage
			Throws.InstanceOf<CannotAccessMemberBeforeTypeIsParsed>());

	[Test]
	public void BaseTypeMemberCallInDerivedType()
	{
		using var program = new Type(type.Package,
			new TypeLines(nameof(BaseTypeMemberCallInDerivedType),
				"has Range",
				"Run",
				"\tlet result = Range.End + 5",
				"\tresult is Number")).ParseMembersAndMethods(parser);
		var body = (Body)program.Methods[0].GetBodyAndParseIfNeeded();
		var assignment = (Declaration)body.Expressions[0];
		Assert.That(((Binary)assignment.Value).Instance,Is.InstanceOf<MemberCall>());
		var comparison = (TypeComparison)((Binary)body.Expressions[1]).Arguments[0];
		Assert.That(comparison.IsConstant, Is.True);
	}

	[Test]
	public void DuplicateMembersAreNotAllowed() =>
		Assert.That(
			() =>
			{
				using var _ = new Type(type.Package, new TypeLines(nameof(DuplicateMembersAreNotAllowed),
					"has something Number",
					"has something Number",
					"Run",
					"\tconstant a = 5")).ParseMembersAndMethods(parser);
			}, //ncrunch: no coverage
			Throws.InstanceOf<TypeParser.DuplicateMembersAreNotAllowed>());

	[Test]
	public void MembersWithDifferentNamesAreAllowed()
	{
		using var program = new Type(type.Package,
			new TypeLines(nameof(MembersWithDifferentNamesAreAllowed),
				"has something Number",
				"has somethingDifferent Number",
				"Run",
				"\tconstant a = 5")).ParseMembersAndMethods(parser);
		Assert.That(program.Members[0].Name, Is.EqualTo("something"));
		Assert.That(program.Members[1].Name, Is.EqualTo("somethingDifferent"));
	}

	[Test]
	public void MemberNameWithDifferentTypeNamesThanOwnNotAllowed() =>
		Assert.That(
			() =>
			{
				using var _ = new Type(type.Package,
				new TypeLines(nameof(MemberNameWithDifferentTypeNamesThanOwnNotAllowed),
					"has numbers Boolean",
					"Run",
					"\t5")).ParseMembersAndMethods(parser);
			}, //ncrunch: no coverage
			Throws.InstanceOf<Member.MemberNameWithDifferentTypeNamesThanOwnAreNotAllowed>().With.
				Message.Contains("numbers"));

	[Test]
	public void VariableNameCannotHaveDifferentTypeNameThanValue() =>
		Assert.That(
			() =>
			{
				using var dummy = new Type(type.Package,
					new TypeLines(nameof(VariableNameCannotHaveDifferentTypeNameThanValue),
						"has text",
						"Run",
						"\tconstant numbers = \"5\"")).ParseMembersAndMethods(parser);
				dummy.Methods[0].GetBodyAndParseIfNeeded();
			}, //ncrunch: no coverage
			Throws.InstanceOf<Body.VariableNameCannotHaveDifferentTypeNameThanValue>().With.Message.
				Contains("Variable name numbers denotes different type than its value type Text. " +
					"Prefer using a different name"));

	[Test]
	public void CannotAccessMemberInSameTypeBeforeTypeIsParsed() =>
		Assert.That(
			() =>
			{
				using var _ = new Type(type.Package,
					new TypeLines(nameof(CannotAccessMemberInSameTypeBeforeTypeIsParsed),
						"has Range",
						"has something = Range(0, 13)",
						"Run",
						"\tconstant a = 5")).ParseMembersAndMethods(parser);
			}, //ncrunch: no coverage
			Throws.InstanceOf<CannotAccessMemberBeforeTypeIsParsed>());
}