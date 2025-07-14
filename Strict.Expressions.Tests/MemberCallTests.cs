using NUnit.Framework;
using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict.Expressions.Tests;

public sealed class MemberCallTests : TestExpressions
{
	[Test]
	public void UseKnownMember() =>
		Assert.That(ParseExpression("Type(\"Hello\").Name").ToString(),
			Is.EqualTo("Type(\"Hello\").Name"));

	[Test]
	public void UnknownMember() =>
		Assert.That(() => ParseExpression("unknown"), Throws.InstanceOf<Body.IdentifierNotFound>());

	[Test]
	public void MembersMustBeWords() =>
		Assert.That(() => ParseExpression("0g9y53"), Throws.InstanceOf<UnknownExpression>());

	[Test]
	public void NestedMemberNotFound() =>
		Assert.That(() => ParseExpression("log.unknown"),
			Throws.InstanceOf<MemberOrMethodNotFound>().With.Message.
				StartsWith("unknown in TestPackage.Log"));

	[Test]
	public void NumbersCanNotStartNestedCall() =>
		Assert.That(() => ParseExpression("1.log"),
			Throws.InstanceOf<NumbersCanNotBeInNestedCalls>());

	[Test]
	public void OperatorsCannotBeInNestedCalls() =>
		Assert.That(() => ParseExpression("+.log"), Throws.InstanceOf<InvalidOperatorHere>());

	[Test]
	public void MultipleWordsMemberNotFound() =>
		Assert.That(() => ParseExpression("directory.GetFiles"),
			Throws.InstanceOf<MemberOrMethodNotFound>());

	[Test]
	public void NestedMemberIsNotAWord() =>
		Assert.That(() => ParseExpression("log.5"),
			Throws.InstanceOf<NumbersCanNotBeInNestedCalls>());

	[Test]
	public void ValidMemberCall() =>
		Assert.That(ParseExpression("Range(0, 5).Start").ToString(), Is.EqualTo("Range(0, 5).Start"));

	[Test]
	public void MemberWithArgumentsInitializerShouldNotHaveType() =>
		Assert.That(
			() => new Type(type.Package,
					new TypeLines("ConstantDeclaration", "has input Text = Text(5)")).
				ParseMembersAndMethods(parser),
			Throws.InstanceOf<ParsingFailed>().With.InnerException.
				InstanceOf<NamedType.AssignmentWithInitializerTypeShouldNotHaveNameWithType>());

	private readonly ExpressionParser parser = new MethodExpressionParser();

	[Test]
	public void UnknownExpressionInMemberInitializer() =>
		Assert.That(
			() => new Type(type.Package,
					new TypeLines(nameof(UnknownExpressionInMemberInitializer), "has input Text = random")).
				ParseMembersAndMethods(parser), Throws.InstanceOf<ParsingFailed>());

	[Test]
	public void NameMustBeAWordWithoutAnySpecialCharacterOrNumber() =>
		Assert.That(
			() => new Type(type.Package,
				new TypeLines(nameof(NameMustBeAWordWithoutAnySpecialCharacterOrNumber),
					"has input1$ = Text(5)")).ParseMembersAndMethods(parser),
			Throws.InstanceOf<ParsingFailed>().With.InnerException.
				InstanceOf<Context.NameMustBeAWordWithoutAnySpecialCharactersOrNumbers>());

	[Test]
	public void MemberWithArgumentsInitializer()
	{
		var assignmentType =
			new Type(type.Package,
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
		var memberCall =
			new Type(type.Package,
				new TypeLines(nameof(MemberGetHashCodeAndEquals), "has input = Text(5)", "GetInput Text",
					"\tinput")).ParseMembersAndMethods(parser);
		Assert.That(memberCall.Members[0].GetHashCode(),
			Is.EqualTo(memberCall.Members[0].Name.GetHashCode()));
		Assert.That(
			memberCall.Members[0].Equals(new Member(memberCall, "input", new Text(memberCall, "5"))),
			Is.True);
	}

	[Test]
	public void MemberWithBinaryExpression()
	{
		// @formatter:off
		var assignmentType =
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
		var program = new Type(type.Package,
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
		var program = new Type(type.Package,
			new TypeLines(nameof(FromConstructorCallUsingMemberName),
				"has file = \"test.txt\"",
				"Run",
				"\tconstant a = 5")).ParseMembersAndMethods(parser);
		Assert.That(program.Members[0].InitialValue?.ReturnType.Name, Is.EqualTo("File"));
	}

	[Test]
	public void MemberCallUsingAnotherMemberIsForbidden() =>
		Assert.That(
			() => new Type(type.Package,
				new TypeLines(nameof(MemberCallUsingAnotherMemberIsForbidden),
					"has file = File(\"test.txt\")",
					"has fileDescription = file.Length > 1000 ? \"big file\" else \"small file\"",
					"Run",
					"\tconstant a = 5")).ParseMembersAndMethods(parser),
			Throws.InstanceOf<CannotAccessMemberBeforeTypeIsParsed>());

	[Test]
	public void BaseTypeMemberCallInDerivedType()
	{
		var program = new Type(type.Package,
			new TypeLines(nameof(BaseTypeMemberCallInDerivedType),
				"has Range",
				"Run",
				"\tconstant result = Range.End + 5")).ParseMembersAndMethods(parser);
		var assignment = (ConstantDeclaration)program.Methods[0].GetBodyAndParseIfNeeded();
		Assert.That(((Binary)assignment.Value).Instance,
			Is.InstanceOf<MemberCall>());
	}

	[Test]
	public void DuplicateMembersAreNotAllowed() =>
		Assert.That(
			() => new Type(type.Package,
				new TypeLines(nameof(DuplicateMembersAreNotAllowed),
					"has something Number",
					"has something Number",
					"Run",
					"\tconstant a = 5")).ParseMembersAndMethods(parser),
			Throws.InstanceOf<TypeParser.DuplicateMembersAreNotAllowed>());

	[Test]
	public void MembersWithDifferentNamesAreAllowed()
	{
		var program = new Type(type.Package,
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
			() => new Type(type.Package,
				new TypeLines(nameof(MemberNameWithDifferentTypeNamesThanOwnNotAllowed),
					"has numbers Boolean",
					"Run",
					"\t5")).ParseMembersAndMethods(parser),
			Throws.InstanceOf<Member.MemberNameWithDifferentTypeNamesThanOwnAreNotAllowed>().With.
				Message.Contains("numbers"));

	[Test]
	public void VariableNameCannotHaveDifferentTypeNameThanValue() =>
		Assert.That(
			() => new Type(type.Package,
					new TypeLines(nameof(VariableNameCannotHaveDifferentTypeNameThanValue),
						"has text",
						"Run",
						"\tconstant numbers = \"5\"")).ParseMembersAndMethods(parser).Methods[0].
				GetBodyAndParseIfNeeded(),
			Throws.InstanceOf<Body.VariableNameCannotHaveDifferentTypeNameThanValue>().With.Message.
				Contains("Variable name numbers denotes different type than its value type Text. " +
					"Prefer using a different name"));

	[Test]
	public void CannotAccessMemberInSameTypeBeforeTypeIsParsed() =>
		Assert.That(
			() => new Type(type.Package,
					new TypeLines(nameof(CannotAccessMemberInSameTypeBeforeTypeIsParsed),
						"has Range",
						"has something = Range(0, 13)",
						"Run",
						"\tconstant a = 5")).
				ParseMembersAndMethods(parser),
			Throws.InstanceOf<CannotAccessMemberBeforeTypeIsParsed>());
}