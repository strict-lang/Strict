using System.Linq;
using NUnit.Framework;

namespace Strict.Language.Expressions.Tests;

public sealed class MemberCallTests : TestExpressions
{
	[SetUp]
	public void CreateParser() => parser = new MethodExpressionParser();

	private ExpressionParser parser = null!;

	[Test]
	public void UseKnownMember() =>
		Assert.That(ParseExpression("Type(\"Hello\").Name").ToString(), Is.EqualTo("Type(\"Hello\").Name"));

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
		Assert.That(() => ParseExpression("1.log"), Throws.InstanceOf<NumbersCanNotBeInNestedCalls>());

	[Test]
	public void OperatorsCannotBeInNestedCalls() =>
		Assert.That(() => ParseExpression("+.log"), Throws.InstanceOf<InvalidOperatorHere>());

	[Test]
	public void MultipleWordsMemberNotFound() =>
		Assert.That(() => ParseExpression("directory.GetFiles"),
			Throws.InstanceOf<MemberOrMethodNotFound>());

	[Test]
	public void NestedMemberIsNotAWord() =>
		Assert.That(() => ParseExpression("log.5"), Throws.InstanceOf<NumbersCanNotBeInNestedCalls>());

	[Test]
	public void ValidMemberCall() =>
		Assert.That(ParseExpression("Range(0, 5).Start").ToString(), Is.EqualTo("Range(0, 5).Start"));

	[Test]
	public void MemberWithArgumentsInitializerShouldNotHaveType() =>
		Assert.That(
			() => new Type(type.Package, new TypeLines("ConstantDeclaration", "has input Text = Text(5)")).
				ParseMembersAndMethods(parser),
			Throws.InstanceOf<ParsingFailed>().With.InnerException.
				InstanceOf<NamedType.AssignmentWithInitializerTypeShouldNotHaveNameWithType>());

	[Test]
	public void UnknownExpressionInMemberInitializer() =>
		Assert.That(
			() => new Type(type.Package,
					new TypeLines(nameof(UnknownExpressionInMemberInitializer), "has input Text = random")).
				ParseMembersAndMethods(parser), Throws.InstanceOf<ParsingFailed>()!);

	[Test]
	public void NameMustBeAWordWithoutAnySpecialCharactersOrNumbers() =>
		Assert.That(
			() => new Type(type.Package, new TypeLines(nameof(NameMustBeAWordWithoutAnySpecialCharactersOrNumbers), "has input1$ = Text(5)")).
				ParseMembersAndMethods(parser),
			Throws.InstanceOf<ParsingFailed>().With.InnerException.
				InstanceOf<Context.NameMustBeAWordWithoutAnySpecialCharactersOrNumbers>());

	[Test]
	public void MemberWithArgumentsInitializer()
	{
		var assignmentType =
			new Type(type.Package,
				new TypeLines(nameof(MemberWithArgumentsInitializer), "has input = Text(5)",
					"GetInput Text", "\tinput")).ParseMembersAndMethods(parser);
		Assert.That(assignmentType.Members[0].Name, Is.EqualTo("input"));
		Assert.That(assignmentType.Members[0].IsPublic, Is.False);
		Assert.That(assignmentType.Members[0].Type, Is.EqualTo(type.GetType(Base.Text)));
		Assert.That(assignmentType.Members[0].Value, Is.InstanceOf<MethodCall>());
		var methodCall = (MethodCall)assignmentType.Members[0].Value!;
		Assert.That(methodCall.Method.ReturnType.Name, Is.EqualTo(Base.Text));
		Assert.That(methodCall.Arguments[0], Is.EqualTo(new Number(type, 5)));
	}

	[Test]
	public void MemberGetHashCodeAndEquals()
	{
		var memberCall =
			new Type(type.Package,
				new TypeLines(nameof(MemberGetHashCodeAndEquals), "has input = Text(5)",
					"GetInput Text", "\tinput")).ParseMembersAndMethods(parser);
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
		var binary = (Binary)assignmentType.Members[0].Value!;
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
		Assert.That(program.Members[0].Value?.ToString(), Is.EqualTo("File(\"test.txt\")"));
		Assert.That(program.Members[0].Value?.ReturnType.Name, Is.EqualTo("File"));
	}

		[Test]
	public void FromConstructorCallUsingMemberName()
	{
		var program = new Type(type.Package,
			new TypeLines(nameof(FromConstructorCallUsingMemberName),
				"has file = \"test.txt\"",
				"Run",
				"\tconstant a = 5")).ParseMembersAndMethods(parser);
		Assert.That(program.Members[0].Value?.ReturnType.Name, Is.EqualTo("File"));
	}

	[Test]
	public void MemberCallUsingAnotherMember()
	{
		var program = new Type(type.Package,
			new TypeLines(nameof(MemberCallUsingAnotherMember),
				"has file = File(\"test.txt\")",
				"has fileDescription = file.Length > 1000 ? \"big file\" else \"small file\"",
				"Run",
				"\tconstant a = 5")).ParseMembersAndMethods(parser);
		Assert.That(program.Members[0].Name, Is.EqualTo("file"));
		Assert.That(program.Members[1].Name, Is.EqualTo("fileDescription"));
		Assert.That(program.Members[1].Value?.ToString(),
			Is.EqualTo("file.Length > 1000 ? \"big file\" else \"small file\""));
	}

	[Test]
	public void BaseTypeMemberCallInDerivedType()
	{
		var program = new Type(type.Package,
			new TypeLines(nameof(BaseTypeMemberCallInDerivedType),
				"has Range",
				"Run",
				"\tconstant a = Range.End + 5")).ParseMembersAndMethods(parser);
		var assignment = (ConstantDeclaration)program.Methods[0].GetBodyAndParseIfNeeded();
		Assert.That(((Binary)assignment.Value).Instance,
			Is.InstanceOf<MemberCall>());
	}

	[Test]
	public void DuplicateMembersAreNotAllowed() =>
		Assert.That(() => new Type(type.Package,
			new TypeLines(nameof(DuplicateMembersAreNotAllowed),
				"has something Number",
				"has something Number",
				"Run",
				"\tconstant a = 5")).ParseMembersAndMethods(parser),
			Throws.InstanceOf<Type.DuplicateMembersAreNotAllowed>());

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
}