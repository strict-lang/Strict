using System.Linq;
using NUnit.Framework;

namespace Strict.Language.Expressions.Tests;

public sealed class MemberCallTests : TestExpressions
{
	[Test]
	public void UseKnownMember() =>
		ParseAndCheckOutputMatchesInput("log.Text",
			new MemberCall(new MemberCall(null, member), member.Type.Members.First(m => m.Name == "Text")));

	[Test]
	public void UnknownMember() =>
		Assert.That(() => ParseExpression("unknown"), Throws.InstanceOf<IdentifierNotFound>());

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
		Assert.That(ParseExpression("\"hello\".Characters"),
			Is.EqualTo(new MemberCall(new Text(method, "hello"),
				new Member(method.GetType(Base.Text), "Characters", null))));

	[Test]
	public void MemberWithArgumentsInitializerShouldNotHaveType() =>
		Assert.That(
			() => new Type(type.Package, new TypeLines("Assignment", "has input Text = Text(5)")).
				ParseMembersAndMethods(new MethodExpressionParser()),
			Throws.InstanceOf<ParsingFailed>().With.InnerException.
				InstanceOf<NamedType.AssignmentWithInitializerTypeShouldNotHaveNameWithType>());

	[Test]
	public void MemberWithArgumentsInitializer()
	{
		var assignmentType =
			new Type(type.Package,
				new TypeLines(nameof(MemberWithArgumentsInitializer), "has input = Text(5)",
					"GetInput Text", "\tinput")).ParseMembersAndMethods(new MethodExpressionParser());
		Assert.That(assignmentType.Members[0].Name, Is.EqualTo("input"));
		Assert.That(assignmentType.Members[0].Type, Is.EqualTo(type.GetType(Base.Text)));
		Assert.That(assignmentType.Members[0].Value, Is.InstanceOf<MethodCall>());
		var methodCall = (MethodCall)assignmentType.Members[0].Value!;
		Assert.That(methodCall.Instance!.ReturnType.Name, Is.EqualTo(Base.Text));
		Assert.That(methodCall.Arguments[0], Is.EqualTo(new Number(type, 5)));
	}

	[Test]
	public void MemberWithBinaryExpression()
	{
		var assignmentType =
			new Type(type.Package,
					new TypeLines(nameof(MemberWithBinaryExpression), "has combinedNumber = 3 + 5",
						"GetCombined Number", "\tcombinedNumber")).
				ParseMembersAndMethods(new MethodExpressionParser());
		Assert.That(assignmentType.Members[0].Name, Is.EqualTo("combinedNumber"));
		Assert.That(assignmentType.Members[0].Type, Is.EqualTo(type.GetType(Base.Number)));
		var binary = (Binary)assignmentType.Members[0].Value!;
		Assert.That(binary.Instance, Is.EqualTo(new Number(type, 3)));
		Assert.That(binary.Arguments[0], Is.EqualTo(new Number(type, 5)));
	}
}