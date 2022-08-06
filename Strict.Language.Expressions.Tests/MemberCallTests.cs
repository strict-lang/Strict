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
		Assert.That(() => ParseExpression("unknown"), Throws.InstanceOf<UnknownExpression>());

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
}