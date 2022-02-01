using System.Linq;
using NUnit.Framework;

namespace Strict.Language.Expressions.Tests;

public sealed class MemberCallTests : TestExpressions
{
	[Test]
	public void UseKnownMember() =>
		ParseAndCheckOutputMatchesInput("log.Text",
			new MemberCall(new MemberCall(member), member.Type.Members.First(m => m.Name == "Text")));

	[Test]
	public void MemberNotFound() =>
		Assert.That(() => ParseExpression(method, "unknown"),
			Throws.InstanceOf<MemberCall.MemberNotFound>());

	[Test]
	public void MembersMustBeWords() =>
		Assert.That(() => ParseExpression(method, "0g9y53"), Throws.InstanceOf<UnknownExpression>());

	[Test]
	public void NestedMemberNotFound() =>
		Assert.That(() => ParseExpression(method, "log.unknown"),
			Throws.InstanceOf<MemberCall.MemberNotFound>());

	[Test]
	public void NestedMemberIsNotAWord() =>
		Assert.That(() => ParseExpression(method, "log.5"),
			Throws.InstanceOf<MemberCall.MemberNotFound>());
}