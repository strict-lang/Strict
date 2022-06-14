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
		Assert.That(() => ParseExpression("unknown"), Throws.InstanceOf<MemberCall.MemberNotFound>());

	[Test]
	public void MembersMustBeWords() =>
		Assert.That(() => ParseExpression("0g9y53"), Throws.InstanceOf<UnknownExpression>());

	[TestCase("log.unknown")]
	[TestCase("1.log")]
	public void NestedMemberNotFound(string lines) =>
		Assert.That(() => ParseExpression(lines),
			Throws.InstanceOf<MemberCall.MemberNotFound>());

	[Test]
	public void MultipleWordsMemberNotFound() =>
		Assert.That(() => ParseExpression("directory.GetFiles"),
			Throws.InstanceOf<MemberCall.MemberNotFound>());

	[Test]
	public void NestedMemberIsNotAWord() =>
		Assert.That(() => ParseExpression("log.5"), Throws.InstanceOf<MemberCall.MemberNotFound>());
}