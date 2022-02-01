using NUnit.Framework;

namespace Strict.Language.Expressions.Tests;

public sealed class MethodCallTests : TestExpressions
{
	[Test]
	public void ParseCall() =>
		ParseAndCheckOutputMatchesInput("log.WriteLine(\"Hi\")",
			new MethodCall(new MemberCall(member), member.Type.Methods[0], new Text(type, "Hi")));

	[Test]
	public void ParseCallWithArgument() =>
		ParseAndCheckOutputMatchesInput("log.WriteLine(bla)",
			new MethodCall(new MemberCall(member), member.Type.Methods[0], new MemberCall(bla)));

	[Test]
	public void ParseCallWithUnknownArgument() =>
		Assert.That(() => ParseExpression(method, "log.WriteLine(unknown)"),
			Throws.InstanceOf<MemberCall.MemberNotFound>());

	[Test]
	public void ParseCallWithUnknownMemberCallArgument() =>
		Assert.That(() => ParseExpression(method, "log.WriteLine(log.unknown)"),
			Throws.InstanceOf<MemberCall.MemberNotFound>().With.Message.EqualTo("unknown in TestPackage.Log"));

	[Test]
	public void MethodCallMembersMustBeWords() =>
		Assert.That(() => ParseExpression(method, "0g9y53.WriteLine()"),
			Throws.InstanceOf<MemberCall.MemberNotFound>());
}