using NUnit.Framework;

namespace Strict.Language.Expressions.Tests;

public sealed class MethodCallTests : TestExpressions
{
	[Test]
	public void ParseLocalMethodCall() =>
		ParseAndCheckOutputMatchesInput("Run", new MethodCall(null, type.Methods[0]));

	[Test]
	public void ParseRemoteMethodCall() =>
		ParseAndCheckOutputMatchesInput("log.Write(\"Hi\")",
			new MethodCall(new MemberCall(member), member.Type.Methods[0], new Text(type, "Hi")));

	[Test]
	public void ParseCallWithArgument() =>
		ParseAndCheckOutputMatchesInput("log.Write(bla)",
			new MethodCall(new MemberCall(member), member.Type.Methods[0], new MemberCall(bla)));

	[Test]
	public void ParseWithMissingArgument() =>
		Assert.That(() => ParseExpression("log.Write"),
			Throws.InstanceOf<MethodCall.ArgumentsDoNotMatchMethodParameters>().With.Message.StartsWith(
				"No arguments does not match TestPackage.Log.Write(text TestPackage.Text)"));

	[Test]
	public void ParseWithTooManyArguments() =>
		Assert.That(() => ParseExpression("log.Write(1, 2)"),
			Throws.InstanceOf<MethodCall.ArgumentsDoNotMatchMethodParameters>().With.Message.StartsWith(
				"Arguments: (1, 2) do not match TestPackage.Log.Write(text TestPackage.Text)"));

	[Test]
	public void ParseWithInvalidExpressionArguments() =>
		Assert.That(() => ParseExpression("log.Write(0g9y53)"),
			Throws.InstanceOf<MethodCall.InvalidExpressionForArgument>().With.Message.StartsWith(
				"0g9y53 for log.Write argument 0"));

	[Test]
	public void ParseUnknownMethod() =>
		Assert.That(() => ParseExpression("log.NotExisting()"),
			Throws.InstanceOf<MethodCall.MethodNotFound>());

	[Test]
	public void ParseCallWithUnknownArgument() =>
		Assert.That(() => ParseExpression("log.Write(unknown)"),
			Throws.InstanceOf<MemberCall.MemberNotFound>());

	[Test]
	public void ParseCallWithUnknownMemberCallArgument() =>
		Assert.That(() => ParseExpression("log.Write(log.unknown)"),
			Throws.InstanceOf<MemberCall.MemberNotFound>().With.Message.StartsWith("unknown in TestPackage.Log"));

	[Test]
	public void MethodCallMembersMustBeWords() =>
		Assert.That(() => ParseExpression("0g9y53.Write()"),
			Throws.InstanceOf<MemberCall.MemberNotFound>());

	[Test]
	public void CallMethodWithTwoArguments() =>
		ParseAndCheckOutputMatchesInput("bla.Clamp(0, 10)",
			new MethodCall(new MemberCall(bla), bla.Type.Methods[1], new Number(type, 0),
				new Number(type, 10)));

	[Test]
	public void CallMethodWithThreeArguments() =>
		Assert.That(() => ParseExpression("bla.Clamp(0, 10, 15)"),
			Throws.InstanceOf<MethodCall.ArgumentsDoNotMatchMethodParameters>());
}