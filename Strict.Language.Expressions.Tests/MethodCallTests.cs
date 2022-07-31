using NUnit.Framework;

namespace Strict.Language.Expressions.Tests;

public sealed class MethodCallTests : TestExpressions
{
	[Test]
	public void ParseLocalMethodCall() =>
		ParseAndCheckOutputMatchesInput("Run", new NoArgumentMethodCall(type.Methods[0]));

	[Test]
	public void ParseCallWithArgument() =>
		ParseAndCheckOutputMatchesInput("log.Write(bla)",
			new OneArgumentMethodCall(member.Type.Methods[0], new MemberCall(member),
				new MemberCall(bla)));

	[Test]
	public void ParseCallWithTextArgument() =>
		ParseAndCheckOutputMatchesInput("log.Write(\"Hi\")",
			new OneArgumentMethodCall(member.Type.Methods[0], new MemberCall(member),
				new Text(type, "Hi")));

	[Test]
	public void ParseWithMissingArgument() =>
		Assert.That(() => ParseExpression("log.Write"),
			Throws.InstanceOf<ArgumentsDoNotMatchMethodParameters>().With.Message.
				StartsWith(
					"No arguments does not match \"TestPackage.Log.Write\" method parameters: (text TestPackage.Text)"));

	[Test]
	public void ParseWithTooManyArguments() =>
		Assert.That(() => ParseExpression("log.Write(1, 2)"),
			Throws.InstanceOf<ArgumentsDoNotMatchMethodParameters>().With.Message.
				StartsWith(
					"Arguments: (1, 2) do not match \"TestPackage.Log.Write\" method parameters: (text TestPackage.Text)"));

	[Test]
	public void ParseWithInvalidExpressionArguments() =>
		Assert.That(() => ParseExpression("log.Write(0g9y53)"),
			Throws.InstanceOf<InvalidExpressionForArgument>().With.Message.
				StartsWith("0g9y53 for log.Write argument 0"));

	[Test]
	public void ParseUnknownMethod() =>
		Assert.That(() => ParseExpression("log.NotExisting()"),
			Throws.InstanceOf<MethodNotFound>());

	[Test]
	public void ParseCallWithUnknownArgument() =>
		Assert.That(() => ParseExpression("log.Write(unknown)"), Throws.InstanceOf<MemberNotFound>());

	[Test]
	public void ParseCallWithUnknownMemberCallArgument() =>
		Assert.That(() => ParseExpression("log.Write(log.unknown)"),
			Throws.InstanceOf<MemberNotFound>().With.Message.StartsWith("unknown in TestPackage.Log"));

	[Test]
	public void MethodCallMembersMustBeWords() =>
		Assert.That(() => ParseExpression("0g9y53.Write()"), Throws.InstanceOf<MemberNotFound>());

	[Test]
	public void FromMethodCall()
	{
		var characterType = type.GetType(Base.Character);
		Assert.That(ParseExpression("Character(7)"),
			Is.EqualTo(new OneArgumentMethodCall(characterType.FindMethod(Method.From)!,
				new From(characterType), new Number(type, 7))));
	}
}