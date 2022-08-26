using System.Collections.Generic;
using NUnit.Framework;

namespace Strict.Language.Expressions.Tests;

public sealed class MethodCallTests : TestExpressions
{
	[SetUp]
	public void AddComplexMethods()
	{
		((List<Method>)type.Methods).Add(new Method(type, 0, this,
			new[] { "complexMethod(numbers, add Number) Number", "\t1" }));
		((List<Method>)type.Methods).Add(new Method(type, 0, this,
			new[] { "complexMethod(texts) Texts", "\t1" }));
	}

	[Test]
	public void ParseLocalMethodCall() =>
		ParseAndCheckOutputMatchesInput("Run", new MethodCall(type.Methods[0]));

	[Test]
	public void ParseCallWithArgument() =>
		ParseAndCheckOutputMatchesInput("log.Write(bla)",
			new MethodCall(member.Type.Methods[0], new MemberCall(null, member),
				new[] { new MemberCall(null, bla) }));

	[Test]
	public void ParseCallWithTextArgument() =>
		ParseAndCheckOutputMatchesInput("log.Write(\"Hi\")",
			new MethodCall(member.Type.Methods[0], new MemberCall(null, member),
				new[] { new Text(type, "Hi") }));

	[Test]
	public void ParseWithMissingArgument() =>
		Assert.That(() => ParseExpression("log.Write"),
			Throws.InstanceOf<Type.ArgumentsDoNotMatchMethodParameters>().With.Message.StartsWith(
				"No arguments does not match:\nWrite(text TestPackage.Text)\nWrite(number TestPackage.Number)"));

	[Test]
	public void ParseWithTooManyArguments() =>
		Assert.That(() => ParseExpression("log.Write(1, 2)"),
			Throws.InstanceOf<Type.ArgumentsDoNotMatchMethodParameters>().With.Message.
				StartsWith("Arguments: TestPackage.Number 1, TestPackage.Number 2 do not match"));

	[Test]
	public void ParseWithInvalidExpressionArguments() =>
		Assert.That(() => ParseExpression("log.Write(g9y53)"),
			Throws.InstanceOf<UnknownExpressionForArgument>().With.Message.
				StartsWith("g9y53 (argument 0)"));

	[Test]
	public void EmptyBracketsAreNotAllowed() =>
		Assert.That(() => ParseExpression("log.NotExisting()"),
			Throws.InstanceOf<List.EmptyListNotAllowed>());

	[Test]
	public void MethodNotFound() =>
		Assert.That(() => ParseExpression("log.NotExisting"),
			Throws.InstanceOf<MemberOrMethodNotFound>());

	[Test]
	public void ArgumentsDoNotMatchMethodParameters() =>
		Assert.That(() => ParseExpression("Character(\"Hi\")"),
			Throws.InstanceOf<Type.ArgumentsDoNotMatchMethodParameters>());

	[Test]
	public void ParseCallWithUnknownMemberCallArgument() =>
		Assert.That(() => ParseExpression("log.Write(log.unknown)"),
			Throws.InstanceOf<MemberOrMethodNotFound>().With.Message.
				StartsWith("unknown in TestPackage.Log"));

	[Test]
	public void MethodCallMembersMustBeWords() =>
		Assert.That(() => ParseExpression("g9y53.Write"), Throws.InstanceOf<MemberOrMethodNotFound>());

	[Test]
	public void UnknownExpressionForArgumentException() =>
		Assert.That(() => ParseExpression("complexMethod((\"1 + 5\" + \"5\"))"),
			Throws.InstanceOf<UnknownExpressionForArgument>().With.Message.
				StartsWith("+ is invalid for argument 0"));

	[Test]
	public void ListTokensAreNotSeparatedByCommaException() =>
		Assert.That(() => ParseExpression("complexMethod((\"1 + 5\" 5, \"5 + 5\"))"),
			Throws.InstanceOf<ListTokensAreNotSeparatedByComma>());

	[Test]
	public void SimpleFromMethodCall() =>
		Assert.That(ParseExpression("Character(7)"),
			Is.EqualTo(CreateFromMethodCall(type.GetType(Base.Character), new Number(type, 7))));

	[TestCase("Count(5)")]
	[TestCase("Count(5).Increment")]
	[TestCase("Count(5).Floor")]
	[TestCase("Range(0, 10)")]
	[TestCase("Range(0, 10).Length")]
	public void FromExample(string fromMethodCall) =>
		Assert.That(ParseExpression(fromMethodCall).ToString(), Is.EqualTo(fromMethodCall));

	[TestCase("complexMethod((1), 2)")]
	[TestCase("complexMethod((1, 2, 3) + (4, 5), 7)")]
	[TestCase("complexMethod((1, 2, 3) + (4, 5), complexMethod((1, 2, 3), 4))")]
	[TestCase("complexMethod((\"1 + 5\", \"5 + 5\"))")]
	public void FindRightMethodCall(string methodCall) =>
		Assert.That(ParseExpression(methodCall).ToString(), Is.EqualTo(methodCall));

	[Test]
	public void IsMethodPublic() =>
		Assert.That((ParseExpression("Run") as MethodCall)?.Method.IsPublic, Is.True);
}