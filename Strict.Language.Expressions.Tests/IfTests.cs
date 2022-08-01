using System;
using System.Linq;
using NUnit.Framework;

namespace Strict.Language.Expressions.Tests;

public sealed class IfTests : TestExpressions
{
	[Test]
	public void MissingCondition() =>
		Assert.That(() => ParseExpression("if"),
			Throws.InstanceOf<If.MissingCondition>().With.Message.
				Contains(@"TestPackage\dummy.strict:line 2"));

	[Test]
	public void InvalidCondition() =>
		Assert.That(() => ParseExpression("if 5", "\treturn 0"),
			Throws.InstanceOf<If.InvalidCondition>());

	[Test]
	public void ReturnTypeOfThenAndElseMustNotBeAny() =>
		Assert.That(() => ParseExpression("if 5 is 6", "\treturn 8", "else", "\treturn \"hello\"").ReturnType,
			Throws.InstanceOf<If.ReturnTypeOfThenAndElseMustHaveMatchingType>());

	[Test]
	public void ReturnTypeOfThenAndElseIsNumberAndCountIsValid() =>
		Assert.That(ParseExpression("if bla is 5", "\treturn Count(0)", "else", "\treturn 5").ReturnType,
			Is.EqualTo(type.GetType(Base.Number)));

	[Test]
	public void ReturnTypeOfThenAndElseIsCountAndCharacterIsValid() =>
		Assert.That(ParseExpression("if bla is 5", "\treturn Count(0)", "else", "\treturn Character(5)").ReturnType,
			Is.EqualTo(type.GetType(Base.Number)));

	[Test]
	public void ParseInvalidSpaceAfterElseIsNotAllowed() =>
		Assert.That(() => ParseExpression("else "), Throws.InstanceOf<Type.ExtraWhitespacesFoundAtEndOfLine>());

	[Test]
	public void ParseJustElseIsNotAllowed() =>
		Assert.That(() => ParseExpression("else"),
			Throws.InstanceOf<If.UnexpectedElse>().With.Message.
				Contains(@"at TestPackage.dummy.Run in "));

	[Test]
	public void ParseIncompleteThen() =>
		Assert.That(() => ParseExpression("if bla is 5"), Throws.InstanceOf<If.MissingThen>());

	[Test]
	public void ParseWrongIndentation() =>
		Assert.That(() => ParseExpression("if bla is 5", "Run"),
			Throws.InstanceOf<Method.InvalidIndentation>());

	[Test]
	public void ParseIf() =>
		Assert.That(ParseExpression("if bla is 5", "\tlog.Write(\"Hey\")"),
			Is.EqualTo(new If(GetCondition(), GetThen())));

	[Test]
	public void ParseMissingElseExpression() =>
		Assert.That(() => ParseExpression("if bla is 5", "\tRun", "else"),
			Throws.InstanceOf<If.UnexpectedElse>().With.Message.
				Contains(@"TestPackage\dummy.strict:line 4"));

	[Test]
	public void ParseIfElse() =>
		Assert.That(ParseExpression("if bla is 5", "\tlog.Write(\"Hey\")", "else", "\tRun"),
			Is.EqualTo(new If(GetCondition(), GetThen(), new MethodCall(method))).And.Not.
				EqualTo(new If(GetCondition(), GetThen())));

	private MethodCall GetThen() =>
		new(member.Type.Methods[0], new MemberCall(null, member), new Text(type, "Hey"));

	private Binary GetCondition() =>
		new(new MemberCall(null, bla), boolean.FindMethod(BinaryOperator.Is)!, number);

	[Test]
	public void ReturnGetHashCode()
	{
		var ifExpression = (If)ParseExpression("if bla is 5", "\tRun");
		Assert.That(ifExpression.GetHashCode(),
			Is.EqualTo(ifExpression.Condition.GetHashCode() ^ ifExpression.Then.GetHashCode()));
	}
}