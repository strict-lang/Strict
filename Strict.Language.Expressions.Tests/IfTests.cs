using System.Linq;
using NUnit.Framework;

namespace Strict.Language.Expressions.Tests;

public sealed class IfTests : TestExpressions
{
	[Test]
	public void ParseIncompleteIf() =>
		Assert.That(() => ParseExpression("if"), Throws.InstanceOf<If.MissingCondition>());

	[Test]
	public void ParseIncompleteThen() =>
		Assert.That(() => ParseExpression("if bla is 5"), Throws.InstanceOf<If.MissingThen>());

	[Test]
	public void ParseWrongIndentation() =>
		Assert.That(() => ParseExpression("if bla is 5", "Run"),
			Throws.InstanceOf<Method.InvalidIndentation>());

	[Test]
	public void ParseIf()
	{
		var expression = ParseExpression("if bla is 5", "\tlog.Write(\"Hey\")");
		Assert.That(expression,
			Is.EqualTo(new If(
				new Binary(new MemberCall(bla), binaryOperators.First(m => m.Name == BinaryOperator.Is),
					number),
				new MethodCall(new MemberCall(member), member.Type.Methods[0], new Text(type, "Hey")),
				null)));
	}

	[Test]
	public void ParseIfElse()
	{
		var expression = ParseExpression("if bla is 5", "\tlog.Write(\"Hey\")", "else", "\tRun");
		Assert.That(expression,
			Is.EqualTo(new If(
				new Binary(new MemberCall(bla), binaryOperators.First(m => m.Name == BinaryOperator.Is),
					number),
				new MethodCall(new MemberCall(member), member.Type.Methods[0], new Text(type, "Hey")),
				null)));
	}

	[Test]
	public void ReturnGetHashCode()
	{
		var ifExpression = (If)ParseExpression("if bla is 5", "\tRun");
		Assert.That(ifExpression.GetHashCode(),
			Is.EqualTo(ifExpression.Condition.GetHashCode() ^ ifExpression.Then.GetHashCode()));
	}
}