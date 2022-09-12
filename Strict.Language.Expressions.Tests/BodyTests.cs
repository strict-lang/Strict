using NUnit.Framework;

namespace Strict.Language.Expressions.Tests;

public sealed class BodyTests : TestExpressions
{
	/*TODO: these tests make close to no sense now, instead look at the uncovered lines and add tests for those!
	[Test]
	public void ReturnAsLastExpressionIsNotNeeded() =>
		Assert.That(() => new Body(method).ParseExpressions(
			new Expression[]
			{
				new Assignment(new Body(method), "num", new Number(method, 5)),
				new Return(new Number(method, 5))
			} /*,
				new[]
				{
					new Method.Line(method, 1, "let num = 5", 1),
					new Method.Line(method, 1, "return num", 2)
				}*), Throws.InstanceOf<ReturnAsLastExpressionIsNotNeeded>());

	[Test]
	public void BodyToString()
	{
		var body = new Body(method);
		body.SetAndValidateExpressions(
			new Expression[]
			{
				new Assignment(new Body(method), "num", new Number(method, 5)), new Number(method, 5)
			} /*,
			new[]
			{
				new Method.Line(method, 1, "let num = 5", 1), new Method.Line(method, 1, "num", 2)
			}*);
		Assert.That(body.ToString(),
			Is.EqualTo(body.Expressions[0] + Environment.NewLine + body.Expressions[1]));
	}
	*/
	[Test]
	public void FindVariableValue() =>
		Assert.That(
			new Body(method).AddVariable("num", new Number(method, 5)).FindVariableValue("num"),
			Is.EqualTo(new Number(method, 5)));

	[Test]
	public void FindParentVariableValue() =>
		Assert.That(
			new Body(method, 0, new Body(method).AddVariable("str", new Text(method, "Hello"))).
				AddVariable("num", new Number(method, 5)).FindVariableValue("str"),
			Is.EqualTo(new Text(method, "Hello")));

	[Test]
	public void CannotUseVariableFromLowerScope() =>
		Assert.That(() => ParseExpression("if bla is 5", "\tlet abc = \"abc\"", "log.Write(abc)"),
			Throws.InstanceOf<IdentifierNotFound>().With.Message.StartWith("abc"));

	[Test]
	public void UnknownVariable() =>
		Assert.That(() => ParseExpression("if bla is 5", "\tlog.Write(unknownVariable)"),
			Throws.InstanceOf<IdentifierNotFound>().With.Message.StartWith("unknownVariable"));

	[Test]
	public void IfHasDifferentScopeThanMethod() =>
		Assert.That(ParseExpression("if bla is 5", "\tlet abc = \"abc\"", "\tlog.Write(abc)"),
			Is.EqualTo(new If(GetCondition(), CreateThenBlock())));

	private Expression CreateThenBlock()
	{
		var body = new Body(method);
		var expressions = new Expression[2];
		expressions[0] = new Assignment(body, "abc", new Text(method, "abc"));
		var arguments = new Expression[] { new VariableCall("abc", body.FindVariableValue("abc")!) };
		expressions[1] = new MethodCall(member.Type.GetMethod("Write", arguments),
			new MemberCall(null, member), arguments);
		body.SetExpressions(expressions);
		return body;
	}

	[Test]
	public void IfAndElseHaveTheirOwnScopes() =>
		Assert.That(() => ParseExpression(
				// @formatter:off
				"if bla is 5",
				"\tlet ifText = \"in if\"",
				"\tlog.Write(ifText)",
				"else",
				"\tlog.Write(ifText)"),
				// @formatter:on
			Throws.InstanceOf<IdentifierNotFound>().With.Message.StartWith("ifText"));

	[Test]
	public void MissingThenDueToIncorrectChildBodyStart() =>
		Assert.That(() => ParseExpression(
				"if bla is 5",
				"let abc = \"abc\"",
				"\tlog.Write(abc)"),
			Throws.InstanceOf<If.MissingThen>());

	[Test]
	public void EmptyInputIsNotAllowed() =>
		Assert.That(() => new Body(method).Parse(),
			Throws.InstanceOf<SpanExtensions.EmptyInputIsNotAllowed>());

	[Test]
	public void CheckVariableCallCurrentValue()
	{
		var ifExpression = ParseExpression(
			"if bla is 5",
			"\tlet abc = \"abc\"",
			"\tlog.Write(abc)") as If;
		var variableCall =
			((ifExpression?.Then as Body)?.Expressions[1] as MethodCall)?.Arguments[0] as VariableCall;
		Assert.That(variableCall?.CurrentValue.ToString(), Is.EqualTo("\"abc\""));
	}

	[Test]
	public void ReturnCorrectValueForSameNameVariableCallBasedOnScope()
	{
		var parentIf = (If)ParseExpression(
			"if bla is 5",
			"\tlet abc = \"abc\"",
			"\tif bla is 5.0",
			"\t\tlet abc = 5",
			"\t\tlog.Write(abc)",
			"\tlog.Write(abc)");
		var childIf = (If)((Body)parentIf.Then).Expressions[1];
		var childVariableCall =
			(VariableCall)((MethodCall)((Body)childIf.Then).Expressions[1]).Arguments[0];
		Assert.That(childVariableCall.CurrentValue.ToString(), Is.EqualTo("5"));
		var parentVariableCall =
			(VariableCall)((MethodCall)((Body)parentIf.Then).Expressions[2]).Arguments[0];
		Assert.That(parentVariableCall.CurrentValue.ToString(), Is.EqualTo("\"abc\""));
	}
}