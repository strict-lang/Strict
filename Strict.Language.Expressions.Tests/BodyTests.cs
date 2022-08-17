using NUnit.Framework;
using static Strict.Language.Body;

namespace Strict.Language.Expressions.Tests;

public sealed class BodyTests : TestExpressions
{
	[Test]
	public void SetExpressionWithReturnAtLast() =>
		Assert.That(
			() => new Body(method).SetAndValidateExpressions(GetFaultyExpressions(),
				new[]
				{
					new Method.Line(method, 1, "let num = 5", 1),
					new Method.Line(method, 1, "return num", 2)
				}), Throws.InstanceOf<ReturnAsLastExpressionIsNotNeeded>());

	private Expression[] GetFaultyExpressions() =>
		new Expression[]
		{
			new Assignment(new Body(method), "num", new Number(method, 5)),
			new Return(new Number(method, 5))
		};

	[Test]
	public void FindVariableValue() =>
		Assert.That(
			new Body(method).AddVariable("num", new Number(method, 5)).FindVariableValue("num"),
			Is.EqualTo(new Number(method, 5)));

	[Test]
	public void FindParentVariableValue() =>
		Assert.That(
			new Body(method, new Body(method).AddVariable("str", new Text(method, "Hello"))).
				AddVariable("num", new Number(method, 5)).FindVariableValue("str"),
			Is.EqualTo(new Text(method, "Hello")));
}