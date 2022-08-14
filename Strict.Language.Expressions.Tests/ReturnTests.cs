using NUnit.Framework;

namespace Strict.Language.Expressions.Tests;

public sealed class ReturnTests : TestExpressions
{
	[Test]
	public void ParseIncompleteReturn() =>
		Assert.That(() => ParseExpression("return"), Throws.InstanceOf<Return.MissingExpression>());

	[Test]
	public void ReturnAsLastExpressionIsNotNeeded() =>
		Assert.That(() => ParseExpression("return 1"),
			Throws.InstanceOf<Body.ReturnAsLastExpressionIsNotNeeded>());

	[Test]
	public void ParseReturnNumber() =>
		ParseAndCheckOutputMatchesInput(new[] { "if true", "\treturn 33", "0" },
			new If(new Boolean(method, true), new Return(new Number(method, 33))));

	[Test]
	public void ReturnGetHashCode()
	{
		var returnExpression = new Return(new Number(method, 33));
		Assert.That(returnExpression.GetHashCode(), Is.EqualTo(returnExpression.Value.GetHashCode()));
	}
}