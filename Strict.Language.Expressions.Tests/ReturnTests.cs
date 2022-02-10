using NUnit.Framework;

namespace Strict.Language.Expressions.Tests;

public sealed class ReturnTests : TestExpressions
{
	[Test]
	public void ParseIncompleteReturn() =>
		Assert.That(() => ParseExpression("return"), Throws.InstanceOf<Return.MissingExpression>());

	[Test]
	public void ParseReturnNumber() =>
		ParseAndCheckOutputMatchesInput("return 33", new Return(new Number(method, 33)));

	[Test]
	public void ReturnGetHashCode()
	{
		var returnExpression = (Return)ParseExpression("return 1");
		Assert.That(returnExpression.GetHashCode(), Is.EqualTo(returnExpression.Value.GetHashCode()));
	}
}