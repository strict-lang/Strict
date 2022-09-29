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
	public void ReturnTypeMustExistIfMethodReturnsSomething() =>
		Assert.That(
			() => new Method(type, 0, this, new[] { "ReturnNumber", "\t5" }).GetBodyAndParseIfNeeded(),
			Throws.InstanceOf<Body.ChildBodyReturnTypeMustMatchMethod>());

	[Test]
	public void ReturnTypeMustMatchWhateverMethodIsReturning() =>
		Assert.That(
			() => new Method(type, 0, this, new[] { "ReturnNumber Text", "\t5" }).GetBodyAndParseIfNeeded(),
			Throws.InstanceOf<Body.ChildBodyReturnTypeMustMatchMethod>());

	[Test]
	public void ParseReturnNumber() =>
		Assert.That(new Method(type, 0, this,
				new[]
				{
					"ReturnParse Number",
					"	if true",
					"		return 33",
					"	0"
				}).GetBodyAndParseIfNeeded(),
			Is.EqualTo(new Body(method).SetExpressions(new Expression[]
			{
				new If(new Boolean(method, true), new Return(new Number(method, 33))),
				new Number(method, 0)
			})));

	[Test]
	public void ReturnGetHashCode()
	{
		var returnExpression = new Return(new Number(method, 33));
		Assert.That(returnExpression.GetHashCode(), Is.EqualTo(returnExpression.Value.GetHashCode()));
	}
}