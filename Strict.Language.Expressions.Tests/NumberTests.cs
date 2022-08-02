using NUnit.Framework;

namespace Strict.Language.Expressions.Tests;

public sealed class NumberTests : TestExpressions
{
	[Test]
	public void NumberWithCharacters() => Assert.That(() => ParseExpression("7abc"), Throws.InstanceOf<UnknownExpression>());

	[Test]
	public void ParseNumber() => ParseAndCheckOutputMatchesInput("77", new Number(method, 77));

	[Test]
	public void TwoNumbersWithTheSameValueAreTheSame() =>
		Assert.That(new Number(method, 5), Is.EqualTo(new Number(method, 5)));

	[TestCase("7.05")]
	[TestCase("0.5")]
	[TestCase("77")]
	public void ValidNumbers(string input) => Assert.That(ParseExpression(input), Is.EqualTo(new Number(method, double.Parse(input))));
}