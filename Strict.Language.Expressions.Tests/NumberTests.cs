using NUnit.Framework;

namespace Strict.Language.Expressions.Tests
{
	public class NumberTests : TestExpressions
	{
		[Test]
		public void ParseNumber() =>
			ParseAndCheckOutputMatchesInput("77", new Number(method, 77));

		[Test]
		public void TwoNumbersWithTheSameValueAreTheSame() =>
			Assert.That(new Number(method, 5), Is.EqualTo(new Number(method, 5)));
	}
}