using NUnit.Framework;

namespace Strict.Language.Expressions.Tests
{
	public class ReturnTests : TestExpressions
	{
		[Test]
		public void ParseReturn() =>
			ParseAndCheckOutputMatchesInput("return true", new Return(new Boolean(method, true)));
	}
}