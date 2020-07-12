using NUnit.Framework;

namespace Strict.Language.Expressions.Tests
{
	public class TextTests : TestExpressions
	{
		[Test]
		public void ParseText() =>
			ParseAndCheckOutputMatchesInput("\"Hi\"", new Text(method, "Hi"));

		[Test]
		public void ParseIncompleteText() =>
			Assert.Throws<UnsupportedToken>(() => ParseAndCheckOutputMatchesInput("Text(", null));
	}
}