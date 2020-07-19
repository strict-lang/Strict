using NUnit.Framework;

namespace Strict.Language.Expressions.Tests
{
	public class ParsingTests : TestExpressions
	{
		[Test]
		public void ParseNumber()
		{
			var input = "5";//"let number = 5";
			var expected = new Number(method, 5);
			Assert.That(Parse(method, input), Is.EqualTo(expected));
		}
	}
}