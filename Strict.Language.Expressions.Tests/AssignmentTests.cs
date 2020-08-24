using NUnit.Framework;

namespace Strict.Language.Expressions.Tests
{
	public class AssignmentTests : TestExpressions
	{
		[Test]
		public void ParseNumber()
		{
			var input = "let number = 5";
			var expected =
				new Assignment(new Expressions.Identifier(nameof(number), number.ReturnType), number);
			Assert.That(ParseExpression(method, input), Is.EqualTo(expected));
		}
	}
}