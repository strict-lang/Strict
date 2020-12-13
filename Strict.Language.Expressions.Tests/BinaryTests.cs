using System.Linq;
using NUnit.Framework;

namespace Strict.Language.Expressions.Tests
{
	public class BinaryTests : TestExpressions
	{
		[Test]
		public void ParseBinary() =>
			ParseAndCheckOutputMatchesInput("5 + 5",
				new Binary(number, number.ReturnType.Methods.First(m => m.Name == "+"), number));

		[Test]
		public void MissingLeftExpression() =>
			Assert.That(() => ParseExpression(method, "bla + 5"),
				Throws.Exception.InstanceOf<Expression.InvalidExpression>());
		
		[Test]
		public void MissingRightExpression() =>
			Assert.That(() => ParseExpression(method, "5 + bla"),
				Throws.Exception.InstanceOf<Expression.InvalidExpression>());
	}
}