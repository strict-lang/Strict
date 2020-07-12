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
	}
}