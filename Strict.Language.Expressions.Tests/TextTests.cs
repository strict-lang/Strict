using NUnit.Framework;

namespace Strict.Language.Expressions.Tests;

public sealed class TextTests : TestExpressions
{
	[Test]
	public void ParseText() => ParseAndCheckOutputMatchesInput("\"Hi\"", new Text(method, "Hi"));
}