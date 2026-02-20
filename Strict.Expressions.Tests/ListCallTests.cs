namespace Strict.Expressions.Tests;

public sealed class ListCallTests : TestExpressions
{
	[TestCase("constant numbers = (1, 2, 3)", "numbers(0)")]
	[TestCase("constant texts = (\"something\", \"someOtherThing\")", "texts(1)")]
	public void ListCallToString(params string[] lines) =>
		Assert.That(ParseExpression(lines).ToString(),
			Is.EqualTo(string.Join(Environment.NewLine, lines)));
}
