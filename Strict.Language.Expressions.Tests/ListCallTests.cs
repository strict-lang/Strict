using System;
using NUnit.Framework;

namespace Strict.Language.Expressions.Tests;

public sealed class ListCallTests : TestExpressions
{
	[TestCase("constant numbers = (1, 2, 3)", "numbers(0)")]
	[TestCase("constant something = \"something\"", "something.Characters(4)")]
	public void ListCallToString(params string[] lines) =>
		Assert.That(ParseExpression(lines).ToString(),
			Is.EqualTo(string.Join(Environment.NewLine, lines)));
}
