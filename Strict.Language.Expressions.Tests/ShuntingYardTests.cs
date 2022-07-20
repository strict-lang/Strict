using System.Linq;
using NUnit.Framework;

namespace Strict.Language.Expressions.Tests;

public sealed class ShuntingYardTests
{
	[TestCase("a", "a")]
	[TestCase("a + b", "a, b, +")]
	[TestCase("(2 + (3 + 5) * 5) * 2", "2, 3, 5, +, 5, *, +, 2, *")]
	[TestCase("a + b * c - d", "a, b, c, *, +, d, -")]
	[TestCase("(a + b) / 2", "a, b, +, 2, /")]
	[TestCase("2 * (a + b)", "2, a, b, +, *")]
	[TestCase("((a + b) / 2)", "a, b, +, 2, /")]
	[TestCase("(a + b) * (c + d)", "a, b, +, c, d, +, *")]
	[TestCase("(a * a + b) / (c + d * e)", "a, a, *, b, +, c, d, e, *, +, /")]
	public void GenerateTokens(string input, string expected) =>
		Assert.That(string.Join(", ", new ShuntingYard(input).Output.Reverse()),
			Is.EqualTo(expected));
}