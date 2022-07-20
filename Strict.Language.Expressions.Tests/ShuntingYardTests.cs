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
	[TestCase("\"Hello\" + \"World\"", "\"Hello\", \"World\", +")]
	[TestCase("(\"Hello\" + \"World\") - \"H\"", "\"Hello\", \"World\", +, \"H\", -")]
	[TestCase("\"Hello + 5\" + \"World\"", "\"Hello + 5\", \"World\", +")]
	[TestCase("(a, b) + (c, d)", "(a, b), (c, d), +")]
	[TestCase("((a, b) + (c, d)) * 2", "(a, b), (c, d), +, 2, *")]
	[TestCase("((a, (b * e)) + (c, d)) * 2", "(a, (b * e)), (c, d), +, 2, *")]
	[TestCase("(\"Hello\", \"World\")", "(\"Hello\", \"World\")")]
	public void Parse(string input, string expected) =>
		Assert.That(string.Join(", ", new ShuntingYard(input).Output.Reverse()),
			Is.EqualTo(expected));
}