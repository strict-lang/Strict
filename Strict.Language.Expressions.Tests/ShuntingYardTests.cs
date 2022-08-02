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
	[TestCase("1, 2, 3, 4, 5", "1, ,, 2, ,, 3, ,, 4, ,, 5")]
	[TestCase("\"Hi\", \" there\"", "\"Hi\", ,, \" there\"")]
	[TestCase("(1, 2, 3) + (3, 4), (4)", "(1, 2, 3), (3, 4), +, ,, (4)")]
	public void Parse(string input, string expected)
	{
		var tokens = new ShuntingYard(input, ..).Output.Reverse().Select(range => input[range]);
		Assert.That(string.Join(", ", tokens), Is.EqualTo(expected));
	}

	//TODO: check if we need more tests, maybe higher up at ParseExpression tests
	// 5 + 3
	// ^ first operand (left) Range
	//   ^ operator (range??????????)
	//     ^ right Range

	// ((1, 2), (3, 4)) + 1
	// ^list            ^op ^number
	// op, number, list (nested)

	// (first + second - 1 + 7 + 7, 3, 5, 6)
	// ^list (with a bunch of things, binary, numbers)

	// log.Write       ("Hi")
	// ^mem ^me        ^arguments (list)

	//(1, 2), (3, 4) -> list , list -> 3 tokens (middle one is NOT an operator)
	//first + second, 3, 5, 6, binary: + second first (then): , 3 , 5 , 6 -> 9 tokens (upside)
	//"Hi" -> 1 token -> Text
}