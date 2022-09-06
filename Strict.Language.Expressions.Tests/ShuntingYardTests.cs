﻿using System;
using System.Linq;
using NUnit.Framework;

namespace Strict.Language.Expressions.Tests;

public sealed class ShuntingYardTests
{
	[TestCase("abc", "abc")]
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
		var tokens = new ShuntingYard(input).Output.Reverse().Select(range => input[range]);
		Assert.That(string.Join(", ", tokens), Is.EqualTo(expected));
	}

	[Test]
	public void ParseIfWithIsNot()
	{
		var postfix = new ShuntingYard("if bla is not 25");
		Assert.That(postfix.Output.Pop(), Is.EqualTo(new Range(7, 13)));
		Assert.That(postfix.Output.Pop(), Is.EqualTo(new Range(14, 16)));
		Assert.That(postfix.Output.Pop(), Is.EqualTo(new Range(3, 6)));
		Assert.That(postfix.Output.Pop(), Is.EqualTo(new Range(0, 2)));
	}

	[Test]
	public void ParseIsNotWithMultipleProceedings()
	{
		var postfix = new ShuntingYard("if bla is not (bla - 25)");
		Assert.That(postfix.Output.Pop(), Is.EqualTo(new Range(7, 13)));
		Assert.That(postfix.Output.Pop(), Is.EqualTo(new Range(19, 20)));
		Assert.That(postfix.Output.Pop(), Is.EqualTo(new Range(21, 23)));
		Assert.That(postfix.Output.Pop(), Is.EqualTo(new Range(15, 18)));
		Assert.That(postfix.Output.Pop(), Is.EqualTo(new Range(3, 6)));
		Assert.That(postfix.Output.Pop(), Is.EqualTo(new Range(0, 2)));
	}
}