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
	[TestCase("(\"Hello World\", \"Hi World\", \"5 + 4 + 5\")", "(\"Hello World\", \"Hi World\", \"5 + 4 + 5\")")]
	[TestCase("1, 2, 3, 4, 5", "1, ,, 2, ,, 3, ,, 4, ,, 5")]
	[TestCase("\"Hi\", \" there\"", "\"Hi\", ,, \" there\"")]
	[TestCase("(1, 2, 3) + (3, 4), (4)", "(1, 2, 3), (3, 4), +, ,, (4)")]
	[TestCase("ArithmeticFunction(10, 5).Calculate(\"add\") is 15", "ArithmeticFunction(10, 5).Calculate(\"add\"), 15, is")]
	[TestCase("Foo(\"Hello World\")", "Foo(\"Hello World\")")]
	public void Parse(string input, string expected)
	{
		var tokens = new ShuntingYard(input).Output.Reverse().Select(range => input[range]);
		Assert.That(string.Join(", ", tokens), Is.EqualTo(expected));
	}

	[Test]
	public void ParseIfWithIsNot()
	{
		const string Input = "if bla is not 5";
		var tokens = new ShuntingYard(Input);
		Assert.That(Input[tokens.Output.Pop()], Is.EqualTo(BinaryOperator.IsNot));
		Assert.That(Input[tokens.Output.Pop()], Is.EqualTo("5"));
		Assert.That(Input[tokens.Output.Pop()], Is.EqualTo("bla"));
		Assert.That(Input[tokens.Output.Pop()], Is.EqualTo("if"));
	}

	[Test]
	public void ParseIsNotWithMultipleProceedings()
	{
		const string Input = "if bla is not (bla - 25)";
		var postfix = new ShuntingYard(Input);
		Assert.That(Input[postfix.Output.Pop()], Is.EqualTo(BinaryOperator.IsNot));
		Assert.That(Input[postfix.Output.Pop()], Is.EqualTo(BinaryOperator.Minus));
		Assert.That(Input[postfix.Output.Pop()], Is.EqualTo("25"));
		Assert.That(Input[postfix.Output.Pop()], Is.EqualTo("bla"));
		Assert.That(Input[postfix.Output.Pop()], Is.EqualTo("bla"));
		Assert.That(Input[postfix.Output.Pop()], Is.EqualTo("if"));
	}

	[TestCase("log.Write(\"Hello)")]
	[TestCase("log.Write(\"Hello\"\")")]
	[TestCase("(1\")")]
	[TestCase("(\"Hel\"lo\")")]
	public void UnterminatedStringInsideBrackets(string input) =>
		Assert.That(() => new ShuntingYard(input),
			Throws.InstanceOf<PhraseTokenizer.UnterminatedString>());

	[Test]
	public void ParseIfWithIsNotInMethod()
	{
		const string Input = "if bla is not in (5, 6, 4)";
		var tokens = new ShuntingYard(Input);
		Assert.That(Input[tokens.Output.Pop()], Is.EqualTo(BinaryOperator.IsNotIn));
		Assert.That(Input[tokens.Output.Pop()], Is.EqualTo("(5, 6, 4)"));
		Assert.That(Input[tokens.Output.Pop()], Is.EqualTo("bla"));
		Assert.That(Input[tokens.Output.Pop()], Is.EqualTo("if"));
	}

	[Test]
	public void ParseIfWithIsInMethod()
	{
		const string Input = "if bla is in (5, 6, 4)";
		var tokens = new ShuntingYard(Input);
		Assert.That(Input[tokens.Output.Pop()], Is.EqualTo(BinaryOperator.IsIn));
		Assert.That(Input[tokens.Output.Pop()], Is.EqualTo("(5, 6, 4)"));
		Assert.That(Input[tokens.Output.Pop()], Is.EqualTo("bla"));
		Assert.That(Input[tokens.Output.Pop()], Is.EqualTo("if"));
	}
}