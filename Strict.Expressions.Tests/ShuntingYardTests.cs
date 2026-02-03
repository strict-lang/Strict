namespace Strict.Expressions.Tests;

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
	[TestCase("(\"Hello World\", \"Hi World\", \"5 + 4 + 5\")",
		"(\"Hello World\", \"Hi World\", \"5 + 4 + 5\")")]
	[TestCase("1, 2, 3, 4, 5", "1, ,, 2, ,, 3, ,, 4, ,, 5")]
	[TestCase("\"Hi\", \" there\"", "\"Hi\", ,, \" there\"")]
	[TestCase("(1, 2, 3) + (3, 4), (4)", "(1, 2, 3), (3, 4), +, ,, (4)")]
	[TestCase("ArithmeticFunction(10, 5).Calculate(\"add\") is 15",
		"ArithmeticFunction(10, 5).Calculate(\"add\"), 15, is")]
	[TestCase("Foo(\"Hello World\")", "Foo(\"Hello World\")")]
	[TestCase("Foo(5 + 5)", "Foo(5 + 5)")]
	[TestCase("Colors(Width * 5 + 1)", "Colors(Width * 5 + 1)")]
	[TestCase("Colors(Width * number.Length + 1)", "Colors(Width * number.Length + 1)")]
	[TestCase("Math.Add(5)", "Math.Add(5)")]
	[TestCase("not true is false", "true, not, false, is")]
	[TestCase("-5 * 2", "-5, 2, *")]
	[TestCase("-(5 * 2)", "5, 2, *, -")]
	public void Parse(string input, string expected) =>
		Assert.That(
			string.Join(", ",
				new ShuntingYard(input).Output.Reverse().Select(range => input[range])),
			Is.EqualTo(expected));

	[Test]
	public void ParseIsNot()
	{
		const string Input = "five is not 5";
		var tokens = new ShuntingYard(Input);
		var tokensText = string.Join(", ", tokens.Output.Select(range => Input[range]));
		Assert.That(Input[tokens.Output.Pop()], Is.EqualTo(UnaryOperator.Not), tokensText);
		Assert.That(Input[tokens.Output.Pop()], Is.EqualTo(BinaryOperator.Is));
		Assert.That(Input[tokens.Output.Pop()], Is.EqualTo("5"));
		Assert.That(Input[tokens.Output.Pop()], Is.EqualTo("five"));
	}

	[Test]
	public void ParseIsNotWithMultipleProceedings()
	{
		const string Input = "five is not (five - 25)";
		var tokens = new ShuntingYard(Input);
		Assert.That(Input[tokens.Output.Pop()], Is.EqualTo(UnaryOperator.Not));
		Assert.That(Input[tokens.Output.Pop()], Is.EqualTo(BinaryOperator.Is));
		Assert.That(Input[tokens.Output.Pop()], Is.EqualTo(BinaryOperator.Minus));
		Assert.That(Input[tokens.Output.Pop()], Is.EqualTo("25"));
		Assert.That(Input[tokens.Output.Pop()], Is.EqualTo("five"));
		Assert.That(Input[tokens.Output.Pop()], Is.EqualTo("five"));
	}

	[TestCase("logger.Log(\"Hello)")]
	[TestCase("logger.Log(\"Hello\"\")")]
	[TestCase("(1\")")]
	[TestCase("(\"Hel\"lo\")")]
	public void UnterminatedStringInsideBrackets(string input) =>
		Assert.That(() => new ShuntingYard(input),
			Throws.InstanceOf<PhraseTokenizer.UnterminatedString>());

	[Test]
	public void ParseIsNotIn()
	{
		const string Input = "\" \" is not in value";
		var tokens = new ShuntingYard(Input);
		Assert.That(Input[tokens.Output.Pop()], Is.EqualTo(UnaryOperator.Not));
		Assert.That(Input[tokens.Output.Pop()], Is.EqualTo(BinaryOperator.In));
		Assert.That(Input[tokens.Output.Pop()], Is.EqualTo(Base.ValueLowercase));
		Assert.That(Input[tokens.Output.Pop()], Is.EqualTo("\" \""));
	}

	[Test]
	public void ParseIsNotInMethod()
	{
		const string Input = "five is not in (5, 6, 4)";
		var tokens = new ShuntingYard(Input);
		Assert.That(Input[tokens.Output.Pop()], Is.EqualTo(UnaryOperator.Not));
		Assert.That(Input[tokens.Output.Pop()], Is.EqualTo(BinaryOperator.In));
		Assert.That(Input[tokens.Output.Pop()], Is.EqualTo("(5, 6, 4)"));
		Assert.That(Input[tokens.Output.Pop()], Is.EqualTo("five"));
	}

	[Test]
	public void ParseInMethod()
	{
		const string Input = "five is in (5, 6, 4)";
		var tokens = new ShuntingYard(Input);
		Assert.That(Input[tokens.Output.Pop()], Is.EqualTo(BinaryOperator.In));
		Assert.That(Input[tokens.Output.Pop()], Is.EqualTo("(5, 6, 4)"));
		Assert.That(Input[tokens.Output.Pop()], Is.EqualTo("five"));
	}

	[Test]
	public void ParseComplexBinaryCheck()
	{
		const string Input = "value and other or (not value) and (not other)";
		//same: const string Input = "(value and other) or ((not value) and (not other))";
		var tokens = new ShuntingYard(Input);
		// Overall or
		Assert.That(Input[tokens.Output.Pop()], Is.EqualTo(BinaryOperator.Or));
		// Right side and
		Assert.That(Input[tokens.Output.Pop()], Is.EqualTo(BinaryOperator.And));
		Assert.That(Input[tokens.Output.Pop()], Is.EqualTo(UnaryOperator.Not));
		Assert.That(Input[tokens.Output.Pop()], Is.EqualTo("other"));
		Assert.That(Input[tokens.Output.Pop()], Is.EqualTo(UnaryOperator.Not));
		Assert.That(Input[tokens.Output.Pop()], Is.EqualTo("value"));
		// Left side and
		Assert.That(Input[tokens.Output.Pop()], Is.EqualTo(BinaryOperator.And));
		Assert.That(Input[tokens.Output.Pop()], Is.EqualTo("other"));
		Assert.That(Input[tokens.Output.Pop()], Is.EqualTo("value"));
		Assert.That(tokens.Output, Has.Count.EqualTo(0));
	}

	[Test]
	public void ParseReverseRangeComparision()
	{
		const string Input = "Range(-5, -10).Reverse is Range(-9, -4)";
		var tokens = new ShuntingYard(Input);
		Assert.That(Input[tokens.Output.Pop()], Is.EqualTo(BinaryOperator.Is));
		Assert.That(Input[tokens.Output.Pop()], Is.EqualTo("Range(-9, -4)"));
		Assert.That(Input[tokens.Output.Pop()], Is.EqualTo("Range(-5, -10).Reverse"));
	}
}