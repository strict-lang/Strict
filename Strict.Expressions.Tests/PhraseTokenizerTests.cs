﻿namespace Strict.Expressions.Tests;

public sealed class PhraseTokenizerTests
{
	[TestCase("\"")]
	[TestCase("\"random")]
	[TestCase("\"random + \"\"")]
	[TestCase("\"\"\"\"\"")]
	public void UnterminatedString(string code) =>
		Assert.That(() => GetTokens(code).Count,
			Throws.InstanceOf<PhraseTokenizer.UnterminatedString>());

	private static List<string> GetTokens(string code)
	{
		var result = new List<string>();
		new PhraseTokenizer(code).ProcessEachToken(range => result.Add(code[range]));
		return result;
	}

	[TestCase(" ")]
	[TestCase("  ")]
	[TestCase("hello ")]
	[TestCase("5 +  2")]
	[TestCase("\"hello \"  + 2")]
	public void InvalidSpacing(string code) =>
		Assert.That(() => GetTokens(code).Count,
			Throws.InstanceOf<PhraseTokenizer.InvalidSpacing>());

	[TestCase("()")]
	[TestCase("Run()")]
	[TestCase("list = ()")]
	[TestCase("() + 5")]
	[TestCase("(")]
	public void InvalidEmptyOrUnmatchedBrackets(string code) =>
		Assert.That(() => GetTokens(code).Count,
			Throws.InstanceOf<PhraseTokenizer.InvalidEmptyOrUnmatchedBrackets>());

	[TestCase("512", 1)]
	[TestCase("hello", 1)]
	[TestCase("\"something\"", 1)]
	[TestCase("5 + 2", 3)]
	[TestCase("5 + 2 + 3 + 5 * 5", 9)]
	[TestCase("\"5 + 2\"", 1)]
	[TestCase("\"5 + 2\" + 5", 3)]
	[TestCase("\"5 + 2\" + \"6 + 3\"", 3)]
	[TestCase("\"hello \"\"Murali\"\"\"", 1)]
	[TestCase("\"hello + \"\"Murali\"\"\"", 1)]
	[TestCase("\"\"\"5\" + 5 + \"Hello\"", 5)]
	[TestCase("(5 + 3) * 2", 7)]
	[TestCase("(5 + (1 + 2)) * 2", 11)]
	[TestCase("(5)", 1)]
	[TestCase("(5, 2)", 1)]
	[TestCase("(5) * 2", 3)]
	[TestCase("(5, 3) * 2", 3)]
	[TestCase("(5, 2) + (6, 2)", 3)]
	[TestCase("(5, (1 + 1)) + (6, 2)", 3)]
	[TestCase("(\"Hello\")", 1)]
	[TestCase("(\"Hello\", \"Hi\") + 1", 3)]
	[TestCase("((\"Hello1\"), (\"Hi\")) + 1", 3)]
	[TestCase("(\"Hello\", \"World\") + (1, 2) is (\"Hello\", \"World\", \"1\", \"2\")", 5)]
	[TestCase("(\"1\", \"2\") to Numbers + (3, 4) is (\"1\", \"2\", \"3\", \"4\")", 7)]
	[TestCase("Run(5, 5).Test(1)", 1)]
	[TestCase("Run((5, 5)).Test(1)", 1)]
	[TestCase("ReduceButGrow(120, 5, 40, 0) is 10", 3)]
	[TestCase("ArithmeticFunction(10, 5).Calculate(\"add\") is 15", 3)]
	[TestCase("ReduceButGrow(2, 3, 4, 5).GetMultiplicationOfNumbers is 120", 3)]
	[TestCase("RemoveExclamation(\"Hello There!\").Remove is \"Hello There\"", 3)]
	public void GetTokenRanges(string code, int expectedTokensCount)
	{
		var tokens = GetTokens(code);
		var tokensAsString = string.Join(' ', tokens).Replace("( ", "(").Replace(" )", ")").
			Replace(" , ", ", ");
		Assert.That(tokens.Count, Is.EqualTo(expectedTokensCount), string.Join(',', tokens));
		Assert.That(tokensAsString, Is.EqualTo(code));
	}

	[TestCase("1, 2", 3)]
	[TestCase("\"Hi\", \" there\"", 3)]
	[TestCase("(1), (2)", 3)]
	[TestCase("1, 2, 3, 4, 5", 9)]
	[TestCase("(1, 2), (3, 4), (5)", 5)]
	public void GetInnerListElements(string code, int expectedTokensCount) =>
		GetTokenRanges(code, expectedTokensCount);
}