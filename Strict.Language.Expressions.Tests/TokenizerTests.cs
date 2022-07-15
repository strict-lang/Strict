using NUnit.Framework;

namespace Strict.Language.Expressions.Tests;

public class TokenizerTests
{
	[TestCase("\"")]
	[TestCase("\"random")]
	[TestCase("\"random + \"\"")]
	[TestCase("\"\"\"\"\"")]
	public void UnterminatedString(string code) =>
		Assert.That(() => new Tokenizer(code).GetTokens(),
			Throws.InstanceOf<Tokenizer.UnterminatedString>());

	[TestCase(" ")]
	[TestCase("  ")]
	[TestCase("hello ")]
	[TestCase("5 +  2")]
	[TestCase("\"hello \"  + 2")]
	public void InvalidSpacing(string code) =>
		Assert.That(() => new Tokenizer(code).GetTokens(),
			Throws.InstanceOf<Tokenizer.InvalidSpacing>());

	[TestCase("()")]
	[TestCase("Run()")]
	[TestCase("let list = ()")]
	[TestCase("let list = () + 5")]
	public void InvalidBrackets(string code) =>
		Assert.That(() => new Tokenizer(code).GetTokens(),
			Throws.InstanceOf<Tokenizer.InvalidBrackets>());

	[TestCase("5", 1)]
	[TestCase("hello", 1)]
	[TestCase("5 hello", 1)]
	[TestCase("\"something\"", 1)]
	[TestCase("5 + 2", 3)]
	[TestCase("5 + 2 + 3 + 5 * 5", 9)]
	[TestCase("(5 + 3) * 2", 7)]
	[TestCase("(5 + (1 + 2)) * 2", 11)]
	[TestCase("\"5 + 2\"", 1)]
	[TestCase("\"5 + 2\" + 5", 3)]
	[TestCase("\"5 + 2\" + \"6 + 3\"", 3)]
	[TestCase("\"hello \"\"Murali\"\"\"", 1)]
	[TestCase("\"hello + \"\"Murali\"\"\"", 1)]
	[TestCase("\"\"\"5\" + 5 + \"Hello\"", 5)]
	[TestCase("(5, 2)", 1)]
	[TestCase("(5)", 1)]
	[TestCase("(5) * 2", 3)]
	[TestCase("(5, 3) * 2", 3)]
	[TestCase("(5, 2) + (6, 2)", 3)]
	[TestCase("(5, (1 + 1)) + (6, 2)", 3)]
	[TestCase("(\"Hello\")", 1)]
	[TestCase("(\"Hello\", \"Hi\") + 1", 3)]
	[TestCase("((\"Hello1\"), (\"Hi\")) + 1", 3)]
	[TestCase("(\"Hello\", \"World\") + (1, 2) is (\"Hello\", \"World\", \"1\", \"2\")", 5)]
	[TestCase("(\"1\", \"2\") to Numbers + (3, 4) is (\"1\", \"2\", \"3\", \"4\")", 7)]
	public void GetTokens(string code, int expectedTokensCount) =>
		Assert.That(new Tokenizer(code).GetTokens().Count, Is.EqualTo(expectedTokensCount), string.Join(", ", new Tokenizer(code).GetTokens()));

	//TODO: Method call, List, Generics parsing should be handled
}