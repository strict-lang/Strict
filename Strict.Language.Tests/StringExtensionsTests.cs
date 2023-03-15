using NUnit.Framework;

namespace Strict.Language.Tests;

public sealed class StringExtensionsTests
{
	[TestCase("(Hello, Hi)", "Hello", "Hi")]
	[TestCase("(1, 2, 5, 6, Hi)", 1, 2, 5, 6, "Hi")]
	[TestCase("(Hello World!, UserName)", "Hello World!", "UserName")]
	public void ToBrackets(string expected, params object[] texts) => Assert.That(texts.ToBrackets(), Is.EqualTo(expected));

	[TestCase("Hello")]
	[TestCase("heLLo")]
	[TestCase("randomText")]
	[TestCase("AbcD")]
	public void IsWord(string text) => Assert.That(text.IsWord(), Is.True);

	[TestCase("Hello!")]
	[TestCase("heL.Lo")]
	[TestCase("!@#$%^&*()_")]
	[TestCase("122452ABCD")]
	public void IsNotWord(string text) => Assert.That(text.IsWord(), Is.False);

	[TestCase("Hello7", 7)]
	[TestCase("aBc5", 5)]
	public void IsWordOrWordWithNumberAtEnd(string text, int expected) =>
		Assert.That(text.IsWordOrWordWithNumberAtEnd(out var number),
			Is.True.And.Matches<bool>(_ => number == expected));

	[TestCase("Hello785")]
	[TestCase("ABCDE.")]
	[TestCase("A$")]
	[TestCase("$7")]
	[TestCase("ranDomStringG.7")]
	public void NotWithNumberAtEnd(string text) =>
		Assert.That(text.IsWordOrWordWithNumberAtEnd(out var number),
			Is.False.And.Matches<bool>(_ => number is -1 or 0)!);

	[TestCase("Hello-World")]
	[TestCase("Kata-Examples-001")]
	[TestCase("heLLo785-100")]
	[TestCase("r7-And-Text")]
	public void IsAlphaNumericWithAllowedSpecialCharacters(string text) => Assert.That(text.IsAlphaNumericWithAllowedSpecialCharacters(), Is.True);

	[TestCase("Hello_World")]
	[TestCase("0-Kata-Examples")]
	[TestCase("heLLo785-100$")]
	[TestCase("r7-And-Text@")]
	public void NotAlphaNumericWithAllowedSpecialCharacters(string text) =>
		Assert.That(text.IsAlphaNumericWithAllowedSpecialCharacters(), Is.False);

	[TestCase("kata", "Kata")]
	[TestCase("hello-world", "Hello-world")]
	[TestCase("Hello World", "Hello World")]
	[TestCase("7heLLo785-100", "7heLLo785-100")]
	public void MakeFirstLetterUppercase(string text, string expected) => Assert.That(text.MakeFirstLetterUppercase(), Is.EqualTo(expected));

	[TestCase("Kata", "kata")]
	[TestCase("hello-world", "hello-world")]
	[TestCase("Hello World", "hello World")]
	[TestCase("@heLLo785-100", "@heLLo785-100")]
	public void MakeFirstLetterLowercase(string text, string expected) => Assert.That(text.MakeFirstLetterLowercase(), Is.EqualTo(expected));

	[TestCase("Kata(12452)", "12452")]
	[TestCase("hello(world(4528))", "world(4528")]
	[TestCase("(Hello World)", "Hello World")]
	[TestCase("@he(LLo785-100", "@he(LLo785-100")]
	public void GetTextInsideBrackets(string text, string expected) => Assert.That(text.GetTextInsideBrackets(), Is.EqualTo(expected));

	[TestCase("Car", "Cars")]
	[TestCase("bus", "buses")]
	[TestCase("boss", "bosses")]
	[TestCase("marsh", "marshes")]
	[TestCase("lunch", "lunches")]
	[TestCase("tax", "taxes")]
	[TestCase("blitz", "blitzes")]
	public void Pluralize(string text, string expected) => Assert.That(text.Pluralize(), Is.EqualTo(expected));

	[TestCase("X")]
	[TestCase("Y")]
	[TestCase("Z")]
	[TestCase("W")]
	[TestCase("+")]
	[TestCase("/")]
	[TestCase(">")]
	public void IsOperatorOrAllowedMethodName(string text) => Assert.That(text.IsOperatorOrAllowedMethodName(), Is.True);

	[TestCase("Hello")]
	[TestCase("@")]
	[TestCase("A")]
	[TestCase(".")]
	[TestCase("0")]
	public void NotOperatorOrAllowedMethodName(string text) => Assert.That(text.IsOperatorOrAllowedMethodName(), Is.False);
}