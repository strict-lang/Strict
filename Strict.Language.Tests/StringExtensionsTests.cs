using System.Collections;

namespace Strict.Language.Tests;

public sealed class StringExtensionsTests
{
	[TestCase("(Hello, Hi)", "Hello", "Hi")]
	[TestCase("(1, 2, 5, 6, Hi)", 1, 2, 5, 6, "Hi")]
	[TestCase("(Hello World!, UserName)", "Hello World!", "UserName")]
	public void ToBrackets(string expected, params object[] texts) =>
		Assert.That(texts.ToBrackets(), Is.EqualTo(expected));

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
			Is.False.And.Matches<bool>(_ => number is -1 or 0));

	[TestCase("Hello-World")]
	[TestCase("Kata-Examples-001")]
	[TestCase("heLLo785-100")]
	[TestCase("r7-And-Text")]
	public void IsAlphaNumericWithAllowedSpecialCharacters(string text) =>
		Assert.That(text.IsAlphaNumericWithAllowedSpecialCharacters(), Is.True);

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
	public void MakeFirstLetterUppercase(string text, string expected) =>
		Assert.That(text.MakeFirstLetterUppercase(), Is.EqualTo(expected));

	[TestCase("Kata", "kata")]
	[TestCase("hello-world", "hello-world")]
	[TestCase("Hello World", "hello World")]
	[TestCase("@heLLo785-100", "@heLLo785-100")]
	public void MakeFirstLetterLowercase(string text, string expected) =>
		Assert.That(text.MakeFirstLetterLowercase(), Is.EqualTo(expected));

	[TestCase("Kata(12452)", "12452")]
	[TestCase("hello(world(4528))", "world(4528")]
	[TestCase("(Hello World)", "Hello World")]
	[TestCase("@he(LLo785-100", "@he(LLo785-100")]
	public void GetTextInsideBrackets(string text, string expected) =>
		Assert.That(text.GetTextInsideBrackets(), Is.EqualTo(expected));

	[TestCase("Car", "Cars")]
	[TestCase("bus", "buses")]
	[TestCase("boss", "bosses")]
	[TestCase("marsh", "marshes")]
	[TestCase("lunch", "lunches")]
	[TestCase("tax", "taxes")]
	[TestCase("blitz", "blitzes")]
	public void Pluralize(string text, string expected) =>
		Assert.That(text.Pluralize(), Is.EqualTo(expected));

	[TestCase("X")]
	[TestCase("Y")]
	[TestCase("Z")]
	[TestCase("W")]
	[TestCase("+")]
	[TestCase("/")]
	[TestCase(">")]
	public void IsOperatorOrAllowedMethodName(string text) =>
		Assert.That(text.IsOperatorOrAllowedMethodName(), Is.True);

	[TestCase("Hello")]
	[TestCase("@")]
	[TestCase("A")]
	[TestCase(".")]
	[TestCase("0")]
	public void NotOperatorOrAllowedMethodName(string text) =>
		Assert.That(text.IsOperatorOrAllowedMethodName(), Is.False);

	[Test]
	public void StartsWith()
	{
		Assert.That(StringExtensions.StartsWith("Hi there, what's up?", "Hi"), Is.True);
		Assert.That(StringExtensions.StartsWith("Hi there, what's up?", "what"), Is.False);
		Assert.That(StringExtensions.StartsWith("bcdeuf", "bc"), Is.True);
		Assert.That(StringExtensions.StartsWith("bcdeuf", "abc"), Is.False);
		Assert.That(StringExtensions.StartsWith("Hi there, what's up?", "Hi", "there", "what"), Is.True);
		Assert.That(StringExtensions.StartsWith("Hi there, what's up?", "she", "there", "what"), Is.False);
	}

	[Test]
	public void ToWordList()
	{
		Assert.That(new List<string> { "hi", "there" }.ToWordList(), Is.EqualTo("hi, there"));
		Assert.That(new[] { 1, 2, 3 }.ToWordList(), Is.EqualTo("1, 2, 3"));
		Assert.That(new Dictionary<string, object?> { { "number", 5 }, { "values", new[] { 0, 1, 2 } } }.
			DictionaryToWordList(), Is.EqualTo("number=5; values=0, 1, 2"));
		IDictionary dict = new Hashtable
		{
			{ "name", "Kata" },
			{ "ids", new List<int> { 1, 2, 3 } }
		};
		Assert.That(dict.DictionaryToWordList(), Is.EqualTo("name=Kata; ids=1, 2, 3"));
		IEnumerable values = new ArrayList { "apple", "banana", "cherry" };
		Assert.That(values.EnumerableToWordList(), Is.EqualTo("apple, banana, cherry"));
	}
}