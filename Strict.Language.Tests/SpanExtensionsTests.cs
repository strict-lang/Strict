using System;
using NUnit.Framework;

namespace Strict.Language.Tests;

public class SpanExtensionsTests
{
	[Test]
	public void EmptyInputIsNotAllowed() =>
		Assert.That(() => "".AsSpan().Split(), Throws.InstanceOf<SpanExtensions.EmptyInputIsNotAllowed>());

	[Test]
	public void GetOuterRange()
	{
		const string Text = "Hello 1234";
		var outerRange = (4..).GetOuterRange((2..8).GetOffsetAndLength(Text.Length));
		Assert.That(outerRange, Is.EqualTo(6..8));
		Assert.That(Text[outerRange], Is.EqualTo("12"));
	}

	[Test]
	public void RemoveEmptyEntriesIsNotSupported() =>
		Assert.That(() => "as ".AsSpan().Split(options: StringSplitOptions.RemoveEmptyEntries), Throws.InstanceOf<NotSupportedException>());

	[TestCase(" ")]
	[TestCase("  ")]
	[TestCase(" a")]
	[TestCase("a  b", 2)]
	public void InvalidConsecutiveSplitter(string input, int expectedIndex = 0) =>
		Assert.That(() =>
			{
				foreach (var word in input.AsSpan().Split())
					Assert.That(word.ToString(), Is.Not.Empty);
			}, //ncrunch: no coverage
			Throws.InstanceOf<SpanSplitEnumerator.InvalidConsecutiveSplitter>().With.Message.
				EqualTo("Input=" + input + ", index=" + expectedIndex));

	[TestCase("a ")]
	[TestCase("aksbdkasbd\nsadnakjsdnk\n", '\n')]
	[TestCase("1 + 2 * 4 ")]
	public void EmptyEntryNotAllowedAtTheEnd(string input, char splitter = ' ') =>
		Assert.That(() =>
			{
				foreach (var word in input.AsSpan().Split(splitter))
					Assert.That(word.ToString(), Is.Not.Empty);
			}, //ncrunch: no coverage
			Throws.InstanceOf<SpanSplitEnumerator.EmptyEntryNotAllowedAtTheEnd>());

	[TestCase("1")]
	[TestCase("1 2")]
	[TestCase("1 2 3 4 5 6 78 word 85 65")]
	[TestCase("1 \n 85 65")]
	[TestCase("first \n 8second 65third")]
	[TestCase(@"has numbers
GetComplicatedSequenceTexts returns Texts
	ConvertingNumbers(1, 21).GetComplicatedSequenceTexts is (""7"", ""16"")
	return for numbers
		 to Text
		Length * Length
		4 + value * 3
		 to Text")]
	public void SplitWords(string input)
	{
		var index = 0;
		var expectedWords = input.SplitWords();
		foreach (var wordSpan in input.AsSpan().Split())
		{
			Assert.That(wordSpan.ToString(), Is.EqualTo(expectedWords[index]));
			index++;
		}
		Assert.That(index, Is.EqualTo(expectedWords.Length));
	}

	[TestCase(@"1, 2")]
	[TestCase(@"word, number, 123")]
	public void SplitWordsByComma(string input)
	{
		var index = 0;
		var expectedWords = input.Split(',', StringSplitOptions.TrimEntries);
		foreach (var wordSpan in input.AsSpan().Split(',', StringSplitOptions.TrimEntries))
		{
			Assert.That(wordSpan.ToString(), Is.EqualTo(expectedWords[index]));
			index++;
		}
		Assert.That(index, Is.EqualTo(expectedWords.Length));
	}

	[TestCase("aksbdkasbd\nsadnakjsdnk")]
	public void SplitLines(string input)
	{
		var expectedLines = input.SplitLines();
		var index = 0;
		foreach (var wordSpan in input.AsSpan().SplitLines())
		{
			Assert.That(wordSpan.ToString(), Is.EqualTo(expectedLines[index]));
			index++;
		}
		Assert.That(index, Is.EqualTo(expectedLines.Length));
	}

	[Test]
	public void Equals()
	{
		Assert.That("5".AsSpan().Compare("5".AsSpan()), Is.True);
		Assert.That("6".AsSpan().Compare("5".AsSpan()), Is.False);
		Assert.That("ABC".AsSpan().Compare("Abc".AsSpan()), Is.False);
		Assert.That("12345".AsSpan().Compare("12".AsSpan()), Is.False);
	}

	[TestCase("+")]
	[TestCase("*")]
	[TestCase("is")]
	public void AnyOperator(string input) =>
		Assert.That(input.AsSpan().Any(new[] { "+", "-", "*", "is" }), Is.True);

	[TestCase("is+")]
	public void NotAnyOperator(string input) =>
		Assert.That(input.AsSpan().Any(new[] { "+", "-", "*", "is" }), Is.False);

	[TestCase("5 * 6")]
	[TestCase("1 + 2")]
	[TestCase(@"""hello"" is Text")]
	public void ContainsOperator(string input) =>
		Assert.That(input.AsSpan().Contains(new[] { "+", "-", "*", "is" }), Is.True);

	[TestCase(@"""hello"" is Text")]
	public void NotContainsOperator(string input) =>
		Assert.That(input.AsSpan().Contains(new[] { "+", "-", "*" }), Is.False);

	[TestCase(@"""hello"" is Text")]
	public void Count(string input) =>
		Assert.That(input.AsSpan().Count('\"'), Is.EqualTo(2));
}