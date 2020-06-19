using System;

namespace Strict.Language
{
	/// <summary>
	/// Simple lexer to just parse one line and go through all the word elements
	/// </summary>
	public class LineLexer
	{
		public LineLexer(string line) => words = line.SplitWordsAndPunctuation();
		private readonly string[] words;
		public string Next() => HasNext ? words[index++] : throw new NoMoreWords();
		public bool HasNext => index < words.Length;
		private int index;

		public class NoMoreWords : Exception { }
	}
}