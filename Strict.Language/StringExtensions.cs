using System;
using System.Collections.Generic;
using System.Linq;

namespace Strict.Language
{
	public static class StringExtensions
	{
		public static string[] SplitLines(this string text)
			=> text.Split(new[] { Environment.NewLine, "\n" }, StringSplitOptions.None);

		public static string[] SplitWords(this string text) => text.Split(' ');

		public static string[] SplitWordsAndPunctuation(this string text) =>
			text.Split(new[] { ' ', '(', ')' }, StringSplitOptions.RemoveEmptyEntries);

		public static string ToWordString<T>(this IReadOnlyCollection<T> list) =>
			string.Join(", ", list);
	}
}