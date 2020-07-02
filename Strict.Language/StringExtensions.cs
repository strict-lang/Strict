using System;
using System.Collections.Generic;

namespace Strict.Language
{
	public static class StringExtensions
	{
		public static string[] SplitLines(this string text)
			=> text.Split(new[] { Environment.NewLine, "\n" }, StringSplitOptions.None);

		public static string[] SplitWords(this string text) => text.Split(' ');

		public static string[] SplitWordsAndPunctuation(this string text) =>
			text.Split(new[] { ' ', '(', ')', ',' }, StringSplitOptions.RemoveEmptyEntries);

		public static string ToWordString<T>(this IReadOnlyCollection<T> list) =>
			string.Join(", ", list);

		public static string InBrackets<T>(this IReadOnlyCollection<T> list) =>
			list.Count > 0
				? "(" + string.Join(", ", list) + ")"
				: "";
	}
}