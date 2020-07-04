using System;
using System.Collections.Generic;
using System.Text.RegularExpressions;

namespace Strict.Language
{
	public static class StringExtensions
	{
		public static string[] SplitLines(this string text)
			=> text.Split(new[] { Environment.NewLine, "\n" }, StringSplitOptions.None);

		public static string[] SplitWords(this string text) => text.Split(' ');

		public static string[] SplitWordsAndPunctuation(this string text) =>
			text.Split(new[] { ' ', '(', ')', ',' }, StringSplitOptions.RemoveEmptyEntries);

		public static string ToWordListString<T>(this IReadOnlyCollection<T> list) =>
			string.Join(", ", list);

		public static string ToBracketsString<T>(this IReadOnlyCollection<T> list) =>
			list.Count > 0
				? "(" + ToWordListString(list) + ")"
				: "";

		public static bool IsWord(this string text) => Regex.IsMatch(text, @"^[A-Za-z]+$");
	}
}