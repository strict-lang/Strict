using System;
using System.Collections.Generic;

namespace Strict.Language;

public static class StringExtensions
{
	public static string[] SplitLines(this string text) =>
		text.Split(new[] { Environment.NewLine, "\n" }, StringSplitOptions.None);

	public static string[] SplitWords(this string text) => text.Split(' ');

	public static string[] SplitWordsAndPunctuation(this string text) =>
		text.Split(new[] { ' ', '(', ')', ',' }, StringSplitOptions.RemoveEmptyEntries);

	public static string ToWordList<T>(this IEnumerable<T> list) => string.Join(", ", list);

	public static string ToBrackets<T>(this IReadOnlyCollection<T> list) =>
		list.Count > 0
			? "(" + ToWordList(list) + ")"
			: "";

	/// <summary>
	/// Faster version of Regex.IsMatch(text, @"^[A-Za-z]+$");
	/// </summary>
	public static bool IsWord(this string text)
	{
		foreach (var c in text)
			if (c is (< 'A' or > 'Z') and (< 'a' or > 'z'))
				return false;
		return true;
	}

	public static bool IsWordWithNumber(this string text)
	{
		for (var index = 0; index < text.Length; index++)
			if (text[index] is (< 'A' or > 'Z') and (< 'a' or > 'z') &&
				(index == 0 || text[index] is < '0' or > '9'))
				return false;
		return true;
	}

	public static string MakeFirstLetterUppercase(this string name) =>
		name[..1].ToUpperInvariant() + name[1..];
}