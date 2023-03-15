using System;
using System.Collections.Generic;
using System.Linq;

namespace Strict.Language;

public static class StringExtensions
{
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
			if (!char.IsAsciiLetter(c))
				return false;
		return true;
	}

	public static bool IsKeyword(this string text) => Keyword.GetAllKeywords.Contains(text);

	public static bool IsWordOrWordWithNumberAtEnd(this string text, out int number)
	{
		number = -1;
		for (var index = 0; index < text.Length; index++)
			if (!char.IsAsciiLetter(text[index]))
				return index == text.Length - 1 && int.TryParse(text[index].ToString(), out number) &&
					number is > 1 and < 10;
		return true;
	}

	public static bool IsAlphaNumericWithAllowedSpecialCharacters(this string text)
	{
		for (var index = 0; index < text.Length; index++)
			if (!char.IsAsciiLetter(text[index]) &&
				(index == 0 || text[index] != '-' && !char.IsNumber(text[index])))
				return false;
		return true;
	}

	public static string MakeFirstLetterUppercase(this string name) =>
		name[..1].ToUpperInvariant() + name[1..];

	public static string MakeFirstLetterLowercase(this string name) =>
		name[..1].ToLowerInvariant() + name[1..];

	public static string GetTextInsideBrackets(this string text)
	{
		var bracketStartIndex = text.IndexOf('(');
		var bracketEndIndex = text.IndexOf(')');
		return bracketStartIndex > -1 && bracketStartIndex < bracketEndIndex
			? text[(bracketStartIndex + 1)..bracketEndIndex]
			: text;
	}

	public static string Pluralize(this string word) =>
		word + (word.EndsWith("s", StringComparison.Ordinal) ||
			word.EndsWith("sh", StringComparison.Ordinal) ||
			word.EndsWith("ch", StringComparison.Ordinal) ||
			word.EndsWith("x", StringComparison.Ordinal) || word.EndsWith("z", StringComparison.Ordinal)
				? word.EndsWith("settings", StringComparison.OrdinalIgnoreCase)
					? ""
					: "es"
				: "s");

	public static bool IsOperatorOrAllowedMethodName(this string name) =>
		name.Length == 1 && name[0].IsSingleCharacterOperator() ||
		name[0] is 'X' or 'Y' or 'Z' or 'W';
}