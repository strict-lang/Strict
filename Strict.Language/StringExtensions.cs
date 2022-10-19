using System.Collections.Generic;

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

	public static string MakeItPlural(this string name) => name + "s";
}