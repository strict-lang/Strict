using System.Collections;

namespace Strict.Language;

public static class StringExtensions
{
	public static string ToWordList<T1, T2>(this IReadOnlyDictionary<T1, T2> list,
		string separator = "; ") where T2 : notnull
	{
		var result = new List<string>();
		foreach (var entry in list)
		{
			var value = "";
			if (entry.Value is IEnumerable values)
				foreach (var valueEntry in values)
					value += (value != ""
						? ", "
						: "") + valueEntry;
			else
				value = entry.Value.ToString();
			result.Add(entry.Key + "=" + value);
		}
		return result.ToWordList(separator);
	}

	public static string ToWordList<T>(this IEnumerable<T> list, string separator = ", ") =>
		string.Join(separator, list);

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

	public static bool StartsWith(this string name, params string[] partialNames) =>
		partialNames.Any(x => name.StartsWith(x, StringComparison.InvariantCultureIgnoreCase));
}