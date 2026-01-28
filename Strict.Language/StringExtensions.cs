using System.Collections;

namespace Strict.Language;

public static class StringExtensions
{
	public static string ToBrackets<T>(this IReadOnlyCollection<T> list) =>
		list.Count > 0
			? "(" + list.ToWordList() + ")"
			: "";

	public static string ToWordList<T>(this IEnumerable<T> list, string separator = ", ") =>
		list is IDictionary<string, object?> dictionary
			? dictionary.DictionaryToWordList(separator)
			: string.Join(separator, list);

	public static string DictionaryToWordList<TKey, TValue>(this IDictionary<TKey, TValue> list,
		string separator = "; ")
	{
		var result = new List<string>();
		foreach (var pair in list)
			result.Add(pair.Key + "=" + (pair.Value is IEnumerable values
				? values.EnumerableToWordList()
				: pair.Value?.ToString()));
		return result.ToWordList(separator);
	}

	public static string IDictionaryToWordList(this IDictionary list, string separator = "; ")
	{
		var enumerator = list.GetEnumerator();
		using var disposeEnumerator = enumerator as IDisposable;
		var result = new List<string>();
		while (enumerator.MoveNext())
			result.Add(enumerator.Key + "=" + (enumerator.Value is IEnumerable values
				? values.EnumerableToWordList()
				: enumerator.Value?.ToString()));
		return result.ToWordList(separator);
	}

	public static string EnumerableToWordList(this IEnumerable values, string separator = ", ") =>
		values is IDictionary<string, object?> valuesDictionary
			? valuesDictionary.DictionaryToWordList()
			: values is IDictionary valuesIDictionary
				? valuesIDictionary.IDictionaryToWordList()
				: values as string ?? values.Cast<object?>().ToWordList(separator);

	extension(string text)
	{
		/// <summary>
		/// Faster version of Regex.IsMatch(text, @"^[A-Za-z]+$");
		/// </summary>
		public bool IsWord()
		{
			foreach (var c in text)
				if (!char.IsAsciiLetter(c))
					return false;
			return true;
		}

		public bool IsKeyword() => Keyword.GetAllKeywords.Contains(text);

		public bool IsWordOrWordWithNumberAtEnd(out int number)
		{
			number = -1;
			for (var index = 0; index < text.Length; index++)
				if (!char.IsAsciiLetter(text[index]))
					return index == text.Length - 1 && int.TryParse(text[index].ToString(), out number) &&
						number is > 1 and < 10;
			return true;
		}

		public bool IsAlphaNumericWithAllowedSpecialCharacters()
		{
			for (var index = 0; index < text.Length; index++)
				if (!char.IsAsciiLetter(text[index]) &&
					(index == 0 || text[index] != '-' && !char.IsNumber(text[index])))
					return false;
			return true;
		}

		public string MakeFirstLetterUppercase() =>
			text[..1].ToUpperInvariant() + text[1..];

		public string MakeFirstLetterLowercase() =>
			text[..1].ToLowerInvariant() + text[1..];

		public string GetTextInsideBrackets()
		{
			var bracketStartIndex = text.IndexOf('(');
			var bracketEndIndex = text.IndexOf(')');
			return bracketStartIndex > -1 && bracketStartIndex < bracketEndIndex
				? text[(bracketStartIndex + 1)..bracketEndIndex]
				: text;
		}

		public string Pluralize() =>
			text + (text.EndsWith("s", StringComparison.Ordinal) ||
				text.EndsWith("sh", StringComparison.Ordinal) ||
				text.EndsWith("ch", StringComparison.Ordinal) ||
				text.EndsWith("x", StringComparison.Ordinal) || text.EndsWith("z", StringComparison.Ordinal)
					? text.EndsWith("settings", StringComparison.OrdinalIgnoreCase)
						? ""
						: "es"
					: "s");

		public bool IsOperatorOrAllowedMethodName() =>
			text.Length == 1 && text[0].IsSingleCharacterOperator() ||
			text[0] is 'X' or 'Y' or 'Z' or 'W';

		public bool StartsWith(params string[] partialNames) =>
			partialNames.Any(x => text.StartsWith(x, StringComparison.InvariantCultureIgnoreCase));
	}
}