using System.Collections;

namespace Strict.Language;

public static class StringExtensions
{
	public static string ToBrackets<T>(this IReadOnlyList<T> list) =>
		list.Any()
			? "(" + string.Join(DefaultSeparator, list) + ")"
			: "";

	private const string DefaultSeparator = ", ";

	public static string ToLines(this IEnumerable<string> lines) =>
		string.Join(Environment.NewLine, lines);

	public static string DictionaryToWordList<Key, Value>(this IReadOnlyDictionary<Key, Value> list,
		string separator = "; ", string keyValueSeparator = "=", bool outputTypes = false)
		where Key : notnull
	{
		var result = new List<string>();
		foreach (var pair in list)
			result.Add(pair.Key + (outputTypes && pair.Key is not string
				? " (" + pair.Key.GetType().Name + ")"
				: "") + keyValueSeparator + (pair.Value is IEnumerable values
				? values as string ?? string.Join(", ", values.Cast<object?>())
				: pair.Value + (outputTypes && pair.Value is not string && pair.Value is not int &&
					pair.Value is not double && pair.Value is not bool &&
					pair.Value?.GetType().Name != "ValueInstance"
						? " (" + pair.Value?.GetType().Name + ")"
						: "")));
		return string.Join<string>(separator, result);
	}

	public static bool IsWordOrWordWithNumberAtEnd(this ReadOnlySpan<char> text, out int number)
	{
		number = -1;
		for (var index = 0; index < text.Length; index++)
			if (!char.IsAsciiLetter(text[index]))
				return index == text.Length - 1 && char.IsAsciiDigit(text[index]) &&
					(number = text[index] - '0') is > 1 and < 10;
		return true;
	}

	public static bool IsKeyword(this ReadOnlySpan<char> text)
	{
		foreach (var keyword in Keyword.GetAllKeywords)
			if (keyword.Compare(text))
				return true;
		return false;
	}

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

		public bool IsAlphaNumericWithAllowedSpecialCharacters()
		{
			for (var index = 0; index < text.Length; index++)
				if (!char.IsAsciiLetter(text[index]) &&
					(index == 0 || text[index] != '-' && !char.IsNumber(text[index])))
					return false;
			return true;
		}

		public string MakeFirstLetterUppercase() =>
			text.Length == 0 || char.IsUpper(text[0])
				? text
				: string.Create(text.Length, text, static (span, s) =>
				{
					span[0] = char.ToUpperInvariant(s[0]);
					s.AsSpan(1).CopyTo(span[1..]);
				});

		public string MakeFirstLetterLowercase() =>
			text.Length == 0 || char.IsLower(text[0])
				? text
				: string.Create(text.Length, text, static (span, s) =>
				{
					span[0] = char.ToLowerInvariant(s[0]);
					s.AsSpan(1).CopyTo(span[1..]);
				});

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