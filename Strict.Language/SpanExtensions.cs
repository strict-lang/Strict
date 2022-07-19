using System;
using System.Collections.Generic;

namespace Strict.Language;

/// <summary>
/// These extensions are only for strict used for parsing Strict files with strict rules
/// </summary>
public static class SpanExtensions
{
	public static SpanSplitEnumerator Split(this ReadOnlySpan<char> input, char splitter = ' ',
		StringSplitOptions options = StringSplitOptions.None)
	{
		if (input.Length == 0)
			throw new EmptyInputIsNotAllowed();
		if (options == StringSplitOptions.RemoveEmptyEntries)
			throw new NotSupportedException("Strict will not allow multiple empty lines or spaces");
		return new SpanSplitEnumerator(input, splitter, options);
	}

	public class EmptyInputIsNotAllowed : Exception { }

	public static SpanSplitEnumerator SplitLines(this ReadOnlySpan<char> input) =>
		Split(input, '\n');

	public static bool Compare(this ReadOnlySpan<char> first, ReadOnlySpan<char> second)
	{
		if (first.Length != second.Length)
			return false;
		for (var index = 0; index < first.Length; index++)
			if (first[index] != second[index])
				return false;
		return true;
	}

	public static bool Any(this ReadOnlySpan<char> input, IEnumerable<string> items)
	{
		foreach (var item in items)
			if (input.Compare(item.AsSpan()))
				return true;
		return false;
	}

	public static bool Contains(this ReadOnlySpan<char> input, IEnumerable<string> items)
	{
		foreach (var item in items)
			if (input.IndexOf(item.AsSpan()) >= 0)
				return true;
		return false;
	}

	public static int Count(this ReadOnlySpan<char> input, char item)
	{
		var count = 0;
		foreach (var letter in input)
			if (letter == item)
				count++;
		return count;
	}
}