using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;

namespace Strict.Language;

/// <summary>
/// Optimized parsing of Strict code, which needs to be exactly in the correct format.
/// </summary>
public static class SpanExtensions
{
	//ncrunch: no coverage start, for performance reasons disabled here
	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static SpanSplitEnumerator Split(this ReadOnlySpan<char> input, char splitter = ' ',
		StringSplitOptions options = StringSplitOptions.None)
	{
		if (input.Length == 0)
			throw new EmptyInputIsNotAllowed();
		if (options == StringSplitOptions.RemoveEmptyEntries)
			throw new NotSupportedException(nameof(StringSplitOptions.RemoveEmptyEntries) +
				" is not allowed as Strict does not allow multiple empty lines or spaces anyways");
		return new SpanSplitEnumerator(input, splitter, options);
	}

	public class EmptyInputIsNotAllowed : Exception { }

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static SpanSplitEnumerator SplitLines(this ReadOnlySpan<char> input) =>
		Split(input, '\n');

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static RangeEnumerator SplitIntoRanges(this ReadOnlySpan<char> input,
		char splitter = ' ', bool removeLeadingSpace = false) =>
		input.Length == 0
			? throw new EmptyInputIsNotAllowed()
			: new RangeEnumerator(input, splitter, removeLeadingSpace);

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static ReadOnlySpan<char> GetSpanFromRange(this string input, Range range)
	{
		var (offset, length) = range.GetOffsetAndLength(input.Length);
		return input.AsSpan(offset, length);
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static Range GetOuterRange(this Range inner, (int, int) offsetAndLength)
	{
		var (elementOffset, length) = inner.GetOffsetAndLength(offsetAndLength.Item2);
		var start = offsetAndLength.Item1 + elementOffset;
		return new Range(start, start + length);
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static bool Compare(this ReadOnlySpan<char> first, ReadOnlySpan<char> second)
	{
		if (first.Length != second.Length)
			return false;
		for (var index = 0; index < first.Length; index++)
			if (first[index] != second[index])
				return false;
		return true;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static bool Any(this ReadOnlySpan<char> input, IEnumerable<string> items)
	{
		foreach (var item in items)
			if (input.Compare(item.AsSpan()))
				return true;
		return false;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static bool Contains(this ReadOnlySpan<char> input, IEnumerable<string> items)
	{
		foreach (var item in items)
			if (input.IndexOf(item.AsSpan()) >= 0)
				return true;
		return false;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static int Count(this ReadOnlySpan<char> input, char item)
	{
		var count = 0;
		foreach (var letter in input)
			if (letter == item)
				count++;
		return count;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static bool IsWordOrText(this ReadOnlySpan<char> input)
	{
		foreach (var c in input)
			if (c is (< 'A' or > 'Z') and (< 'a' or > 'z') && c != '"')
				return false;
		return true;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static bool IsTrueText(this ReadOnlySpan<char> input) =>
		input.Length == 4 && input[0] == 't' && input[1] == 'r' && input[2] == 'u' && input[3] == 'e';

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static bool IsFalseText(this ReadOnlySpan<char> input) =>
		input.Length == 5 && input[0] == 'f' && input[1] == 'a' && input[2] == 'l' &&
		input[3] == 's' && input[4] == 'e';
}