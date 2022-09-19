using System;
using System.Collections.Generic;
using System.Linq;
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
	public static Range RemoveFirstAndLast(this Range range, int outerLength)
	{
		var (offset, length) = range.GetOffsetAndLength(outerLength);
		return new Range(offset + 1, offset + length - 1);
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
	public static bool IsWord(this ReadOnlySpan<char> input)
	{
		foreach (var c in input)
			if (c is (< 'A' or > 'Z') and (< 'a' or > 'z'))
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

	/// <summary>
	/// Heavily optimized number parsing, which can be 10 times faster than int.TryParse and 50
	/// times faster than double.TryParse.
	/// </summary>
	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	// ReSharper disable once MethodTooLong
	// ReSharper disable once ExcessiveIndentation
	public static bool TryParseNumber(this ReadOnlySpan<char> input, out double number)
	{
		if (input[0] == '-')
		{
			if (!TryParseNumber(input[1..], out number))
				return false;
			number = -number;
			return true;
		}
		number = input[0] - '0';
		// Trick from char.IsDigit, which calls char.IsBetween using uint to do an in between check
		if ((uint)number > 9)
			return false;
		var decimalPosition = input.Length;
		for (var index = 1; index < input.Length; index++)
		{
			var letter = input[index];
			var digit = (uint)(letter - '0');
			if (digit <= 9)
				number = number * 10 + digit;
			else if (letter == '.')
				decimalPosition = index + 1;
			else if (letter == 'e' && index + 2 < input.Length)
			{
				var existingExponent = decimalPosition == input.Length
					? 0
					: index - decimalPosition;
				index++;
				var exponentSign = input[index] == '-'
					? -1
					: input[index] == '+'
						? 1
						: 0;
				if (exponentSign == 0)
					exponentSign = 1;
				else
					index++;
				if (!TryParseNumber(input[index..], out var exponent))
					return false;
				exponent = exponentSign * exponent - existingExponent;
				number *= Math.Pow(10, exponent);
				return true;
			}
			// Ignore any extra letter after number, abort if this is not the very end and parsing failed
			else if (index + 1 < input.Length)
				return false;
		}
		if (decimalPosition < input.Length)
			number /= Math.Pow(10, input.Length - decimalPosition);
		return true;
	}

	public static IEnumerable<string> GetTextsFromRanges(this ReadOnlySpan<char> input, IEnumerable<Range> ranges)
	{
		var count = ranges.Count();
		var texts = new string[count];
		for (var index = 0; index < count; index++)
			texts[index] = input[ranges.ElementAt(index)].ToString();
		return texts;
	}

	// ReSharper disable once ExcessiveIndentation
	public static int FindMatchingBracketIndex(this ReadOnlySpan<char> input, int startIndex)
	{
		if (startIndex < 0)
			return -1;
		var bracketCount = 1;
		for (var index = startIndex + 1; index < input.Length; index++)
			if (input[index] == '(')
				bracketCount++;
			else if (input[index] == ')')
			{
				bracketCount--;
				if (bracketCount == 0)
					return index;
			}
		return -1;
	}
}