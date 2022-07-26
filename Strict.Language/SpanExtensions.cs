﻿using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;

namespace Strict.Language;

/// <summary>
/// Used for parsing Strict code, which needs to be exactly in the correct format
/// </summary>
public static class SpanExtensions
{
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

	public static Range GetOuterRange(this Range inner, (int, int) offsetAndLength)
	{
		var (elementOffset, length) = inner.GetOffsetAndLength(offsetAndLength.Item2);
		var start = offsetAndLength.Item1 + elementOffset;
		return new Range(start, start + length);
	}

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

	public static bool IsWordOrText(this ReadOnlySpan<char> input)
	{
		foreach (var c in input)
			if (c is (< 'A' or > 'Z') and (< 'a' or > 'z') && c != '"')
				return false;
		return true;
	}
}

/// <summary>
/// Instead of using one of String or other Split methods here, use this and SpanExtensions to
/// avoid allocating new memory on every split, especially in the tokenizer and parser.
/// </summary>
public ref struct RangeEnumerator
{
	public RangeEnumerator(ReadOnlySpan<char> input, char splitter, bool removeLeadingSpace = false)
	{
		this.input = input;
		this.splitter = splitter;
		this.removeLeadingSpace = removeLeadingSpace;
	}

	private readonly ReadOnlySpan<char> input;
	private readonly char splitter;
	private readonly bool removeLeadingSpace;
	private int offset = 0;
	public Range Current { get; private set; } = default;
	public readonly RangeEnumerator GetEnumerator() => this;

	public bool MoveNext()
	{
		if (offset >= input.Length)
			return false;
		for (var index = offset; index < input.Length; index++)
			if (input[index] == splitter)
				return GetWordBeforeSplitter(index);
		if (removeLeadingSpace && input[offset] == ' ')
			offset++;
		Current = offset..;
		offset = input.Length;
		return true;
	}

	private bool GetWordBeforeSplitter(int index)
	{
		if (index == offset)
			throw new SpanSplitEnumerator.InvalidConsecutiveSplitter(input.ToString(), index);
		if (index + 1 == input.Length)
			throw new SpanSplitEnumerator.EmptyEntryNotAllowedAtTheEnd(input.ToString(), index);
		if (removeLeadingSpace && input[offset] == ' ')
			offset++;
		Current = offset..index;
		offset = index + 1;
		return true;
	}
}

/*tst
public static class MemoryExtensions
{
	public static StringRangeEnumerator SplitRanges(this ReadOnlyMemory<char> input, char splitter = ' ',
		StringSplitOptions options = StringSplitOptions.None)
	{
		SpanExtensions.CheckParameters(input.Length, options);
		return new StringRangeEnumerator(input, splitter, options);
	}
}
*
/// <summary>
/// Instead of using one of String or other Split methods here, use this and SpanExtensions to
/// avoid allocating new memory on every split, especially in the tokenizer and parser.
/// </summary>
public ref struct StringRangeEnumerator// : IEnumerable<Range>
{
	public StringRangeEnumerator(ReadOnlySpan<char> input, char splitter)
	{
		this.input = input;
		this.splitter = splitter;
	}

	private readonly ReadOnlySpan<char> input;
	private readonly char splitter;

	public IEnumerator<Range> GetEnumerator()
	{
		if (offset >= input.Length)
			yield break;
		var start = offset;
		for (; offset < input.Length; offset++)
			if (input[offset] == splitter)
				yield return start..offset++;
		yield return start..;
	}

	private int offset = 0;
	//IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
}
*/