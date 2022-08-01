using System;

namespace Strict.Language;

/// <summary>
/// Instead of using one of String or other Split methods here, use this and SpanExtensions to
/// avoid allocating new memory on every split, especially in the tokenizer and parser.
/// </summary>
public ref struct RangeEnumerator
{
	//ncrunch: no coverage start, for performance reasons disabled here
	public RangeEnumerator(ReadOnlySpan<char> input, char splitter, Index outerRangeOffset)
	{
		this.input = input;
		this.splitter = splitter;
		removeLeadingSpace = splitter == ',';
		outerStart = outerRangeOffset.Value;
	}

	private readonly ReadOnlySpan<char> input;
	private readonly char splitter;
	private readonly bool removeLeadingSpace = false;
	private readonly int outerStart = 0;
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
		Current = (outerStart + offset)..(outerStart + input.Length);
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
		Current = (outerStart + offset)..(outerStart + index);
		offset = index + 1;
		return true;
	}
}