using System;

namespace Strict.Language;

public ref struct SpanSplitEnumerator
{
	public SpanSplitEnumerator(ReadOnlySpan<char> input, char splitter, StringSplitOptions options)
	{
		this.input = input;
		this.splitter = splitter;
		this.options = options;
		offset = 0;
		Current = default;
	}

	private readonly ReadOnlySpan<char> input;
	private readonly char splitter;
	private readonly StringSplitOptions options;
	private int offset;
	public ReadOnlySpan<char> Current { get; private set; }
	public readonly SpanSplitEnumerator GetEnumerator() => this;

	public bool MoveNext()
	{
		if (offset >= input.Length)
			return false;
		for (var index = offset; index < input.Length; index++)
			if (input[index] == splitter)
				return GetWordBeforeSplitter(index);
		Current = options == StringSplitOptions.TrimEntries
			? input[offset..].Trim()
			: input[offset..];
		offset = input.Length;
		return true;
	}

	private bool GetWordBeforeSplitter(int index)
	{
		if (index == offset)
			throw new InvalidConsecutiveSplitter(input, index);
		if (index + 1 == input.Length)
			throw new EmptyEntryNotAllowedAtTheEnd(input, index);
		Current = options == StringSplitOptions.TrimEntries
			? input[offset..index].Trim()
			: input[offset..index];
		offset = index + 1;
		return true;
	}

	public sealed class InvalidConsecutiveSplitter : Exception
	{
		public InvalidConsecutiveSplitter(ReadOnlySpan<char> input, int index) : base("Input=" +
			input.ToString() + ", index=" + index) { }
	}

	public sealed class EmptyEntryNotAllowedAtTheEnd : Exception
	{
		public EmptyEntryNotAllowedAtTheEnd(ReadOnlySpan<char> input, int index) : base("Input=" +
			input.ToString() + ", index=" + index) { }
	}
}