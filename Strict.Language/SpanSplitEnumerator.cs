namespace Strict.Language;

public ref struct SpanSplitEnumerator
{
	//ncrunch: no coverage start, for performance reasons disabled here
	public SpanSplitEnumerator(ReadOnlySpan<char> input, char splitter, StringSplitOptions options)
	{
		this.input = input;
		this.splitter = splitter;
		this.options = options;
	}

	private readonly ReadOnlySpan<char> input;
	private readonly char splitter;
	private readonly StringSplitOptions options;
	private int offset = 0;
	public ReadOnlySpan<char> Current { get; private set; } = default;
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
			throw new InvalidConsecutiveSplitter(input.ToString(), index);
		if (index + 1 == input.Length)
			throw new EmptyEntryNotAllowedAtTheEnd(input.ToString(), index);
		Current = options == StringSplitOptions.TrimEntries
			? input[offset..index].Trim()
			: input[offset..index];
		offset = index + 1;
		return true;
	}

	public sealed class InvalidConsecutiveSplitter : Exception
	{
		public InvalidConsecutiveSplitter(string input, int index) : base("Input=" +
			input + ", index=" + index) { }
	}

	public sealed class EmptyEntryNotAllowedAtTheEnd : Exception
	{
		public EmptyEntryNotAllowedAtTheEnd(string input, int index) : base("Input=" +
			input + ", index=" + index) { }
	}
}