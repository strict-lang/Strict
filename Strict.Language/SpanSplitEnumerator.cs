namespace Strict.Language;

public ref struct SpanSplitEnumerator(ReadOnlySpan<char> input, char splitter,
	StringSplitOptions options)
{
	//ncrunch: no coverage start, for performance reasons disabled here
	private readonly ReadOnlySpan<char> input = input;
	private int offset = 0;
	private int bracketDepth = 0;
  private bool isInsideText = false;
	public ReadOnlySpan<char> Current { get; private set; } = default;
	public readonly SpanSplitEnumerator GetEnumerator() => this;

	public bool MoveNext()
	{
		if (offset >= input.Length)
			return false;
		var wordBeforeSplitter = GetWordBeforeSplitterAndTrackBrackets();
		if (wordBeforeSplitter != null)
			return wordBeforeSplitter.Value;
		Current = options == StringSplitOptions.TrimEntries
			? input[offset..].Trim()
			: input[offset..];
		offset = input.Length;
		return true;
	}

	private bool? GetWordBeforeSplitterAndTrackBrackets()
	{
		for (var index = offset; index < input.Length; index++)
		{
      if (input[index] == '"' && !IsEscapedQuote(index))
			{
				if (isInsideText && index + 1 < input.Length && input[index + 1] == '"')
					index++;
				else
					isInsideText = !isInsideText;
			}
			else if (!isInsideText && input[index] == '(')
				bracketDepth++;
     else if (!isInsideText && input[index] == ')')
				bracketDepth--;
      else if (!isInsideText && input[index] == splitter &&
				(splitter != ',' || bracketDepth == 0))
				return GetWordBeforeSplitter(index);
		}
		return null;
	}

	private bool IsEscapedQuote(int quoteIndex)
	{
		if (quoteIndex == 0 || input[quoteIndex - 1] != '\\')
			return false;
		var slashCount = 0;
		for (var slashIndex = quoteIndex - 1; slashIndex >= 0 && input[slashIndex] == '\\'; slashIndex--)
			slashCount++;
		return slashCount % 2 == 1;
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

	public sealed class InvalidConsecutiveSplitter(string input, int index)
		: Exception("Input=" + input + ", index=" + index);

	public sealed class EmptyEntryNotAllowedAtTheEnd(string input, int index)
		: Exception("Input=" + input + ", index=" + index);
}