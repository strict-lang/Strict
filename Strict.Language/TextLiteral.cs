namespace Strict.Language;

/// <summary>
/// Scans over code skip text literals, an escaped quote or backslash inside a literal does not end it.
/// </summary>
public static class TextLiteral
{
	public static bool Advance(ReadOnlySpan<char> input, ref int index, bool isInText)
	{
		if (isInText && input[index] == '\\')
		{
			index++;
			return true;
		}
		return input[index] == '"'
			? !isInText
			: isInText;
	}
}
