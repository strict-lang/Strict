using System;
using System.Collections.Generic;

namespace Strict.Language.Expressions;

/// <summary>
/// Phrases are any expressions containing spaces (if not they are just single expressions and
/// don't need any tokenizing). They could come from a full line of code, conditions of ifs,
/// right part of assignments or method call arguments. Optimized for speed and memory efficiency
/// (no new), no memory is allocated except for the check if we are in a list or just grouping.
/// </summary>
// ReSharper disable MethodTooLong
// ReSharper disable ExcessiveIndentation
//ncrunch: no coverage start, for better performance
public sealed class PhraseTokenizer
{
	public PhraseTokenizer(string input, Range partToParse)
	{
		var part = input.GetSpanFromRange(partToParse);
		if (part.Length < 3)
		{
			if (part[0] == ' ')
				throw new InvalidSpacing(part.ToString());
			if (part[0] == '\"' && (part.Length == 1 || part[1] != '\"'))
				throw new UnterminatedString(part.ToString());
			if (part[0] == '(')
				throw new InvalidEmptyOrUnmatchedBrackets(part.ToString());
			throw new NotSupportedException("Input should never be this small: " + part.ToString());
		}
#if LOG_DETAILS
		Logger.Info("* " + nameof(PhraseTokenizer) + ": " + part.ToString());
#endif
		if (part.Length == 0 || part[0] == ' ' || part[^1] == ' ' ||
			part.Contains("  ", StringComparison.Ordinal))
			throw new InvalidSpacing(input);
		if (part.Contains("()", StringComparison.Ordinal))
			throw new InvalidEmptyOrUnmatchedBrackets(input);
		this.input = input;
		this.partToParse = partToParse;
	}

	private readonly string input;
	private readonly Range partToParse;

	public void ProcessEachToken(Action<Range> processToken)
	{
		textStart = -1;
		tokenStart = -1;
		var (offset, length) = partToParse.GetOffsetAndLength(input.Length);
		for (index = offset; index < offset + length; index++)
			if (input[index] == '\"')
			{
				if (textStart == -1)
					textStart = index;
				else if (index + 1 < input.Length && input[index + 1] == '\"')
					index++; // next character is still a text (double quote), continue text
				else
				{
					processToken(textStart..(index + 1));
					textStart = -1;
					tokenStart = -1;
				}
			}
			else if (textStart == -1)
				ProcessNormalToken(processToken);
		if (textStart != -1)
			throw new UnterminatedString(input[partToParse]);
		if (tokenStart >= 0)
			processToken(tokenStart..(offset + length));
	}

	private int index;
	private int textStart = -1;
	private int tokenStart = -1;

	private void ProcessNormalToken(Action<Range> processToken)
	{
		if (input[index] == OpenBracket)
			foreach (var token in GetTokensTillMatchingClosingBracket())
				processToken(token);
		else if (input[index] == ' ')
		{
			if (tokenStart >= 0)
			{
				// If our previous character was a , and not alone we parsed it as one token (outside list
				// element), it needs to be split into two tokens like for complex cases so processing
				// elements works in ParseListElements
				if (index > tokenStart + 1 && input[index - 1] == ',')
				{
					processToken(tokenStart..(index - 1));
					processToken((index - 1)..index);
				}
				else
					processToken(tokenStart..index);
			}
			tokenStart = -1;
		}
		else if (tokenStart == -1)
			tokenStart = index;
	}

	internal const char OpenBracket = '(';
	internal const char CloseBracket = ')';

	private IReadOnlyList<Range> GetTokensTillMatchingClosingBracket()
	{
		tokenStart = index + 1;
		var result = new List<Range> { index..(index + 1) };
		var foundListSeparator = false;
		for (index++; index < input.Length; index++)
			if (input[index] == CloseBracket)
			{
				if (tokenStart >= 0)
					result.Add(tokenStart..index);
				tokenStart = -1;
				result.Add(index..(index + 1));
				break;
			}
			else if (input[index] == ',')
			{
				foundListSeparator = true;
				result.Add(index..(index + 1));
			}
			else
				ProcessNormalToken(result.Add);
		if (result.Count < 3)
			throw new InvalidEmptyOrUnmatchedBrackets(input[partToParse]);
		return result.Count == 3 || foundListSeparator
			? MergeAllTokensIntoSingleList(result)
			: result;
	}

	private static Range[] MergeAllTokensIntoSingleList(List<Range> result) =>
		new[] { result[0].Start..result[^1].End };

	public sealed class UnterminatedString : Exception
	{
		public UnterminatedString(string input) : base(input) { }
	}

	public class InvalidSpacing : Exception
	{
		public InvalidSpacing(string input) : base(input) { }
	}

	public class InvalidEmptyOrUnmatchedBrackets : Exception
	{
		public InvalidEmptyOrUnmatchedBrackets(string input) : base(input) { }
	}
}