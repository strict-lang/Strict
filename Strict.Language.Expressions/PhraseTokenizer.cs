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
	public PhraseTokenizer(string input)
	{
		var part = input.AsSpan();
		if (part.Length < 3)
		{
			if (part[0] == ' ')
				throw new InvalidSpacing(input);
			if (part[0] == '\"' && (part.Length == 1 || part[1] != '\"'))
				throw new UnterminatedString(input);
			if (part[0] == '(')
				throw new InvalidEmptyOrUnmatchedBrackets(input);
			throw new NotSupportedException("Input should never be this small: " + input);
		}
#if LOG_DETAILS
		Logger.Info("* " + nameof(PhraseTokenizer) + ": " + input);
#endif
		if (part.Length == 0 || part[0] == ' ' || part[^1] == ' ' ||
			part.Contains("  ", StringComparison.Ordinal))
			throw new InvalidSpacing(input);
		if (part.Contains("()", StringComparison.Ordinal))
			throw new InvalidEmptyOrUnmatchedBrackets(input);
		this.input = input;
	}

	private readonly string input;

	public void ProcessEachToken(Action<Range> processToken)
	{
		textStart = -1;
		tokenStart = -1;
		for (index = 0; index < input.Length; index++)
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
			throw new UnterminatedString(input);
		if (tokenStart >= 0)
			processToken(tokenStart..input.Length);
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

				// ReSharper disable once ComplexConditionExpression
				else if (input[index - 1] == 's' && input[index..].Length > 4 && input[(index + 1)..(index + 5)] == "not ")
				{
					processToken(tokenStart..(index + 4));
					index += 4;
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
		if (tokenStart < 0)
			tokenStart = index;
		// It is very important to catch everything before the opening bracket as well. However if this
		// is a binary expression we want to catch, split the initial range as its own method call.
		var result = new List<Range>();
		if (index > tokenStart)
			result.Add(tokenStart..index);
		result.Add(index..(index + 1));
		tokenStart = index + 1;
		var foundListSeparator = false;
		var foundNoSpace = true;
		for (index++; index < input.Length; index++)
			if (input[index] == CloseBracket)
			{
				if (tokenStart >= 0)
					result.Add(tokenStart..index);
				tokenStart = -1;
				result.Add(index..(index + 1));
				break;
			}
			else if (input[index] == ',' || input[index] == '?')
			{
				foundListSeparator = true;
				result.Add(index..(index + 1));
			}
			else
			{
				if (input[index] == ' ')
					foundNoSpace = false;
				ProcessNormalToken(result.Add);
			}
		if (result.Count < 3)
			throw new InvalidEmptyOrUnmatchedBrackets(input);
		if (result.Count == 3 || foundListSeparator || foundNoSpace)
			return MergeAllTokensIntoSingleList(result);
		return result;
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