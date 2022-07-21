using System;
using System.Collections.Generic;

namespace Strict.Language.Expressions;

/// <summary>
/// Optimized for speed and memory efficiency (no new), no memory is allocated except for the check
/// if we are in a list or just grouping things when a bracket is found. Passed onto ShuntingYard.
/// </summary>
// ReSharper disable MethodTooLong
// ReSharper disable ExcessiveIndentation
public sealed class PhraseTokenizer
{
	//public PhraseTokenizer(string input)
	//{
	//	if (input.Length == 0 || input[0] == ' ' || input[^1] == ' ' ||
	//		input.Contains("  ", StringComparison.Ordinal))
	//		throw new InvalidSpacing(input);
	//	if (input.Contains("()", StringComparison.Ordinal))
	//		throw new InvalidEmptyOrUnmatchedBrackets(input);
	//	this.input = input;
	//}

	//private readonly string input;
	private int index;
	private int textStart = -1;
	private int tokenStart = -1;

	public IEnumerable<Range> GetTokenRanges(ReadOnlySpan<char> input)
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
					yield return textStart..(index + 1);
					textStart = -1;
					tokenStart = -1;
				}
			}
			else if (textStart == -1)
				foreach (var token in GetNormalToken(input))
					yield return token;
		if (textStart != -1)
			throw new UnterminatedString(input.ToString());
		if (tokenStart >= 0)
			yield return tokenStart..;
	}

	private IEnumerable<Range> GetNormalToken(ReadOnlySpan<char> input)
	{
		if (input[index] == '(')
			foreach (var token in GetTokensTillMatchingClosingBracket(input))
				yield return token;
		else if (input[index] == ' ')
		{
			if (tokenStart >= 0)
				yield return tokenStart..index;
			tokenStart = -1;
		}
		else if (tokenStart == -1)
			tokenStart = index;
	}

	private IReadOnlyList<Range> GetTokensTillMatchingClosingBracket(ReadOnlySpan<char> input)
	{
		tokenStart = index + 1;
		var result = new List<Range> { index..(index + 1) };
		var foundListSeparator = false;
		for (index++; index < input.Length; index++)
			if (input[index] == ')')
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
				foreach (var token in GetNormalToken(input))
					result.Add(token);
		if (result.Count < 3)
			throw new InvalidEmptyOrUnmatchedBrackets(input.ToString());
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