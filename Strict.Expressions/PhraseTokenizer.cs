using Strict.Language;

namespace Strict.Expressions;

/// <summary>
/// Phrases are any expressions containing spaces (if not, they are just single expressions and
/// don't need any tokenizing). They could come from a full line of code, conditions of ifs, the
/// right part of assignments or method call arguments. Optimized for speed and memory efficiency
/// (no new), no memory is allocated except for the check if we are in a list or just grouping.
/// </summary>
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
		textStart = tokenStart = -1;
		for (index = 0; index < input.Length; index++)
			if (input[index] == '\"')
				GetSingleTokenTillEndOfText(processToken);
			else if (textStart == -1)
				ProcessNormalToken(processToken);
		if (textStart != -1)
			throw new UnterminatedString(input);
		if (tokenStart >= 0)
			processToken(tokenStart..input.Length);
	}

	private void GetSingleTokenTillEndOfText(Action<Range> processToken)
	{
		if (textStart == -1)
			textStart = index;
		else if (index + 1 < input.Length && input[index + 1] == '\"')
			index++; // the next character is still a text (double quote), continue text
		else
		{
			processToken(textStart..(index + 1));
			textStart = -1;
			tokenStart = -1;
		}
	}

	private int index;
	private int textStart = -1;
	private int tokenStart = -1;

	private void ProcessNormalToken(Action<Range> processToken)
	{
		if (input[index] == OpenBracket)
			foreach (var token in new TokensTillMatchingBracketGrabber(this).GetRanges())
				processToken(token);
		else if (input[index] == ' ')
		{
			if (tokenStart >= 0)
				ProcessTokenAfterSpace(processToken);
			tokenStart = -1;
		}
		else if (tokenStart == -1)
			tokenStart = index;
	}

	private void ProcessTokenAfterSpace(Action<Range> processToken)
	{
		// If our previous character was a space, and not alone, we parsed it as one token (outside
		// a list element); it needs to be split into two tokens like for complex cases so processing
		// elements works in ParseListElements
		if (index > tokenStart + 1 && input[index - 1] == ',')
		{
			processToken(tokenStart..(index - 1));
			processToken((index - 1)..index);
		}
		/*TODO: remove, but we still need to handle "is not, is in, is not in"
		else if (input.IsMultiCharacterOperatorWithSpace(index, out var tokenEnd))
		{
			processToken(tokenStart..(index + tokenEnd));
			index += tokenEnd;
		}
		*/
		else
			processToken(tokenStart..index);
	}

	internal const char OpenBracket = '(';
	internal const char CloseBracket = ')';

	/// <summary>
	/// It is very important to catch everything before the opening bracket as well. However, if
	/// this is a binary expression we want to catch, split the initial range as its own method call.
	/// </summary>
	private sealed class TokensTillMatchingBracketGrabber
	{
		public TokensTillMatchingBracketGrabber(PhraseTokenizer tokens)
		{
			this.tokens = tokens;
			if (tokens.tokenStart < 0)
				tokens.tokenStart = tokens.index;
			if (tokens.index > tokens.tokenStart)
				result.Add(tokens.tokenStart..tokens.index);
			result.Add(tokens.index..(tokens.index + 1));
			tokens.tokenStart = tokens.index + 1;
		}

		private readonly PhraseTokenizer tokens;
		private readonly List<Range> result = [];

		public IReadOnlyList<Range> GetRanges()
		{
			for (tokens.index++; tokens.index < tokens.input.Length; tokens.index++)
				if (FoundLastToken())
					break;
			if (tokens.textStart != -1)
				throw new UnterminatedString(tokens.input);
			if (result.Count < 3)
				throw new InvalidEmptyOrUnmatchedBrackets(tokens.input);
			if (result.Count == 3 || foundListSeparator || !foundSpace ||
				tokens.input[tokens.index - 1] == ' ' || foundBinaryOperationInMethodCall)
				return MergeAllTokensIntoSingleList(result);
			return result;
		}

		private bool FoundLastToken()
		{
			if (tokens.input[tokens.index] == '\"')
			{
				tokens.GetSingleTokenTillEndOfText(result.Add);
				return false;
			}
			return tokens.textStart == -1 && (tokens.input[tokens.index] == CloseBracket
				? HandleCloseBracket()
				: HandleListSeparator() && HandleMethodCall());
		}

		private bool HandleCloseBracket()
		{
			if (tokens.tokenStart >= 0)
				result.Add(tokens.tokenStart..tokens.index);
			tokens.tokenStart = -1;
			result.Add(tokens.index..(tokens.index + 1));
			return tokens.index + 1 < tokens.input.Length &&
				// Consume a nested member or method call as a single token
				tokens.input[tokens.index + 1] != '.';
		}

		private bool HandleListSeparator()
		{
			if (tokens.input[tokens.index] != ',' && tokens.input[tokens.index] != '?')
				return true;
			foundListSeparator = true;
			result.Add(tokens.index..(tokens.index + 1));
			return false;
		}

		private bool HandleMethodCall()
		{
			HandleMethodCallStates();
			tokens.ProcessNormalToken(result.Add);
			return isInMethodCall && (tokens.index + 1 < tokens.input.Length &&
				tokens.input[tokens.index] == CloseBracket &&
				(tokens.input[tokens.index + 1] != '.' || foundBinaryOperationInMethodCall) ||
				tokens.MemberOrMethodCallWithNoArguments() && !foundBinaryOperationInMethodCall);
		}

		private void HandleMethodCallStates()
		{
			if (tokens.input[tokens.index - 1] == '.')
				isInMethodCall = true;
			if (tokens.input[tokens.index - 1] == OpenBracket && tokens.index > 2 &&
				tokens.input[tokens.index - 2] != ' ')
				foundOpeningBracketForMethodCall = true;
			if (tokens.input[tokens.index] == ' ')
				foundSpace = true;
			if (foundOpeningBracketForMethodCall &&
				tokens.input[tokens.index].IsSingleCharacterOperator())
				foundBinaryOperationInMethodCall = true;
		}

		private bool foundListSeparator;
		private bool isInMethodCall;
		private bool foundSpace;
		private bool foundOpeningBracketForMethodCall;
		private bool foundBinaryOperationInMethodCall;
	}

	private bool MemberOrMethodCallWithNoArguments() =>
		index > 0 && input[index - 1] == ' ' && input[index - 2] != ',';

	private static Range[] MergeAllTokensIntoSingleList(List<Range> result) =>
	[
		result[0].Start..result[^1].End
	];

	public sealed class UnterminatedString(string input) : Exception(input);
	public sealed class InvalidSpacing(string input) : Exception(input);
	public class InvalidEmptyOrUnmatchedBrackets(string input) : Exception(input);
}