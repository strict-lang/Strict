using Strict.Language;

namespace Strict.Expressions;

/// <summary>
/// https://en.wikipedia.org/wiki/Shunting_yard_algorithm
/// Supports strict lists, generics, arguments, inner lists, groups (all the same brackets) and texts.
/// </summary>
public sealed class ShuntingYard
{
	private const int IsNotPrecedence = 7;

	public ShuntingYard(string input)
	{
		this.input = input;
		var tokenizer = new PhraseTokenizer(input);
		tokenizer.ProcessEachToken(PutTokenIntoStacks);
		ApplyHigherOrEqualPrecedenceOperators();
		if (Output.Count == 0)
			throw new NothingFound(input); //ncrunch: no coverage
	}

	public sealed class NothingFound(string input) : Exception("Nothing found! Should never happen. " + input);
	private readonly string input;

	private void PutTokenIntoStacks(Range tokenRange)
	{
		var (_, length) = tokenRange.GetOffsetAndLength(input.Length);
		if (length == 1)
			PutSingleCharacterTokenIntoStacks(tokenRange);
		else if (input.AsSpan()[tokenRange].IsMultiCharacterOperator())
		{
			ApplyHigherOrEqualPrecedenceOperators(GetOperatorPrecedence(tokenRange));
			operators.Push(tokenRange);
		}
		else
			Output.Push(tokenRange);
	}

	private int GetOperatorPrecedence(Range tokenRange)
	{
		var token = input.AsSpan()[tokenRange];
		return token.Compare(BinaryOperator.Is) && IsFollowedByNot(tokenRange)
			? BinaryOperator.GetPrecedence(BinaryOperator.In.AsSpan())
			: GetPrecedence(token);
	}

	private bool IsFollowedByNot(Range tokenRange)
	{
		var endIndex = tokenRange.End.Value;
		return endIndex + 5 <= input.Length &&
			input.AsSpan(endIndex, 5).Compare(" not ");
	}

	private void PutSingleCharacterTokenIntoStacks(Range tokenRange)
	{
		var firstCharacter = input[tokenRange.Start.Value];
		if (firstCharacter == PhraseTokenizer.OpenBracket)
			operators.Push(tokenRange);
		else if (firstCharacter == PhraseTokenizer.CloseBracket)
			ApplyHigherOrEqualPrecedenceOperators();
		else if (firstCharacter.IsSingleCharacterOperator())
		{
			ApplyHigherOrEqualPrecedenceOperators(BinaryOperator.GetPrecedence(firstCharacter));
			operators.Push(tokenRange);
		}
		else
			FlushCommaList(tokenRange, firstCharacter);
	}

	private void FlushCommaList(Range tokenRange, char firstCharacter)
	{
		// Comma lists always need to flush, happens when parsing inner elements via ParseListArguments
		if (firstCharacter == ',')
			ApplyHigherOrEqualPrecedenceOperators();
		Output.Push(tokenRange);
	}

	private readonly Stack<Range> operators = new();
	public Stack<Range> Output { get; } = new();

	private void ApplyHigherOrEqualPrecedenceOperators(int precedence = 0)
	{
		while (operators.Count > 0)
			if (!IsOpeningBracket(precedence) && GetTopOperatorPrecedence() >= precedence)
				AddOperatorToOutput();
			else
				return;
	}

	private int GetTopOperatorPrecedence()
	{
		var token = input.AsSpan()[operators.Peek()];
		return token.IsNot() && IsTopOperatorPartOfIsNot
			? IsNotPrecedence
			: GetPrecedence(token);
	}

	private bool IsTopOperatorPartOfIsNot
	{
		get
		{
			var skippedTopOperator = false;
			foreach (var operatorRange in operators)
				if (!skippedTopOperator)
					skippedTopOperator = true;
				else
					return input.AsSpan()[operatorRange].Compare(BinaryOperator.Is);
			return false;
		}
	}

	private static int GetPrecedence(ReadOnlySpan<char> token) =>
		token.IsNot()
			? 15
			: BinaryOperator.GetPrecedence(token);

	private void AddOperatorToOutput()
	{
		var newOperator = operators.Pop();
		if (input.AsSpan()[newOperator].IsNot() && operators.Count > 0 &&
			input.AsSpan()[operators.Peek()].Compare(BinaryOperator.Is))
		{
			var isOperator = operators.Pop();
			if (Output.Count == 0 || !input.AsSpan()[Output.Peek()].Compare(BinaryOperator.In))
				Output.Push(isOperator);
			Output.Push(newOperator);
			return;
		}
		if (input.AsSpan()[newOperator].Compare(BinaryOperator.Is))
		{
			if (Output.Count == 0 || !input.AsSpan()[Output.Peek()].Compare(BinaryOperator.In))
				Output.Push(newOperator);
		}
		else
			Output.Push(newOperator);
	}

	private bool IsOpeningBracket(int precedence)
	{
		if (input[operators.Peek().Start.Value] != PhraseTokenizer.OpenBracket)
			return false;
		if (precedence == 0)
			operators.Pop();
		return true;
	}
}
