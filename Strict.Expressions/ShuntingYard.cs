using Strict.Language;

namespace Strict.Expressions;

/// <summary>
/// https://en.wikipedia.org/wiki/Shunting_yard_algorithm
/// Supports strict lists, generics, arguments, inner lists, groups (all the same brackets) and texts.
/// </summary>
public sealed class ShuntingYard
{
	public ShuntingYard(string input)
	{
		this.input = input;
		var tokenizer = new PhraseTokenizer(input);
		tokenizer.ProcessEachToken(PutTokenIntoStacks);
		ApplyHigherOrEqualPrecedenceOperators();
		if (Output.Count == 0)
			throw new NotSupportedException("Nothing found! Should never happen."); //ncrunch: no coverage
	}

	private readonly string input;

	private void PutTokenIntoStacks(Range tokenRange)
	{
		var (_, length) = tokenRange.GetOffsetAndLength(input.Length);
		if (length == 1)
			PutSingleCharacterTokenIntoStacks(tokenRange);
		else if (input.AsSpan()[tokenRange].IsMultiCharacterOperator())
		{
			ApplyHigherOrEqualPrecedenceOperators(GetPrecedence(input.AsSpan()[tokenRange]));
			operators.Push(tokenRange);
		}
		else
			Output.Push(tokenRange);
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
		return token.IsNot() && IsNotPartOfIsNot
			? BinaryOperator.GetPrecedence(token)
			: GetPrecedence(token);
	}

	private bool IsNotPartOfIsNot
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
		// If "is not" or "is not in" was parsed, flip them to make BinaryExpression parsing easier
		if (input.AsSpan()[newOperator].Compare(BinaryOperator.Is))
		{
			if (input.AsSpan()[Output.Peek()].Compare(UnaryOperator.Not))
			{
				var notRange = Output.Pop();
				if (!input.AsSpan()[Output.Peek()].Compare(BinaryOperator.In))
					Output.Push(newOperator);
				Output.Push(notRange);
			}
			else if (!input.AsSpan()[Output.Peek()].Compare(BinaryOperator.In))
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
