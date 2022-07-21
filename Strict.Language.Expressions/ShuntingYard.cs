using System;
using System.Collections.Generic;

namespace Strict.Language.Expressions;

/// <summary>
/// https://en.wikipedia.org/wiki/Shunting_yard_algorithm
/// </summary>
public sealed class ShuntingYard
{
	public ShuntingYard(ReadOnlySpan<char> input) //TODO: ReadOnlyMemory<char> tokens should be used here
	{
		foreach (var token in new PhraseTokenizer().GetTokenRanges(input))
			PutTokenIntoStacks(token, input);
		ApplyHigherOrEqualPrecedenceOperators();
		//TODO: remove after done:
		Console.WriteLine("Operators: " + string.Join(", ", operators) + " Output: " + string.Join(", ", Output));
	}

	//public string Input { get; }

	// ReSharper disable once ExcessiveIndentation
	// ReSharper disable once MethodTooLong
	private void PutTokenIntoStacks(Range tokenRange, ReadOnlySpan<char> input)
	{
		if (tokenRange.End.Value - tokenRange.Start.Value == 1)
		{
			var firstCharacter = input[tokenRange.Start.Value];
			if (firstCharacter == '(')
				operators.Push(OpenBracket);
			else if (firstCharacter == ')')
				ApplyHigherOrEqualPrecedenceOperators();
			else if (firstCharacter.IsSingleCharacterOperator())
			{
				ApplyHigherOrEqualPrecedenceOperators(BinaryOperator.GetPrecedence(firstCharacter));
				operators.Push(firstCharacter.ToString());
			}
			else
				Output.Push(firstCharacter.ToString());
		}
		else
		{
			var token = input[tokenRange].ToString();
			if (token.IsMultiCharacterOperator())
			{
				ApplyHigherOrEqualPrecedenceOperators(BinaryOperator.GetPrecedence(token));
				operators.Push(token);
			}
			else
				Output.Push(token);
		}
		//Console.WriteLine("Consumed " + tokenRange + " Operators: " + string.Join(", ", operators) + " Output: " + string.Join(", ", Output));
	}

	private const string OpenBracket = "(";
	private const string CloseBracket = ")";
	private readonly Stack<string> operators = new();
	public Stack<string> Output { get; } = new();

	private void ApplyHigherOrEqualPrecedenceOperators(int precedence = 0)
	{
		while (operators.Count > 0)
			if (!IsOpeningBracket(precedence) && BinaryOperator.GetPrecedence(operators.Peek()) >= precedence)
				Output.Push(operators.Pop());
			else
				return;
	}

	private bool IsOpeningBracket(int precedence)
	{
		if (operators.Peek() != "(")
			return false;
		if (precedence == 0)
			operators.Pop();
		return true;
	}
}
