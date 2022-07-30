using System;
using System.Collections.Generic;

namespace Strict.Language.Expressions;

/// <summary>
/// https://en.wikipedia.org/wiki/Shunting_yard_algorithm
/// Supports strict lists, generics, arguments, inner lists, groups (all the same brackets) and texts.
/// </summary>
public sealed class ShuntingYard
{
	public ShuntingYard(string input, Range partToParse)
	{
		this.input = input;
		var tokenizer = new PhraseTokenizer(input, partToParse);
		tokenizer.ProcessEachToken(PutTokenIntoStacks);
		ApplyHigherOrEqualPrecedenceOperators();
		if (Output.Count == 0)
			throw new NotSupportedException("Nothing found! Should never happen."); //ncrunch: no coverage
		//TODO: remove after done:
		Console.WriteLine("Operators: " + string.Join(", ", operators) + " Output Ranges: " + string.Join(", ", Output));
	}

	private readonly string input;

	private void PutTokenIntoStacks(Range tokenRange)
	{
		var (_, length) = tokenRange.GetOffsetAndLength(input.Length);
		Console.WriteLine(nameof(PutTokenIntoStacks) + ": " + input[tokenRange]);
		if (length == 1)
			PutSingleCharacterTokenIntoStacks(tokenRange);
		else
		{
			var token = input[tokenRange];
			if (token.IsMultiCharacterOperator())
			{
				ApplyHigherOrEqualPrecedenceOperators(BinaryOperator.GetPrecedence(token));
				operators.Push(tokenRange);
			}
			else
				Output.Push(tokenRange);
		}
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
		{
			// comma lists always need to flush everything out, this only happens when parsing inner elements via ParseListArguments
			if (firstCharacter == ',')
				ApplyHigherOrEqualPrecedenceOperators();
			Output.Push(tokenRange);
		}
	}

	private readonly Stack<Range> operators = new();
	public Stack<Range> Output { get; } = new();

	private void ApplyHigherOrEqualPrecedenceOperators(int precedence = 0)
	{
		while (operators.Count > 0)
			if (!IsOpeningBracket(precedence) && BinaryOperator.GetPrecedence(input.GetSpanFromRange(operators.Peek())) >= precedence)
				Output.Push(operators.Pop());
			else
				return;
	}

	// 5 + 3
	// ^ first operand (left) Range
	//   ^ operator (range??????????)
	//     ^ right Range

	private bool IsOpeningBracket(int precedence)
	{
		if (input[operators.Peek().Start.Value] != PhraseTokenizer.OpenBracket)
			return false;
		if (precedence == 0)
			operators.Pop();
		return true;
	}
}
