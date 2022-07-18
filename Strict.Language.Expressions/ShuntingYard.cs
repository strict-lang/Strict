using System;
using System.Collections.Generic;

namespace Strict.Language.Expressions;

/// <summary>
/// https://en.wikipedia.org/wiki/Shunting_yard_algorithm
/// </summary>
public sealed class ShuntingYard
{
	public ShuntingYard(IEnumerable<string> tokens)//TODO: ReadOnlyMemory<char> tokens should be used here
	{
		foreach (var token in tokens)
			PutTokenIntoStacks(token);
		ApplyHigherOrEqualPrecedenceOperators();
		//TODO: remove after done:
		Console.WriteLine("Operators: " + string.Join(", ", operators) + " Output: " + string.Join(", ", Output));
	}

	private void PutTokenIntoStacks(string token)
	{
		if (token[0] == '(')
			operators.Push(token);
		else if (token[0] == ')')
			ApplyHigherOrEqualPrecedenceOperators();
		else if ("+-*/".Contains(token[0]))
		{
			ApplyHigherOrEqualPrecedenceOperators(GetPrecedence(token));
			operators.Push(token);
		}
		else
			Output.Push(token);
		//Console.WriteLine("Consumed " + token + " Operators: " + string.Join(", ", operators) + " Output: " + string.Join(", ", Output));
	}

	private readonly Stack<string> operators = new();
	public Stack<string> Output { get; } = new();

	private void ApplyHigherOrEqualPrecedenceOperators(int precedence = 0)
	{
		while (operators.Count > 0)
			if (!IsOpeningBracket(precedence) && GetPrecedence(operators.Peek()) >= precedence)
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

	private static int GetPrecedence(string token) =>
		token switch
		{
			"+" => 1,
			"-" => 1,
			"*" => 2,
			"/" => 2,
			_ => throw new NotSupportedException(token) //ncrunch: no coverage
		};
}