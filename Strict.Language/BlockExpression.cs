using System.Collections.Generic;
using System;

namespace Strict.Language;

public class BlockExpression : Expression
{
	public BlockExpression(Type returnType, IReadOnlyList<Expression> expressions) :
		base(returnType) =>
		Expressions = expressions;

	public IReadOnlyList<Expression> Expressions { get; }
	/// <summary>
	/// Dictionaries are slow and eats up a lot of memory, only created when needed.
	/// </summary>
	private Dictionary<string, Expression>? variables;

	public void AddVariable(string name, Expression value)
	{
		variables ??= new Dictionary<string, Expression>(StringComparer.Ordinal);
		variables.Add(name, value);
	}

	public Expression? FindVariableValue(ReadOnlySpan<char> searchFor)
	{
		if (variables == null)
			return null;
		foreach (var (name, value) in variables)
			if (searchFor.Equals(name, StringComparison.Ordinal))
				return value;
		return null;
	}
}