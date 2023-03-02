using System;

namespace Strict.Language.Expressions;

public sealed class VariableCall : ConcreteExpression
{
	public VariableCall(string name, Expression currentValue) : base(currentValue.ReturnType, currentValue.IsMutable)
	{
		Name = name;
		CurrentValue = currentValue;
	}

	public static Expression? TryParse(Body body, ReadOnlySpan<char> input)
	{
		var variableValue = body.FindVariableValue(input);
		return variableValue != null
			? new VariableCall(input.ToString(), variableValue)
			: null;
	}

	public string Name { get; }
	public Expression CurrentValue { get; }
	public override string ToString() => Name;
}