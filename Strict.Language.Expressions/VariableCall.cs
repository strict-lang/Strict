namespace Strict.Language.Expressions;

public sealed class VariableCall(string name, Expression currentValue)
	: ConcreteExpression(currentValue.ReturnType, currentValue.IsMutable)
{
	public static Expression? TryParse(Body body, ReadOnlySpan<char> input)
	{
		var variableValue = body.FindVariableValue(input);
		return variableValue != null
			? new VariableCall(input.ToString(), variableValue)
			: null;
	}

	public string Name { get; } = name;
	public Expression CurrentValue { get; } = currentValue;
	public override string ToString() => Name;
}