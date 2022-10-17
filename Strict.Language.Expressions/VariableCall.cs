namespace Strict.Language.Expressions;

public sealed class VariableCall : ConcreteExpression
{
	public VariableCall(string name, Expression currentValue) : base(currentValue.ReturnType)
	{
		Name = name;
		CurrentValue = currentValue;
	}

	public string Name { get; }
	public Expression CurrentValue { get; }
	public override string ToString() => Name;
}