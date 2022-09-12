namespace Strict.Language.Expressions;

// ReSharper disable once HollowTypeName
public sealed class VariableCall : Expression
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