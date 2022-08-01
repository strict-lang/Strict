namespace Strict.Language.Expressions;

// ReSharper disable once HollowTypeName
public sealed class VariableCall : Expression
{
	public VariableCall(string name, Expression value) : base(value.ReturnType)
	{
		Name = name;
		Value = value;
	}

	public string Name { get; }
	public Expression Value { get; }
	public override string ToString() => Name;
}