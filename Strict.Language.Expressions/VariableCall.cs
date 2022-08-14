namespace Strict.Language.Expressions;

// ReSharper disable once HollowTypeName
public sealed class VariableCall : Expression//TODO: maybe rename to LetCall, but then again only variables will survive the optimizers (let with constant values will always be inlined)
{
	//TODO: need tests, also need to check if variable is available, crash if trying to use something that is not in scope, etc.
	public VariableCall(string name, Expression currentValue) : base(currentValue.ReturnType)
	{
		Name = name;
		CurrentValue = currentValue;
	}

	public string Name { get; }
	public Expression CurrentValue { get; }
	public override string ToString() => Name;

	public sealed class IdentifierNotFound : ParsingFailed
	{
		public IdentifierNotFound(Method.Line line, string name) : base(line, name) { }
	}
}