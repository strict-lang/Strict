namespace Strict.Language.Expressions;

public sealed class ParameterCall : Expression
{
	public ParameterCall(Parameter parameter) : base(parameter.Type) => Parameter = parameter;
	public Parameter Parameter { get; internal set; }
	public override string ToString() => Parameter.Name;
}