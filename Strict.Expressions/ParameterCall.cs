using Strict.Language;

namespace Strict.Expressions;

public sealed class ParameterCall(Parameter parameter, int lineNumber = 0)
	: Expression(parameter.Type, lineNumber, parameter.IsMutable)
{
	public Parameter Parameter { get; } = parameter;
	public override bool IsConstant => Parameter.IsConstant;

	public static Expression? TryParse(Body body, ReadOnlySpan<char> input)
	{
		foreach (var parameter in body.Method.Parameters)
			if (input.Equals(parameter.Name, StringComparison.Ordinal))
				return new ParameterCall(parameter, body.CurrentFileLineNumber);
		return null;
	}

	public override string ToString() => Parameter.Name;
	//ncrunch: no coverage start
	public override bool Equals(Expression? other) =>
		ReferenceEquals(this, other) ||
		(other is ParameterCall pc && Parameter.Name == pc.Parameter.Name &&
			Parameter.Type == pc.Parameter.Type);
	public override int GetHashCode() => Parameter.GetHashCode();
}