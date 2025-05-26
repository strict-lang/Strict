namespace Strict.Language.Expressions;

public sealed class ParameterCall(Parameter parameter)
	: Expression(parameter.Type, parameter.IsMutable)
{
	public Parameter Parameter { get; } = parameter;

	public static Expression? TryParse(Body body, ReadOnlySpan<char> input)
	{
		foreach (var parameter in body.Method.Parameters)
			if (input.Equals(parameter.Name, StringComparison.Ordinal))
				return new ParameterCall(parameter);
		return null;
	}

	public override string ToString() => Parameter.Name;
}