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
				return new ParameterCall(parameter, body.Method.TypeLineNumber + body.ParsingLineNumber);
		return null;
	}

	public override string ToString() => Parameter.Name;
}