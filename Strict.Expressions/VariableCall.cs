using Strict.Language;

namespace Strict.Expressions;

public sealed class VariableCall(Variable variable, int lineNumber = 0)
	: ConcreteExpression(variable.Type, lineNumber, variable.IsMutable)
{
	public static Expression? TryParse(Body body, ReadOnlySpan<char> input)
	{
		var variable = body.FindVariable(input);
		return variable != null
			? new VariableCall(variable, body.CurrentFileLineNumber)
			: null;
	}

	public Variable Variable { get; } = variable;
	public override string ToString() => Variable.Name;
	public override bool IsConstant => Variable.InitialValue.IsConstant && !Variable.IsMutable;
}