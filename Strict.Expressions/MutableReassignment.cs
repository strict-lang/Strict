using Strict.Language;

namespace Strict.Expressions;

public sealed class MutableReassignment : ConcreteExpression
{
	private MutableReassignment(Body scope, Expression target, Expression newValue) : base(newValue.ReturnType, true)
	{
		if (target is { IsMutable: false })
			throw new Body.ValueIsNotMutableAndCannotBeChanged(scope, target.ToString());
		Name = target switch
		{
			VariableCall variableCall => variableCall.Variable.Name,
			ParameterCall parameterCall => parameterCall.Parameter.Name,
			MemberCall memberCall => memberCall.Member.Name,
			_ => target.ToString()
		};
		Target = target;
		Value = newValue;
		var contextType = (target as MemberCall)?.Instance?.ReturnType ?? scope.Method.Type;
		if (target is not ListCall)
			scope.CheckIfWeCouldUpdateMutableParameterOrVariable(contextType, Name, Value);
	}

	public string Name { get; }
	public Expression Target { get; }
	public Expression Value { get; }

	public static Expression? TryParse(Body body, ReadOnlySpan<char> line) =>
		line.Contains(" = ", StringComparison.Ordinal)
			? TryParseReassignment(body, line)
			: null;

	private static Expression TryParseReassignment(Body body, ReadOnlySpan<char> line)
	{
		var parts = line.Split('=', StringSplitOptions.TrimEntries);
		parts.MoveNext();
		var expression = body.Method.ParseExpression(body, parts.Current, true);
		var newExpression = body.Method.ParseExpression(body, line[(parts.Current.Length + 3)..]);
		if (!newExpression.ReturnType.IsSameOrCanBeUsedAs(expression.ReturnType, false))
			throw new ValueTypeNotMatchingWithAssignmentType(body, expression.ReturnType.Name,
				newExpression.ReturnType.Name);
		return new MutableReassignment(body, expression, newExpression);
	}

	public override string ToString() => Name + " = " + Value;

	public sealed class ValueTypeNotMatchingWithAssignmentType(Body body,
		string currentValueType, string newValueType) : ParsingFailed(body,
		$"Cannot assign {newValueType} value type to {currentValueType} member or variable");
}