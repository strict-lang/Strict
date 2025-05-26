namespace Strict.Language.Expressions;

public sealed class MutableAssignment : ConcreteExpression
{
	private MutableAssignment(Body scope, string name, Expression value) :
		base(value.ReturnType, true)
	{
		Name = name;
		Value = value;
		value.IsMutable = true;
		scope.UpdateVariable(name, value);
	}

	public string Name { get; }
	public Expression Value { get; }

	public static Expression? TryParse(Body body, ReadOnlySpan<char> line) =>
		line.Contains(" = ", StringComparison.Ordinal)
			? TryParseReassignment(body, line)
			: null;

	private static Expression TryParseReassignment(Body body, ReadOnlySpan<char> line)
	{
		var parts = line.Split('=', StringSplitOptions.TrimEntries);
		parts.MoveNext();
		var expression = body.Method.ParseExpression(body, parts.Current);
		var newExpression = body.Method.ParseExpression(body, line[(parts.Current.Length + 3)..]);
		if (!newExpression.ReturnType.IsCompatible(expression.ReturnType))
			throw new ValueTypeNotMatchingWithAssignmentType(body, expression.ReturnType.Name,
				newExpression.ReturnType.Name);
		return expression is VariableCall variableCall
			? new MutableAssignment(body, variableCall.Name, newExpression)
			: UpdateMemberOrVariableValue(body, expression, newExpression);
	}

	private static Expression UpdateMemberOrVariableValue(Body body,
		Expression expression, Expression newExpression)
	{
		switch (expression)
		{
		case MemberCall memberCall:
			memberCall.Member.UpdateValue(newExpression, body);
			return memberCall;
		case ParameterCall parameterCall:
			parameterCall.Parameter.UpdateValue(newExpression, body);
			return parameterCall;
		case ListCall listCall:
		{
			if (listCall.List is VariableCall { CurrentValue: List listExpression })
				listExpression.UpdateValue(body, listCall.Index, newExpression);
			return listCall;
		}
		default:
			throw new InvalidAssignmentTarget(body, expression.ToString()); //ncrunch: no coverage
		}
	}

	public override string ToString() => Name + " = " + Value;

	public sealed class ValueTypeNotMatchingWithAssignmentType(Body body,
		string currentValueType,
		string newValueType) : ParsingFailed(body,
		$"Cannot assign {newValueType} value type to {currentValueType} member or variable");

	public sealed class InvalidAssignmentTarget(Body body, string message)
		: ParsingFailed(body, message)
	{
		//ncrunch: no coverage
	}
}