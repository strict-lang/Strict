namespace Strict.Language.Expressions;

public sealed class MutableAssignment : ConcreteExpression
{
	private MutableAssignment(Body scope, string variableOrParameterName, Expression newValue, Expression oldValue) :
		base(newValue.ReturnType, true)
	{
		if (oldValue is { IsMutable: false })
			throw new Body.ValueIsNotMutableAndCannotBeChanged(scope, oldValue.ToString());
		Name = variableOrParameterName;
		newValue.IsMutable = true;
		Value = newValue;
		OldValue = oldValue;
		scope.UpdateVariableOrParameter(Name, newValue);
	}

	public string Name { get; }
	public Expression Value { get; }
	public Expression OldValue { get; }

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
		if (!expression.ReturnType.IsSameOrCanBeUsedAs(newExpression.ReturnType))
			throw new ValueTypeNotMatchingWithAssignmentType(body, expression.ReturnType.Name,
				newExpression.ReturnType.Name);
		return new MutableAssignment(body, expression switch
		{
			VariableCall variableCall => variableCall.Name,
			ParameterCall parameterCall => parameterCall.Parameter.Name,
			_ => throw new InvalidAssignmentTarget(body, expression.ToStringWithType())
		}, newExpression, expression);
	}

	//TODO: this is a bit strange, why would we update the value here already, we are still parsing,
	//this should be done every time running the method, not now
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
		string currentValueType, string newValueType) : ParsingFailed(body,
		$"Cannot assign {newValueType} value type to {currentValueType} member or variable");

	public sealed class InvalidAssignmentTarget(Body body, string message)
		: ParsingFailed(body, message);
}