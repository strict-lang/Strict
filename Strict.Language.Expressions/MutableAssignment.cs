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
		if (!expression.IsMutable)
			throw new Body.ValueIsNotMutableAndCannotBeChanged(body, expression.ToString());
		return UpdateMemberOrVariableValue(body, expression, newExpression);
	}

	private static Expression UpdateMemberOrVariableValue(Body body,
		Expression expression, Expression newExpression)
	{
		if (expression is VariableCall variableCall)
			return new MutableAssignment(body, variableCall.Name, newExpression);
		if (expression is MemberCall memberCall)
		{
			memberCall.Member.UpdateValue(newExpression, body);
			return memberCall;
		}
		if (expression is ParameterCall parameterCall)
		{
			parameterCall.Parameter.UpdateValue(newExpression, body);
			return parameterCall;
		}
		if (expression is ListCall listCall)
		{
			if (listCall.List is VariableCall { CurrentValue: List listExpression })
				listExpression.UpdateValue(body, listCall.Index, newExpression);
			return listCall;
		}
		throw new InvalidAssignmentTarget(body, expression.ToString()); //ncrunch: no coverage
	}

	public override string ToString() => Name + " = " + Value;

	public sealed class ValueTypeNotMatchingWithAssignmentType : ParsingFailed
	{
		public ValueTypeNotMatchingWithAssignmentType(Body body, string currentValueType, string newValueType) : base(body, $"Cannot assign {newValueType} value type to {currentValueType} member or variable") { }
	}

	public sealed class InvalidAssignmentTarget : ParsingFailed
	{
		public InvalidAssignmentTarget(Body body, string message) : base(body, message) { } //ncrunch: no coverage
	}
}