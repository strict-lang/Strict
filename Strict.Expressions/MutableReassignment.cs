using Strict.Language;

namespace Strict.Expressions;

public sealed class MutableReassignment : ConcreteExpression
{
	private MutableReassignment(Body scope, string variableOrParameterName, Expression newValue,
		Expression oldValue) : base(newValue.ReturnType, true)
	{
		if (oldValue is { IsMutable: false })
			throw new Body.ValueIsNotMutableAndCannotBeChanged(scope, oldValue.ToString());
		Name = variableOrParameterName;
		Value = newValue;
		OldValue = oldValue;
		scope.CheckIfWeCouldUpdateMutableParameterOrVariable(Name, Value);
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
		var expression = body.Method.ParseExpression(body, parts.Current, true);
		var newExpression = body.Method.ParseExpression(body, line[(parts.Current.Length + 3)..], false);
		if (!expression.ReturnType.IsSameOrCanBeUsedAs(newExpression.ReturnType))
			throw new ValueTypeNotMatchingWithAssignmentType(body, expression.ReturnType.Name,
				newExpression.ReturnType.Name);
		return new MutableReassignment(body, expression switch
		{
			VariableCall variableCall => variableCall.Variable.Name,
			ParameterCall parameterCall => parameterCall.Parameter.Name,
			MemberCall memberCall => memberCall.Member.Name,
			ListCall listCall => listCall.ToString(),
			_ => throw new InvalidAssignmentTarget(body, expression)
		}, newExpression, expression);
	}
	/*
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
	*/

	public override string ToString() => Name + " = " + Value;

	public sealed class ValueTypeNotMatchingWithAssignmentType(Body body,
		string currentValueType, string newValueType) : ParsingFailed(body,
		$"Cannot assign {newValueType} value type to {currentValueType} member or variable");

	public sealed class InvalidAssignmentTarget(Body body, Expression expression)
		: ParsingFailed(body, expression.ToStringWithType() + " " + expression.GetType().Name);
}