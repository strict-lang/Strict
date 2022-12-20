using System;

namespace Strict.Language.Expressions;

public sealed class MutableAssignment : ConcreteExpression
{
	private MutableAssignment(Body scope, string name, Expression newValue) :
		base(newValue.ReturnType, true)
	{
		scope.UpdateVariable(name, newValue);
	}

	public static Expression? TryParse(Body body, ReadOnlySpan<char> line) =>
		line.Contains(" = ", StringComparison.Ordinal)
			? TryParseReassignment(body, line)
			: null;

	private static Expression TryParseReassignment(Body body, ReadOnlySpan<char> line)
	{
		var parts = line.Split();
		parts.MoveNext();
		var expression = body.Method.ParseExpression(body, parts.Current);
		var newExpression = body.Method.ParseExpression(body, line[(parts.Current.Length + 3)..]);
		return expression.IsMutable
			? UpdateMemberOrVariableValue(body, expression, newExpression)
			: throw new Body.ValueIsNotMutableAndCannotBeChanged(body, parts.Current.ToString());
	}

	private static Expression UpdateMemberOrVariableValue(Body body,
		Expression expression, Expression newExpression)
	{
		if (!expression.ReturnType.IsCompatible(newExpression.ReturnType))
			throw new ValueTypeNotMatchingWithAssignmentType(body, expression.ReturnType.Name,
				newExpression.ReturnType.Name);
		switch (expression)
		{
		case MemberCall memberCall:
			memberCall.Member.Value = newExpression;
			return memberCall;
		case VariableCall variableCall:
			return new MutableAssignment(body, variableCall.Name, newExpression);
		default:
			throw new InvalidAssignmentTarget(body, expression.ToString());
		}
	}

	public sealed class ValueTypeNotMatchingWithAssignmentType : ParsingFailed
	{
		public ValueTypeNotMatchingWithAssignmentType(Body body, string currentValueType, string newValueType) : base(body, $"Cannot assign {newValueType} value type to {currentValueType} member or variable") { }
	}

	public sealed class InvalidAssignmentTarget : ParsingFailed
	{
		public InvalidAssignmentTarget(Body body, string message) : base(body, message) { }
	}

	public sealed class DirectUsageOfMutableTypesOrImplementsAreForbidden : ParsingFailed
	{
		public DirectUsageOfMutableTypesOrImplementsAreForbidden(Body body, string expressionText, string variableName) : base(body, $"Direct usage of mutable types or type that implements Mutable {expressionText} are not allowed. Instead use immutable types for variable {variableName}") { }
	}
}