using System;
using System.Linq;

namespace Strict.Language.Expressions;

public class Mutable
{
	public static Expression? TryParse(Body body, ReadOnlySpan<char> line) =>
		line.Contains(new[] { " = " })
			? TryParseReassignment(body, line)
			: null;

	private static Expression TryParseReassignment(Body body, ReadOnlySpan<char> line)
	{
		var parts = line.Split();
		parts.MoveNext();
		var expression = body.Method.ParseExpression(body, parts.Current);
		return IsMutable(expression)
			? UpdateMemberOrVariableValue(body, expression, line[(parts.Current.Length + 1 + 1 + 1)..])
			: throw new ImmutableTypesCannotBeChanged(body, parts.Current.ToString());
	}

	private static bool IsMutable(Expression expression) =>
		expression.ReturnType.Name == Base.Mutable ||
		expression.ReturnType.Implements.Any(t => t.Name == Base.Mutable);

	private static Expression UpdateMemberOrVariableValue(Body body,
		Expression expression, ReadOnlySpan<char> remainingLineSpan)
	{
		switch (expression)
		{
		case MemberCall memberCall:
			memberCall.Member.Value = body.Method.ParseExpression(body, remainingLineSpan);
			return memberCall;
		case VariableCall variableCall:
			body.AddOrUpdateVariable(variableCall.Name, body.Method.ParseExpression(body, remainingLineSpan));
			return variableCall;
		default:
			throw new InvalidAssignmentTarget(body, expression.ToString());
		}
	}

	public sealed class InvalidAssignmentTarget : ParsingFailed
	{
		public InvalidAssignmentTarget(Body body, string message) : base(body, message) { }
	}

	public sealed class ImmutableTypesCannotBeChanged : ParsingFailed
	{
		public ImmutableTypesCannotBeChanged(Body body, string message) : base(body, message) { }
	}
}