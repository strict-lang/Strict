using System;
using System.Linq;

namespace Strict.Language.Expressions;

public sealed class Mutable : Value
{
	private Mutable(Context context, Expression expression) : base(GetMutableReturnType(context, expression),
		expression)
	{
		ReturnType.AddDataReturnTypeToMutableImplements(DataReturnType);
		ReturnType.MutableReturnType = DataReturnType;
	}

	private static Type GetMutableReturnType(Context context, Expression expression) =>
		expression.ReturnType.Name.StartsWith(Base.Mutable, StringComparison.Ordinal)
			? expression.ReturnType
			: context.GetType(Base.Mutable + "(" + expression.ReturnType.Name + ")");

	public static Expression? TryParse(Body body, ReadOnlySpan<char> line) =>
		line.Contains(" = ", StringComparison.Ordinal)
			? TryParseReassignment(body, line)
			: TryParseInitialization(body, line);

	public Type DataReturnType => ((Expression)Data).ReturnType;

	private static Expression TryParseReassignment(Body body, ReadOnlySpan<char> line)
	{
		var parts = line.Split();
		parts.MoveNext();
		var expression = body.Method.ParseExpression(body, parts.Current);
		return IsMutable(expression)
			? UpdateMemberOrVariableValue(body, expression, line[(parts.Current.Length + 3)..])
			: throw new ImmutableTypesCannotBeChanged(body, parts.Current.ToString());
	}

	private static Expression? TryParseInitialization(Body body, ReadOnlySpan<char> line) =>
		line.StartsWith("Mutable", StringComparison.Ordinal)
			? new Mutable(body.Method,
				body.Method.ParseExpression(body, line[(line.IndexOf('(') + 1)..line.LastIndexOf(')')]))
			: null;

	private static bool IsMutable(Expression expression) =>
		expression.ReturnType.Name == Base.Mutable ||
		expression.ReturnType.Implements.Any(t => t.Name == Base.Mutable) || expression is MemberCall memberCall && memberCall.Member.IsMutable;

	private static Expression UpdateMemberOrVariableValue(Body body,
		Expression expression, ReadOnlySpan<char> remainingLineSpan)
	{
		switch (expression)
		{
		case MemberCall memberCall:
			memberCall.Member.Value = body.Method.ParseExpression(body, remainingLineSpan);
			return new Mutable(body.Method, memberCall);
		case VariableCall variableCall:
			body.UpdateVariable(variableCall.Name, body.Method.ParseExpression(body, remainingLineSpan));
			return new Mutable(body.Method, variableCall);
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