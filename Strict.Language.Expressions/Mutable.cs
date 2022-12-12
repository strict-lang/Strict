using System;
using System.Linq;

namespace Strict.Language.Expressions;

public sealed class Mutable : Value
{
	public Mutable(Context context, Expression expression) : base(GetMutableReturnType(context, expression),
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
			: null;

	public Type DataReturnType => ((Expression)Data).ReturnType;

	private static Expression TryParseReassignment(Body body, ReadOnlySpan<char> line)
	{
		var parts = line.Split();
		parts.MoveNext();
		var expression = body.Method.ParseExpression(body, parts.Current);
		return IsMutable(expression)
			? UpdateMemberOrVariableValue(body, expression, body.Method.ParseExpression(body, line[(parts.Current.Length + 3)..]))
			: throw new ImmutableTypesCannotBeChanged(body, parts.Current.ToString());
	}

	private static bool IsMutable(Expression expression) =>
		expression.ReturnType.Name == Base.Mutable ||
		expression.ReturnType.Implements.Any(t => t.Name == Base.Mutable) || expression is MemberCall memberCall && memberCall.Member.IsMutable;

	private static Expression UpdateMemberOrVariableValue(Body body,
		Expression expression, Expression newExpression)
	{
		if (!expression.ReturnType.IsCompatible(newExpression.ReturnType))
			throw new InvalidDataAssignment(body, expression.ReturnType.Name,
				newExpression.ReturnType.Name);
		switch (expression)
		{
		case MemberCall memberCall:
			memberCall.Member.Value = newExpression;
			return memberCall;
		case VariableCall variableCall:
			return new Assignment(body, variableCall.Name, newExpression);
		default:
			throw new InvalidAssignmentTarget(body, expression.ToString());
		}
	}

	public sealed class InvalidDataAssignment : ParsingFailed
	{
		public InvalidDataAssignment(Body body, string currentValueType, string newValueType) : base(body, $"Cannot assign {newValueType} value type to {currentValueType} member or variable") { }
	}

	public sealed class InvalidAssignmentTarget : ParsingFailed
	{
		public InvalidAssignmentTarget(Body body, string message) : base(body, message) { }
	}

	public sealed class ImmutableTypesCannotBeChanged : ParsingFailed
	{
		public ImmutableTypesCannotBeChanged(Body body, string message) : base(body, message) { }
	}

	public override string ToString() =>
		Data is List { Values.Count: 0 }
			? DataReturnType.Name
			: base.ToString();
}