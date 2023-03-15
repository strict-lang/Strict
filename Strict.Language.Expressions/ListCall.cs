using static Strict.Language.Expressions.MethodExpressionParser;
using System.Collections.Generic;

namespace Strict.Language.Expressions;

public sealed class ListCall : ConcreteExpression
{
	public ListCall(Expression list, Expression index) : base(
		list.ReturnType is GenericTypeImplementation listReturnType
			? listReturnType.ImplementationTypes[0]
			: list.ReturnType, list.IsMutable)
	{
		List = list;
		Index = index;
	}

	public Expression List { get; }
	public Expression Index { get; }

	public static Expression? TryParse(Body body, Expression? variable,
		IReadOnlyList<Expression> arguments) =>
		variable is null or MethodCall
			? variable
			: arguments.Count > 0
				? variable.ReturnType.IsIterator
					? new ListCall(variable, arguments[0])
					: throw new InvalidArgumentItIsNotMethodOrListCall(body, variable, arguments)
				: variable;

	public override string ToString() => $"{List}({Index})";
}