using static Strict.Language.Expressions.MethodExpressionParser;

namespace Strict.Language.Expressions;

public sealed class ListCall(Expression list, Expression index) : ConcreteExpression(
	list.ReturnType is GenericTypeImplementation listReturnType
		? listReturnType.ImplementationTypes[0]
		: list.ReturnType, list.IsMutable)
{
	public Expression List { get; } = list;
	//TODO: also do a IndexOutOfRangeInListExpressions check if List has known fixed length!
	public Expression Index { get; } = index;

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