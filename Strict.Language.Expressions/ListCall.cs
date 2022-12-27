namespace Strict.Language.Expressions;

public sealed class ListCall : ConcreteExpression
{
	public ListCall(Expression list, Expression index) : base(
		list.ReturnType is GenericType listReturnType
			? listReturnType.ImplementationTypes[0]
			: list.ReturnType)
	{
		List = list;
		Index = index;
	}

	public Expression List { get; }
	public Expression Index { get; }
	public override string ToString() => $"{List}({Index})";
}