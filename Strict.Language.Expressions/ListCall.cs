namespace Strict.Language.Expressions;

public sealed class ListCall : ConcreteExpression
{
	public ListCall(Expression list, Expression index) : base(
		((GenericType)list.ReturnType).ImplementationTypes[0])
	{
		List = list;
		Index = index;
	}

	public Expression List { get; }
	public Expression Index { get; }
	public override string ToString() => $"{List}({Index})";
}