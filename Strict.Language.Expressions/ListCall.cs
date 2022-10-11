namespace Strict.Language.Expressions;

public sealed class ListCall : NonGenericExpression
{
	public ListCall(Expression list, Expression index) : base(
		((GenericType)list.ReturnType).Implementation)
	{
		List = list;
		Index = index;
	}

	public Expression List { get; }
	public Expression Index { get; }
	public override string ToString() => $"{List}({Index})";
}