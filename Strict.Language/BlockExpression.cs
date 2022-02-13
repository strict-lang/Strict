namespace Strict.Language;

public abstract class BlockExpression : Expression
{
	protected BlockExpression(Type returnType) : base(returnType) { }
}