/*probably makes no sense, we just return the value of the expression/method (last line?) any line should be an assert anyway
namespace Strict.Language.Expressions
{
public class Return : Expression
{
	public Return(Expression expression) : base(expression.ReturnType) =>
		Expression = expression;

	public Expression Expression { get; }
	public override string ToString() => DefinitionToken.Return + " " + Expression;
}
}
*/