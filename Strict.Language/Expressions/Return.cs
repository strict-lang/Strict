using Strict.Tokens;

namespace Strict.Language.Expressions
{
	public class Return : Expression
	{
		public Return(Expression expression) : base(expression.ReturnType) =>
			Expression = expression;

		public Expression Expression { get; }
		public override string ToString() => Token.Return + " " + Expression;
	}
}