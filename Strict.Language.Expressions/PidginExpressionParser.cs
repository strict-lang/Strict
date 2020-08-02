using Pidgin;
using Pidgin.Expression;
using static Pidgin.Parser;
using static Pidgin.Parser<char>;

namespace Strict.Language.Expressions
{
	public class PidginExpressionParser : ExpressionParser
	{
		public override Expression Parse(Method method, string lines) => Expression(method).ParseOrThrow(lines);
		
		private static Parser<char, Expression> Expression(Method method) =>
			Pidgin.Expression.ExpressionParser.Build<char, Expression>(expr => (
				OneOf(Number(method), Boolean(method), Text(method)
					//TODO: Identifier,
					//TODO: Literal,
					//TODO? Parenthesised(expr).Labelled("parenthesised expression")
				), new OperatorTableRow<char, Expression>[]
				{
					//TODO? Pidgin.Expression.Operator.PostfixChainable(Call(expr)),
					//TODO? Pidgin.Expression.Operator.Prefix(Neg).And(Operator.Prefix(Complement)),
					//TODO? Pidgin.Expression.Operator.InfixL(Mul),
					//lator: Pidgin.Expression.Operator.InfixL(Add(method))
				})).Labelled(nameof(Expression));

		private static Parser<char, Expression> Number(Context method) =>
			ExpressionToken(Num, nameof(Number)).
				Select<Expression>(value => new Number(method, value));

		private static Parser<char, T> ExpressionToken<T>(Parser<char, T> token, string label) =>
			Try(token).Before(End).Labelled(label);

		private static Parser<char, Expression> Boolean(Context method) =>
			ExpressionToken(String("true").Or(String("false")), nameof(Boolean)).
				Select<Expression>(value => new Boolean(method, value == "true"));

		private static Parser<char, Expression> Text(Context method) =>
			ExpressionToken(Token(c => c != '"').ManyString().Between(Quote), nameof(Text)).
				Select<Expression>(text => new Text(method, text));
			
		private static readonly Parser<char, char> Quote = Char('"');
	}
}