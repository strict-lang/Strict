using System;
using System.Collections.Immutable;
using System.Linq;
using Pidgin;
using Pidgin.Expression;
using static Pidgin.Parser;
using static Pidgin.Parser<char>;

namespace Strict.Language.Expressions
{
	public class PidginExpressionParser : ExpressionParser
	{
		public override Expression Parse(Method context, string lines) => Expression(context).ParseOrThrow(lines);
		
		private static Parser<char, Expression> Expression(Method context) =>
			Pidgin.Expression.ExpressionParser.Build<char, Expression>(expr => (
				OneOf(Number(context), Boolean(context), Text(context)
					//TODO: Identifier,
					//TODO: Literal,
					//TODO? Parenthesised(expr).Labelled("parenthesised expression")
				), new OperatorTableRow<char, Expression>[]
				{
					//Operator.PostfixChainable(Call(expr)),
					//TODO? Pidgin.Expression.Operator.Prefix(Neg).And(Operator.Prefix(Complement)),
					//TODO? Pidgin.Expression.Operator.InfixL(Mul),
					Operator.InfixL(Add)
				})).Labelled(nameof(Expression));

		private static Parser<char, Expression> Number(Context context) =>
			ExpressionToken(Num, nameof(Number)).
				Select<Expression>(value => new Number(context, value));

		private static Parser<char, T> ExpressionToken<T>(Parser<char, T> token, string label) =>
			Try(token).Before(End).Labelled(label);

		private static Parser<char, Expression> Boolean(Context method) =>
			ExpressionToken(String("true").Or(String("false")), nameof(Boolean)).
				Select<Expression>(value => new Boolean(method, value == "true"));

		private static Parser<char, Expression> Text(Context method) =>
			ExpressionToken(Token(c => c != '"').ManyString().Between(Quote), nameof(Text)).
				Select<Expression>(text => new Text(method, text));
			
		private static readonly Parser<char, char> Quote = Char('"');
		/*TODO
		private static Parser<char, Func<Expression, Expression>> Call(Parser<char, Expression> subExpr)
			=> Parenthesized(subExpr.Separated(Char(',')))
				.Select<Func<Expression, Expression>>(args => method => new MethodCall(method, method, args.ToImmutableArray()))
				.Labelled(nameof(MethodCall));
		*/
		private static Parser<char, T> Parenthesized<T>(Parser<char, T> parser) =>
			parser.Between(Char('('), Char(')'));

		private static Parser<char, Func<Expression, Expression, Expression>>
			Binary(Parser<char, char> operation, string methodName) =>
			operation.Select<Func<Expression, Expression, Expression>>(type =>
				(l, r) => new Binary(l, l.ReturnType.Methods.First(m => m.Name == methodName), r));

		private static readonly Parser<char, Func<Expression, Expression, Expression>> Add =
			Binary(Char(BinaryOperator.Plus), BinaryOperator.Plus.ToString());
		/*TODO: something more clever than
        private static readonly Parser<char, Func<IExpr, IExpr, IExpr>> Mul
            = Binary(Tok("*").ThenReturn(BinaryOperatorType.Mul));
        private static readonly Parser<char, Func<IExpr, IExpr>> Neg
            = Unary(Tok("-").ThenReturn(UnaryOperatorType.Neg));
        private static readonly Parser<char, Func<IExpr, IExpr>> Complement
            = Unary(Tok("~").ThenReturn(UnaryOperatorType.Complement));
		*/
	}
}