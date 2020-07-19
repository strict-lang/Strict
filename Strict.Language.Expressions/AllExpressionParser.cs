using System;
using System.Collections.Generic;
using System.Linq;
using Strict.Tokens;
using Pidgin;
using Pidgin.Expression;
using static Pidgin.Parser;
using static Pidgin.Parser<char>;

namespace Strict.Language.Expressions
{
	public class AllExpressionParser : ExpressionParser//TODO: maybe merge .Parsing into here
	{
		public override void ParseOldTODO(Method method, List<Token> tokens)
		{
			//TODO: this is very annoying to edit, maybe we should just create a new project and slowly get all features working there
			if (tokens.Count == 2 && tokens[0].Name.IsBinaryOperator() && tokens[1].IsNumber)
				AddBinary(method, tokens);
			else if (tokens.Count == 2 && tokens[0] == Strict.Tokens.Token.Dot)
				AddMethodCall(method);
			else if (tokens.Count == 2 && tokens[0] == Strict.Tokens.Token.Return)
				AddReturn(method, tokens);
			else if (tokens.Count == 1 && !tokens[0].Name.IsBinaryOperator() &&
				tokens[0] != Strict.Tokens.Token.Dot && tokens[0] != Strict.Tokens.Token.Return)
				AddSingleToken(method, tokens);
			else
				return;
			tokens.Clear();
		}

		public override Expression Parse(Method method, string lines)
		{
			return ParseMethodCall(method).ParseOrThrow(lines);
			//ts": return new MethodBodyExpression(method);
		}
		
		private static Parser<char, Expression> ParseMethodCall(Method method)
			=> Pidgin.Expression.ExpressionParser.Build<char, Expression>(
			expr => (
				OneOf(
					ParseNumber(method)
					//TODO: Identifier,
					//TODO: Literal,
					//TODO? Parenthesised(expr).Labelled("parenthesised expression")
				),
				new OperatorTableRow<char, Expression>[]
				{
					//TODO? Pidgin.Expression.Operator.PostfixChainable(Call(expr)),
					//TODO? Pidgin.Expression.Operator.Prefix(Neg).And(Operator.Prefix(Complement)),
					//TODO? Pidgin.Expression.Operator.InfixL(Mul),
					//lator: Pidgin.Expression.Operator.InfixL(Add(method))
				}
			)
		).Labelled("expression");
		
		private static Parser<char, Expression> ParseNumber(Method method)
			=> Try(Num)
				.Select<Expression>(value => new Number(method, value))
				.Labelled("number");
		/*TODO
		private static Parser<char, Func<Expression, Expression, Expression>> ParseBinary(Parser<char, BinaryOperatorType> op)
			=> op.Select<Func<Expression, Expression, Expression>>(type => (l, r) => new Binary(type, l, r));

		private static Parser<char, Func<Expression, Expression, Expression>> ParseAdd(Method method)
			=> ParseBinary(Try(Char('+').ThenReturn(BinaryOperatorType...Add));

		*/
		private void AddBinary(Method method, List<Token> tokens) =>
			expressions[^1] = new Binary(expressions[^1],
				method.GetType(Base.Number).Methods.First(m => m.Name == tokens[0].Name),
				GetValue(method, tokens[1]));
		private static Value GetValue(Method method, Token token) =>
			new Value(token.Name == Strict.Tokens.Token.Number
				? method.GetType(Base.Number)
				: method.GetType(Base.Boolean), token.Value!);

		private void AddMethodCall(Method method) =>
			expressions[^1] = new MethodCall(expressions[^1], method.Type.Methods[0]);

		private void AddReturn(Method method, List<Token> tokens) =>
			expressions.Add(new Return(new Boolean(method, (bool)tokens[1].Value!)));

		private void AddSingleToken(Method method, List<Token> tokens)
		{
			if (tokens[0].IsNumber)
				expressions.Add(new Number(method, (double)tokens[0].Value!));
			else if (tokens[0].IsBoolean)
				expressions.Add(new Boolean(method, (bool)tokens[0].Value!));
			else if (tokens[0].IsText)
				expressions.Add(new Text(method, (string)tokens[0].Value!));
			else if (tokens[0].IsIdentifier)
				expressions.Add(new MemberCall(method.Type.Members[0]));
			else
				throw new UnsupportedToken(tokens[0]);
		}

		public class UnsupportedToken : Exception
		{
			public UnsupportedToken(Token token) : base(token.ToString()) { }
		}
	}

	public class MethodBodyExpression : Expression//TODO: merge with MethodBody, more refactoring needed
	{
		public MethodBodyExpression(Method method) : base(method.ReturnType) { }
	}
}