using System;
using System.Collections.Generic;
using System.Linq;
using Strict.Tokens;

namespace Strict.Language.Expressions
{
	public class AllExpressionParser : ExpressionParser
	{
		public override void Parse(Method method, List<Token> tokens)
		{
			if (tokens.Count == 2 && tokens[0].Name.IsBinaryOperator() && tokens[1].IsNumber)
				AddBinary(method, tokens);
			else if (tokens.Count == 2 && tokens[0] == Token.Dot)
				AddMethodCall(method);
			else if (tokens.Count == 2 && tokens[0] == Token.Return)
				AddReturn(method, tokens);
			else if (tokens.Count == 1 && !tokens[0].Name.IsBinaryOperator() &&
				tokens[0] != Token.Dot && tokens[0] != Token.Return)
				AddSingleToken(method, tokens);
			else
				return;
			tokens.Clear();
		}

		private void AddBinary(Method method, List<Token> tokens) =>
			expressions[^1] = new Binary(expressions[^1],
				method.GetType(Base.Number).Methods.First(m => m.Name == tokens[0].Name),
				GetValue(method, tokens[1]));

		private static Value GetValue(Method method, Token token) =>
			new Value(token.Name == Token.Number
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
}