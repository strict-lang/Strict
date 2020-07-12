using System.Collections.Generic;
using System.Linq;
using Strict.Tokens;

namespace Strict.Language.Expressions
{
	public class AllExpressionParser : ExpressionParser
	{
		public override void Parse(Method method, List<Token> tokens)
		{
			if (tokens.Count == 3 && tokens[1].Name.IsBinaryOperator() && tokens[0].IsNumber &&
				tokens[2].IsNumber)
			{
				expressions.Add(new Binary(GetValue(method, tokens[0]),
					method.GetType(Base.Number).Methods.First(m => m.Name == tokens[1].Name),
					GetValue(method, tokens[0])));
				tokens.Clear();
			}
		}

		private static Value GetValue(Method method, Token token) =>
			new Value(token.Name == Token.Number
				? method.GetType(Base.Number)
				: method.GetType(Base.Boolean), token.Value!);

		/*dummy			
			if (lines[0].StartsWith("\treturn"))
			{
				var number = method.GetType(Base.Number);
				expressions.Add(new Return(new Binary(new Number(method, 5), number.Methods[0],
					new Boolean(method, true))));
			}
			else
			{
				var log = method.GetType(Base.Log);
				expressions.Add(new MethodCall(new MemberCall(method.Type.Members[0]),
					log.Methods.First(m => m.Name == "WriteLine"), new MediaTypeNames.Text(method, "Hey")));
			}
			*/
	}
}