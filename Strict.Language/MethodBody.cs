using System;
using System.Collections.Generic;
using Strict.Language.Expressions;
using Strict.Tokens;

namespace Strict.Language
{
	public class MethodBody : Tokenizer {
		public MethodBody(Method method,IReadOnlyList<string> lines, ExpressionParser parser)
		{
			var lexer = new LineLexer(this);
			this.parser = parser;
			parser.Restart();
			foreach (var line in lines)
				lexer.Process(line);
			if (unprocessedTokens.Count > 0)
				throw new UnprocessedTokensAtEndOfFile(unprocessedTokens.ToWordListString());
			Expressions = parser.Expressions;
			/*
			var lineLexer = new LineLexer(this);
			for (var index = 0; index < lines.Count; index++)
				lineLexer.Process(lines[index]);
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
		{
			get;
		}
			*/
		}

		private readonly ExpressionParser parser;
		private readonly List<Token> unprocessedTokens = new List<Token>();

		public class UnprocessedTokensAtEndOfFile : Exception
		{
			public UnprocessedTokensAtEndOfFile(string message) : base(message) { }
		}

		public void Add(Token token)
		{
			unprocessedTokens.Add(token);
			if (parser.Parse(unprocessedTokens))
				unprocessedTokens.Clear();
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

		public IReadOnlyList<Expression> Expressions { get; }
	}
}