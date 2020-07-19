using System;
using System.Collections.Generic;
using Strict.Tokens;

namespace Strict.Language
{
	public class MethodBody : Tokenizer
	{
		public MethodBody(Method method, ExpressionParser parser, string[] lines)
		{
			var lexer = new LineLexer(this);//TODO: also remove, just use ExpressionParser to get a tree for each method
			this.method = method;
			this.parser = parser;
			parser.Restart();
			for (var line = 1; line < lines.Length; line++)
				lexer.Process(lines[line]);
			if (unprocessedTokens.Count > 0)
				throw new UnprocessedTokensAtEndOfFile(method, unprocessedTokens);
			Expressions = parser.Expressions;
		}

		private readonly Method method;
		private readonly ExpressionParser parser;
		private readonly List<Token> unprocessedTokens = new List<Token>();

		public class UnprocessedTokensAtEndOfFile : Exception
		{
			public UnprocessedTokensAtEndOfFile(Method method, IReadOnlyList<Token> tokens) : base(
				method + "\nUnprocessed Tokens: " + tokens.ToWordListString()) { }
		}

		public Expression[] Expressions { get; }

		public void Add(Token token)
		{
			unprocessedTokens.Add(token);
			parser.ParseOldTODO(method, unprocessedTokens);
		}
	}
}