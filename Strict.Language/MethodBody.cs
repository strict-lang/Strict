using System.Collections.Generic;

namespace Strict.Language
{
	/// <summary>
	/// Every method body is just an expression, which might contain multiple expressions, which are
	/// all executed and then the final result is returned (all previous expressions must succeed).
	/// Method parameters are in this context and can be used by any of the expressions nested here.
	/// </summary>
	public class MethodBody : Expression
	{
		public MethodBody(Method method, IReadOnlyList<Expression> expressions) :
			base(method.ReturnType) =>
			Expressions = expressions;

		public IReadOnlyList<Expression> Expressions { get; }
	}
	/*simply don't use anymore, just use ExpressionParser.Parse and assign this to evaluated expression lazily to method.body
	public class MethodBody //Tokenizer remove this and just use Pidgin here, not directly, just create a base interface/abstract class in preparation!
	{
		public MethodBody(Method method, ExpressionParser parser, string[] lines)
		{
			//var lexer = new LineLexer(this);//also remove, just use ExpressionParser to get a tree for each method
			this.method = method;
			this.parser = parser;
			/*obs parser.Restart();
			for (var line = 1; line < lines.Length; line++)
				lexer.Process(lines[line]);
			if (unprocessedTokens.Count > 0)
				throw new UnprocessedTokensAtEndOfFile(method, unprocessedTokens);
			Expressions = parser.Expressions;
			*

		}

		private readonly Method method;
		private readonly ExpressionParser parser;
		private readonly List<DefinitionToken> unprocessedTokens = new List<DefinitionToken>();

		public class UnprocessedTokensAtEndOfFile : Exception
		{
			public UnprocessedTokensAtEndOfFile(Method method, IReadOnlyList<DefinitionToken> tokens) : base(
				method + "\nUnprocessed Tokens: " + tokens.ToWordListString()) { }
		}

		public Expression[] Expressions { get; }

		public void Add(DefinitionToken token)
		{
			unprocessedTokens.Add(token);
			parser.Parse(method, unprocessedTokens);//just parse string lines directly!
		}
	}
	*/
}