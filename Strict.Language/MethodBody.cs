using System.Collections.Generic;
using Strict.Language.Tokens;

namespace Strict.Language
{
	public class MethodBody : Tokenizer
	{
		public MethodBody(Method method, IReadOnlyList<string> lines)
		{
			var lineLexer = new LineLexer(this);
			for (var index = 0; index < lines.Count; index++)
				lineLexer.Process(lines[index]);
			expressions.Add(new MethodExpression(method.GetType(Base.None)));
		}

		private readonly List<MethodExpression> expressions = new List<MethodExpression>();
		public IReadOnlyList<MethodExpression> Expressions => expressions;
		public void Add(Token token) { }
	}
}