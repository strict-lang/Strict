using System.Collections.Generic;
using Strict.Language.Expressions;
using Strict.Tokens;

namespace Strict.Language
{
	public abstract class ExpressionParser
	{
		public void Restart() => expressions.Clear();
		public Expression[] Expressions => expressions.ToArray();
		private readonly List<Expression> expressions = new List<Expression>();
		public bool Parse(List<Token> tokens) => false;
	}
}