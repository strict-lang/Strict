using System.Collections.Generic;

namespace Strict.Language
{
	/// <summary>
	/// Abstracts the actual expressions and parsing away to the Expressions project.
	/// <see cref="Method.Body"/> will call this lazily when it is called the first time, which is
	/// usually not happening until use. This improves performance a lot as we almost do no parsing.
	/// </summary>
	public abstract class ExpressionParser
	{
		/*should not longer be needed
		public void Restart() => expressions.Clear();
		public Expression[] Expressions => expressions.ToArray();
		protected readonly List<Expression> expressions = new List<Expression>();
		*/
		public abstract Expression Parse(Method method, string lines);
	}
}