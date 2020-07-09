namespace Strict.Language.Expressions
{
	/// <summary>
	/// Each line in a method is an expression, many expressions have child expressions (if, for,
	/// while) or consists of multiple expressions (e.g. binary operations have two expressions).
	/// There are no statements in Strict, every line in a method is an expression, every other
	/// line in a .strict file is either implement, has or a method definition.
	/// </summary>
	public abstract class Expression
	{
		protected Expression(Type returnType) => ReturnType = returnType;
		public Type ReturnType { get; }
	}
}