namespace Strict.Language
{
	/// <summary>
	/// Each line in a method is an expression, many expressions have child expressions (if, for,
	/// while) or consists of multiple expressions (e.g. binary operations have two expressions)
	/// </summary>
	public class MethodExpression
	{
		public MethodExpression(Type returnType) => ReturnType = returnType;
		public Type ReturnType { get; }
	}
}