using System.Linq;

namespace Strict.Language.Expressions;

public class Binary : MethodCall
{
	public Binary(Expression left, Method operatorMethod, Expression right) : base(left,
		operatorMethod, right) { }

	public Expression Left => Instance;
	public Expression Right => Arguments[0];
	public override string ToString() => Left + " " + Method.Name + " " + Right;

	public new static Expression? TryParse(Method context, string input)
	{
		var parts = input.Split(' ', 3);
		return parts.Length == 3 && parts[1].IsOperator()
			? TryParseBinary(context, parts)
			: null;
	}

	private static Expression TryParseBinary(Method method, string[] parts)
	{
		var left = method.TryParse(parts[0]) ??
			throw new MethodExpressionParser.UnknownExpression(method, parts[0]);
		var binaryOperator = parts[1];
		var right = method.TryParse(parts[2]) ??
			throw new MethodExpressionParser.UnknownExpression(method, parts[2]);
		var operatorMethod = left.ReturnType.Methods.FirstOrDefault(m => m.Name == binaryOperator) ??
			method.GetType(Base.BinaryOperator).Methods.First(m => m.Name == binaryOperator);
		return new Binary(left, operatorMethod, right);
	}
}