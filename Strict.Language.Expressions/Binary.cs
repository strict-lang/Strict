using System.Linq;

namespace Strict.Language.Expressions
{
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
			return parts.Length == 3 && parts[1][0].IsOperator()
				? TryParseBinary(context, parts)
				: null;
		}

		private static Expression TryParseBinary(Method context, string[] parts)
		{
			var left = MethodExpressionParser.TryParse(context, parts[0]);
			if (left == null)
				throw new InvalidExpression(parts.ToText(" "), nameof(left));
			var binaryOperator = parts[1];
			var right = MethodExpressionParser.TryParse(context, parts[2]);
			if (right == null)
				throw new InvalidExpression(parts.ToText(" "), nameof(right));
			return new Binary(left, left.ReturnType.Methods.First(m => m.Name == binaryOperator),
				right);
		}
	}
}