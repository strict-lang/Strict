using System.Collections.Generic;
using System.Linq;

namespace Strict.Language.Expressions;

public class Binary : MethodCall
{
	public Binary(Expression left, Method operatorMethod, Expression right) : base(left,
		operatorMethod, right) { }

	public Expression Left => Instance!;
	public Expression Right => Arguments[0];
	public override string ToString() => Left + " " + Method.Name + " " + Right;

	public new static Expression? TryParse(Method.Line line, string input)
	{
		var parts = new string[3];
		if (input.Contains('(') && input.Contains(')') && input.HasOperator(out var operatorIndex))
		{
			parts[0] = input[..operatorIndex].Trim();
			parts[1] = input[operatorIndex].ToString();
			parts[2] = input[(operatorIndex + 1)..].Trim();
		}
		else
		{
			parts = input.Split(' ', 3);
		}
		return parts.Length == 3 && parts[1].IsOperator()
			? TryParseBinary(line, parts)
			: null;
	}

	private static Expression TryParseBinary(Method.Line line, IReadOnlyList<string> parts)
	{
		var left = line.Method.TryParseExpression(line, parts[0]) ??
			throw new MethodExpressionParser.UnknownExpression(line, parts[0]);
		var binaryOperator = parts[1];
		var right = line.Method.TryParseExpression(line, parts[2]) ??
			throw new MethodExpressionParser.UnknownExpression(line, parts[2]);
		if (left is List list && list.Values.First() is not Text && right is Text)
			throw new MismatchingTypeFound(line, parts[2]);
		return new Binary(left,
			left.ReturnType.Methods.FirstOrDefault(m => m.Name == binaryOperator) ?? line.Method.
				GetType(Base.BinaryOperator).Methods.First(m => m.Name == binaryOperator), right);
	}

	public class MismatchingTypeFound : ParsingFailed
	{
		public MismatchingTypeFound(Method.Line line, string error = "") : base(line, error) { }
	}
}
