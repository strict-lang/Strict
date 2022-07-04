using System;
using System.Collections.Generic;
using System.Linq;

namespace Strict.Language.Expressions;

public sealed class Binary : MethodCall
{
	public Binary(Expression left, Method operatorMethod, Expression right) : base(left,
		operatorMethod, right) { }

	public Expression Left => Instance!;
	public Expression Right => Arguments[0];
	public override string ToString() => Left + " " + Method.Name + " " + Right;

	public new static Expression? TryParse(Method.Line line, string input)
	{
		if (!input.HasOperator(out var operatorText))
			return null;
		var parts = new string[3];
		parts[0] = input[..(input.IndexOf(operatorText, StringComparison.Ordinal) - 1)];
		parts[1] = operatorText;
		parts[2] =
			input[(input.IndexOf(operatorText, StringComparison.Ordinal) + operatorText.Length + 1)..];
		return TryParseBinary(line, parts);
	}

	private static Expression TryParseBinary(Method.Line line, IReadOnlyList<string> parts)
	{
		var left = line.Method.TryParseExpression(line, parts[0]) ??
			throw new MethodExpressionParser.UnknownExpression(line, parts[0]);
		var right = line.Method.TryParseExpression(line, parts[2]) ??
			throw new MethodExpressionParser.UnknownExpression(line, parts[2]);
		if (List.HasMismatchingTypes(left, right))
			throw new MismatchingTypeFound(line, parts[2]);
		if (parts[1] == "*" && List.HasIncompatibleDimensions(left, right))
			throw new List.ListsHaveDifferentDimensions(line, parts[0] + " " + parts[2]);
		return new Binary(left,
			left.ReturnType.Methods.FirstOrDefault(m => m.Name == parts[1]) ??
			line.Method.
				GetType(Base.BinaryOperator).Methods.First(m => m.Name == parts[1]), right);
	}

	public class MismatchingTypeFound : ParsingFailed
	{
		public MismatchingTypeFound(Method.Line line, string error = "") : base(line, error) { }
	}
}
