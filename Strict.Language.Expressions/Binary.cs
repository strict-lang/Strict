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
		var operatorText = input.FindFirstOperator();
		if (operatorText == null)
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
		CheckForAnyExpressions(line, left, right);
		var operatorMethod = left.ReturnType.Methods.FirstOrDefault(m => m.Name == parts[1]) ??
			line.Method.GetType(Base.BinaryOperator).Methods.FirstOrDefault(m => m.Name == parts[1]) ??
			throw new NoMatchingOperatorFound(left.ReturnType, parts[1]);
		return new Binary(left, operatorMethod, right);
	}

	private static void CheckForAnyExpressions(Method.Line line, Expression left, Expression right)
	{
		if (left.ReturnType == line.Method.GetType(Base.Any))
			throw new AnyIsNotAllowed(line.Method, left);
		if (right.ReturnType == line.Method.GetType(Base.Any))
			throw new AnyIsNotAllowed(line.Method, right);
	}

	private sealed class AnyIsNotAllowed : Exception
	{
		public AnyIsNotAllowed(Method lineMethod, Expression operand) : base("\n" + lineMethod +
			"\n" + string.Join('\n', lineMethod.bodyLines) + "\noperand=" + operand + ", type=" +
			operand.ReturnType) { }
	}

	public sealed class MismatchingTypeFound : ParsingFailed
	{
		public MismatchingTypeFound(Method.Line line, string error = "") : base(line, error) { }
	}

	public sealed class NoMatchingOperatorFound : Exception
	{
		public NoMatchingOperatorFound(Type leftType, string operatorMethod) : base(nameof(leftType) + "=" + leftType + " or " + Base.BinaryOperator + " does not contain " + operatorMethod) { }
	}
}
