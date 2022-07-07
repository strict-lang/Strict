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
		var parts = GetParts(input, operatorText);
		//var parts = input.Split('(', ')', ' ');
		//var binaryParts = new string[3];
		//var firstPart = "";
		//if (parts.Length == 3 && parts[1].IsOperator())
		//	return TryParseBinary(line, parts);
		//if (parts[0] == "" && parts[2] != ",")
		//{
		//	for (var index = 1; index < parts.Length; index++)
		//	{
		//		if (parts[index] == "")
		//		{
		//			binaryParts[0] = firstPart;
		//			binaryParts[1] = parts[index + 1];
		//			binaryParts[2] = input[(index + 6)..];
		//			break;
		//		}
		//		firstPart += parts[index] + " ";
		//	}
		//	return TryParseBinary(line, binaryParts);
		//}
		return TryParseBinary(line, parts);
	}

	private static string[] GetParts(string input, string operatorText)
	{
		var parts = new string[3];
		if (input.Contains('(') && input.Contains(')') &&
			input[(input.IndexOf('(') + 1)..input.IndexOf(')')].HasOperator(out _))
		{
			if (input.IndexOf('(') == 0)
			{
				parts[0] = input[(input.IndexOf('(') + 1)..input.IndexOf(')')];
				parts[1] = input.Substring(input.IndexOf(')') + 2, 1);
				parts[2] = input[(input.IndexOf(')') + 3)..].Trim();
			}
			else if (input.IndexOf(')') < input.Length - 1)
			{
				parts[0] = input[..(input.IndexOf('(') - 2)];
				parts[1] = input[2..(input.IndexOf('(') - 1)];
				parts[2] = input[input.IndexOf('(')..];
			}
			else
			{
				parts[0] = input[..(input.IndexOf('(') - 2)];
				parts[1] = input[2..(input.IndexOf('(') - 1)];
				parts[2] = input[(input.IndexOf('(') + 1)..input.IndexOf(')')];
			}
		}
		else
		{
			parts[0] = input[..(input.IndexOf(operatorText, StringComparison.Ordinal) - 1)];
			parts[1] = operatorText;
			parts[2] =
				input[(input.IndexOf(operatorText, StringComparison.Ordinal) + operatorText.Length + 1)..];
		}
		return parts;
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
		if (left.ReturnType == line.Method.GetType(Base.Any))
			throw new AnyIsNotAllowed(line.Method, left);
		if (right.ReturnType == line.Method.GetType(Base.Any))
			throw new AnyIsNotAllowed(line.Method, right);
		var operatorMethod = left.ReturnType.Methods.FirstOrDefault(m => m.Name == parts[1]) ??
			line.Method.GetType(Base.BinaryOperator).Methods.FirstOrDefault(m => m.Name == parts[1]) ??
			throw new NoMatchingOperatorFound(left.ReturnType, parts[1]);
		return new Binary(left, operatorMethod, right);
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
