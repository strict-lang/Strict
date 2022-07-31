using System;
using System.Collections.Generic;
using System.Linq;

namespace Strict.Language.Expressions;

public sealed class Binary : MethodCall
{
	public Binary(Expression left, Method operatorMethod, Expression right) : base(operatorMethod,
		left, right) { }

	//TODO: we should check if groupings are correctly restored on ToString, e.g. (1 + 2) * 3 should not be regenerated as 1 + 2 * 3!
	public override string ToString() => Instance + " " + Method.Name + " " + Arguments[0];

	public static Expression Parse(Method.Line line, Stack<Range> postfixTokens) =>
		postfixTokens.Count < 3
			? throw new IncompleteTokensForBinaryExpression(line, postfixTokens)
			: BuildBinaryExpression(line, postfixTokens.Pop(), postfixTokens);

	public sealed class IncompleteTokensForBinaryExpression : ParsingFailed
	{
		public IncompleteTokensForBinaryExpression(Method.Line line, IEnumerable<Range> postfixTokens) :
			base(line, postfixTokens.Select(t => line.Text[t]).Reverse().ToWordList()) { }
	}

	private static Expression BuildBinaryExpression(Method.Line line, Range operatorTokenRange, Stack<Range> tokens)
	{
		var right = GetUnaryOrBuildNestedBinary(line, tokens.Pop(), tokens);
		var left = GetUnaryOrBuildNestedBinary(line, tokens.Pop(), tokens);
		if (MethodExpressionParser.HasMismatchingTypes(left, right))
			throw new MismatchingTypeFound(line);
		var operatorToken = line.Text[operatorTokenRange]; //TODO: make more efficient
		if (operatorToken == ",")
			throw new NoMatchingOperatorFound(left.ReturnType,
				operatorToken + " should be used for lists, not " + nameof(Binary));
		if (operatorToken == "*" && MethodExpressionParser.HasIncompatibleDimensions(left, right))
			throw new MethodExpressionParser.ListsHaveDifferentDimensions(line, left + " " + right);
		// TODO: Match operator param types before
		var operatorMethod = left.ReturnType.FindMethod(operatorToken) ??
			//not longer needed: line.Method.GetType(Base.BinaryOperator).Methods.FirstOrDefault(m => m.Name == operatorToken) ??
			throw new NoMatchingOperatorFound(right.ReturnType, operatorToken);
		return new Binary(left, operatorMethod, right);
	}

	private static Expression GetUnaryOrBuildNestedBinary(Method.Line line, Range nextTokenRange,
		Stack<Range> tokens)
	{
		var nextToken = line.Text.GetSpanFromRange(nextTokenRange);//TODO: only needed for operator check, can be done more efficiently
		return nextToken[0].IsSingleCharacterOperator() || nextToken.IsMultiCharacterOperator()
			? BuildBinaryExpression(line, nextTokenRange, tokens)
			: line.Method.ParseExpression(line, nextTokenRange);
	}

	//TODO; Remove
	//private static Expression TryParseBinary(Method.Line line, IReadOnlyList<string> parts)
	//{
	//	var left = line.Method.TryParseExpression(line, parts[0]) ??
	//		throw new MethodExpressionParser.UnknownExpression(line, parts[0]);
	//	var right = line.Method.TryParseExpression(line, parts[2]) ??
	//		throw new MethodExpressionParser.UnknownExpression(line, parts[2]);
	//	if (List.HasMismatchingTypes(left, right))
	//		throw new MismatchingTypeFound(line, parts[2]);
	//	if (parts[1] == "*" && List.HasIncompatibleDimensions(left, right))
	//		throw new List.ListsHaveDifferentDimensions(line, parts[0] + " " + parts[2]);
	//	CheckForAnyExpressions(line, left, right);
	//	var operatorMethod = left.ReturnType.Methods.FirstOrDefault(m => m.Name == parts[1]) ??
	//		line.Method.GetType(Base.BinaryOperator).Methods.FirstOrDefault(m => m.Name == parts[1]) ??
	//		throw new NoMatchingOperatorFound(left.ReturnType, parts[1]);
	//	return new Binary(left, operatorMethod, right);
	//}

	// TODO: check if this needs to be called anywhere
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
		public NoMatchingOperatorFound(Type leftType, string operatorMethod) : base(nameof(leftType) + "=" + leftType + " does not contain " + operatorMethod) { }
	}
}
