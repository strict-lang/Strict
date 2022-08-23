﻿using System;
using System.Collections.Generic;
using System.Linq;

namespace Strict.Language.Expressions;

public sealed class Binary : MethodCall
{
	public Binary(Expression left, Method operatorMethod, Expression[] right) :
		base(operatorMethod, left, right) { }

	public override string ToString() =>
		AddNestedBracketsIfNeeded(Instance!) + " " + Method.Name + " " +
		AddNestedBracketsIfNeeded(Arguments[0]);

	private string AddNestedBracketsIfNeeded(Expression child) =>
		child is Binary childBinary && BinaryOperator.GetPrecedence(childBinary.Method.Name) <
		BinaryOperator.GetPrecedence(Method.Name)
			? $"({child})"
			: child.ToString();

	public static Expression Parse(Body body, ReadOnlySpan<char> input, Stack<Range> postfixTokens) =>
		postfixTokens.Count < 3
			? throw new IncompleteTokensForBinaryExpression(body, input, postfixTokens)
			: BuildBinaryExpression(body, input, postfixTokens.Pop(), postfixTokens);

	public sealed class IncompleteTokensForBinaryExpression : ParsingFailed
	{
		public IncompleteTokensForBinaryExpression(Body body, ReadOnlySpan<char> input,
			IEnumerable<Range> postfixTokens) :
			base(body, input.GetTextsFromRanges(postfixTokens).Reverse().ToWordList()) { }
	}

	private static Expression BuildBinaryExpression(Body body, ReadOnlySpan<char> input, Range operatorTokenRange, Stack<Range> tokens)
	{
		var right = GetUnaryOrBuildNestedBinary(body, input, tokens.Pop(), tokens);
		var left = GetUnaryOrBuildNestedBinary(body, input, tokens.Pop(), tokens);
		var operatorToken = input[operatorTokenRange].ToString();
		if (operatorToken == "*" && HasIncompatibleDimensions(left, right))
			throw new ListsHaveDifferentDimensions(body, left + " " + right);
		var arguments = new[] { right };
		return new Binary(left, left.ReturnType.GetMethod(operatorToken, arguments), arguments);
	}

	private static Expression GetUnaryOrBuildNestedBinary(Body body, ReadOnlySpan<char> input, Range nextTokenRange,
		Stack<Range> tokens) =>
		input[nextTokenRange.Start.Value].IsSingleCharacterOperator() ||
		input[nextTokenRange].IsMultiCharacterOperator()
			? BuildBinaryExpression(body, input, nextTokenRange, tokens)
			: body.Method.ParseExpression(body, input[nextTokenRange]);

	public static bool HasIncompatibleDimensions(Expression left, Expression right) =>
		left is List leftList && right is List rightList &&
		leftList.Values.Count != rightList.Values.Count;

	public sealed class ListsHaveDifferentDimensions : ParsingFailed
	{
		public ListsHaveDifferentDimensions(Body body, string error) : base(body, error) { }
	}
}