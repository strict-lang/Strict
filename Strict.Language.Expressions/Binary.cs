using System;
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
		var operatorToken = line.Text[operatorTokenRange];
		if (operatorToken == "*" && HasIncompatibleDimensions(left, right))
			throw new ListsHaveDifferentDimensions(line, left + " " + right);
		var arguments = new[] { right };
		return new Binary(left, left.ReturnType.GetMethod(operatorToken, arguments), arguments);
	}

	private static Expression GetUnaryOrBuildNestedBinary(Method.Line line, Range nextTokenRange,
		Stack<Range> tokens) =>
		line.Text[nextTokenRange.Start.Value].IsSingleCharacterOperator() ||
		line.Text.GetSpanFromRange(nextTokenRange).IsMultiCharacterOperator()
			? BuildBinaryExpression(line, nextTokenRange, tokens)
			: line.Method.ParseExpression(line, nextTokenRange);

	public static bool HasIncompatibleDimensions(Expression left, Expression right) =>
		left is List leftList && right is List rightList &&
		leftList.Values.Count != rightList.Values.Count;

	public sealed class ListsHaveDifferentDimensions : ParsingFailed
	{
		public ListsHaveDifferentDimensions(Method.Line line, string error) : base(line, error) { }
	}
}
