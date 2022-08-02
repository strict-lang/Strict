using System;
using System.Collections.Generic;
using System.Linq;

namespace Strict.Language.Expressions;

public sealed class Binary : MethodCall
{
	public Binary(Expression left, Method operatorMethod, Expression[] right) :
		base(operatorMethod, left, right) { }

	//TODO: we should check if groupings are correctly restored on ToString, e.g. (1 + 2) * 3 should not be regenerated as 1 + 2 * 3! Adding brackets around each binary expression will save the original expression but cumbersome to read? Confirm with Ben
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
		if (operatorToken == "*" && MethodExpressionParser.HasIncompatibleDimensions(left, right))
			throw new MethodExpressionParser.ListsHaveDifferentDimensions(line, left + " " + right);
		var arguments = new[] { right };
		return new Binary(left, left.ReturnType.GetMethod(operatorToken, arguments), arguments);
	}

	private static Expression GetUnaryOrBuildNestedBinary(Method.Line line, Range nextTokenRange,
		Stack<Range> tokens) =>
		line.Text[nextTokenRange.Start.Value].IsSingleCharacterOperator() ||
		line.Text.GetSpanFromRange(nextTokenRange).IsMultiCharacterOperator()
			? BuildBinaryExpression(line, nextTokenRange, tokens)
			: line.Method.ParseExpression(line, nextTokenRange);

	public sealed class MismatchingTypeFound : ParsingFailed
	{
		public MismatchingTypeFound(Method.Line line, string error = "") : base(line, error) { }
	}
}
