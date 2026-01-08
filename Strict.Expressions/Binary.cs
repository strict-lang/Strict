using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict.Expressions;

public sealed class Binary(Expression left, Method operatorMethod, Expression[] right)
	: MethodCall(operatorMethod, left, right, null, left.LineNumber)
{
	public override string ToString() =>
		AddNestedBracketsIfNeeded(Instance!) + " " + Method.Name + " " +
		AddNestedBracketsIfNeeded(Arguments[0]);

	private string AddNestedBracketsIfNeeded(Expression child) =>
		child is Binary childBinary && BinaryOperator.GetPrecedence(childBinary.Method.Name) <
		BinaryOperator.GetPrecedence(Method.Name) || child is If
			? $"({child})"
			: child.ToString();

	public static Expression
		Parse(Body body, ReadOnlySpan<char> input, Stack<Range> postfixTokens) =>
		postfixTokens.Count < 3
			? throw new IncompleteTokensForBinaryExpression(body, input, postfixTokens)
			: BuildBinaryExpression(body, input, postfixTokens.Pop(), postfixTokens);

	public sealed class IncompleteTokensForBinaryExpression(Body body,
		ReadOnlySpan<char> input,	IEnumerable<Range> postfixTokens) : ParsingFailed(body, //ncrunch: no coverage
		input.GetTextsFromRanges(postfixTokens).Reverse().ToWordList());

	// ReSharper disable once TooManyArguments
	private static Expression BuildBinaryExpression(Body body, ReadOnlySpan<char> input,
		Range operatorTokenRange, Stack<Range> tokens)
	{
		var operatorToken = input[operatorTokenRange].ToString();
		return operatorToken == BinaryOperator.To
			? To.Parse(body, input[tokens.Pop()], GetUnaryOrBuildNestedBinary(body, input, tokens))
			: operatorToken == UnaryOperator.Not
				? Not.Parse(body, input, tokens.Pop())
				: BuildRegularBinaryExpression(body, input, tokens, operatorToken);
	}

	// ReSharper disable once TooManyArguments
	private static Binary BuildRegularBinaryExpression(Body body, ReadOnlySpan<char> input,
		Stack<Range> tokens, string operatorToken)
	{
		var right = GetUnaryOrBuildNestedBinary(body, input, tokens,
			operatorToken is BinaryOperator.Is or BinaryOperator.IsNot);
		var left = GetUnaryOrBuildNestedBinary(body, input, tokens);
		if (operatorToken == BinaryOperator.Multiply && HasIncompatibleDimensions(left, right))
			throw new ListsHaveDifferentDimensions(body, left + " " + right);
		var arguments = new[] { right };
		return new Binary(left, operatorToken == BinaryOperator.In
			? right.ReturnType.GetMethod(operatorToken, [left])
			: left.ReturnType.GetMethod(operatorToken, arguments), arguments);
	}

	private static Expression GetUnaryOrBuildNestedBinary(Body body, ReadOnlySpan<char> input,
		Stack<Range> tokens, bool checkRightForIsTypeComparison = false)
	{
		var nextTokenRange = tokens.Pop();
		var expression = input[nextTokenRange.Start.Value].IsSingleCharacterOperator() ||
			input[nextTokenRange].IsMultiCharacterOperator()
				? BuildBinaryExpression(body, input, nextTokenRange, tokens)
				: checkRightForIsTypeComparison
					? TypeComparison.Parse(body, input, nextTokenRange)
					: body.Method.ParseExpression(body, input[nextTokenRange]);
		if (expression.ReturnType.IsGeneric)
			//ncrunch: no coverage start, cannot be reached: Type.FindMethod already filters this condition
			throw new Type.GenericTypesCannotBeUsedDirectlyUseImplementation(expression.ReturnType,
				expression.ToString());
		//ncrunch: no coverage end
		return expression;
	}

	private static bool HasIncompatibleDimensions(Expression left, Expression right) =>
		left is List leftList && right is List rightList &&
		leftList.Values.Count != rightList.Values.Count;

	public sealed class ListsHaveDifferentDimensions(Body body, string error)
		: ParsingFailed(body, error);
}