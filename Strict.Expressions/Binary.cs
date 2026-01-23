using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict.Expressions;

public sealed class Binary(Expression left, Method operatorMethod, Expression[] right)
	: MethodCall(operatorMethod, left, right, null, left.LineNumber)
{
	public override string ToString() =>
		// For "in" we have to swap left and right (in is always implemented in the Iterator)
		Method.Name is BinaryOperator.In
			? AddNestedBracketsIfNeeded(Arguments[0]) + " is in " + AddNestedBracketsIfNeeded(Instance!)
			: AddNestedBracketsIfNeeded(Instance!) + " " + (Method.Name is UnaryOperator.Not
				? "is "
				: "") + Method.Name + " " + AddNestedBracketsIfNeeded(Arguments[0]);

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

	private static Expression BuildBinaryExpression(Body body, ReadOnlySpan<char> input,
		Range operatorTokenRange, Stack<Range> tokens)
	{
		var operatorToken = input[operatorTokenRange].ToString();
		return operatorToken switch
		{
			BinaryOperator.To => To.Parse(body, input[tokens.Pop()],
				GetUnaryOrBuildNestedBinary(body, input, tokens)),
			UnaryOperator.Not => BuildNotBinaryExpression(body, input, tokens),
			BinaryOperator.Is => BuildIsExpression(body, input, tokens, operatorToken),
			_ => BuildRegularBinaryExpression(body, input, tokens, operatorToken)
		};
	}

	private static Expression BuildNotBinaryExpression(Body body, ReadOnlySpan<char> input, Stack<Range> tokens)
	{
		var isExpression = BuildIsExpression(body, input, tokens, input[tokens.Pop()].ToString());
		return new Not(isExpression.ReturnType.GetMethod(UnaryOperator.Not, []), isExpression);
	}

	/// <summary>
	/// Handles special cases like "is in", "is not in", "is not". The "is" here is not
	/// important and stripped, it is not a BinaryExpression itself, the last word is.
	/// </summary>
	private static Expression BuildIsExpression(Body body, ReadOnlySpan<char> input,
		Stack<Range> tokens, string operatorToken)
	{
		switch (input[tokens.Peek()])
		{
		case UnaryOperator.Not:
			operatorToken = input[tokens.Pop()].ToString();
			if (input[tokens.Peek()] == BinaryOperator.In)
			{
				var inExpression = BuildRegularBinaryExpression(body, input, tokens,
					input[tokens.Pop()].ToString());
				return new Not(inExpression.ReturnType.GetMethod(UnaryOperator.Not, []),
					inExpression);
			}
			return BuildRegularBinaryExpression(body, input, tokens, operatorToken);
		case BinaryOperator.In:
			return BuildRegularBinaryExpression(body, input, tokens,
				input[tokens.Pop()].ToString());
		default:
			return BuildRegularBinaryExpression(body, input, tokens, operatorToken);
		}
	}

	// ReSharper disable once TooManyArguments
	private static Binary BuildRegularBinaryExpression(Body body, ReadOnlySpan<char> input,
		Stack<Range> tokens, string operatorToken)
	{
		var right = GetUnaryOrBuildNestedBinary(body, input, tokens,
			operatorToken is BinaryOperator.Is or UnaryOperator.Not);
		var left = GetUnaryOrBuildNestedBinary(body, input, tokens);
		if (operatorToken == BinaryOperator.Multiply && HasIncompatibleDimensions(left, right))
			throw new ListsHaveDifferentDimensions(body, left + " " + right);
		return operatorToken is BinaryOperator.In
			? new Binary(right, right.ReturnType.GetMethod(BinaryOperator.In, [left]), [left])
			: new Binary(left, left.ReturnType.GetMethod(operatorToken, [right]), [right]);
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