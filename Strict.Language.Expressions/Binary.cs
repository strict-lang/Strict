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
		BinaryOperator.GetPrecedence(Method.Name) || child is If
			? $"({child})"
			: child.ToString();

	public static Expression
		Parse(Body body, ReadOnlySpan<char> input, Stack<Range> postfixTokens) =>
		postfixTokens.Count < 3
			? throw new IncompleteTokensForBinaryExpression(body, input, postfixTokens)
			: BuildBinaryExpression(body, input, postfixTokens.Pop(), postfixTokens);

	public sealed class IncompleteTokensForBinaryExpression : ParsingFailed
	{
		public IncompleteTokensForBinaryExpression(Body body, ReadOnlySpan<char> input,
			IEnumerable<Range> postfixTokens) :
			base(body, input.GetTextsFromRanges(postfixTokens).Reverse().ToWordList()) { } //ncrunch: no coverage
	}

	private static Expression BuildBinaryExpression(Body body, ReadOnlySpan<char> input,
		Range operatorTokenRange, Stack<Range> tokens)
	{
		var operatorToken = input[operatorTokenRange].ToString();
		return operatorToken == BinaryOperator.To
			? To.Parse(body, input[tokens.Pop()], GetUnaryOrBuildNestedBinary(body, input, tokens))
			: BuildRegularBinaryExpression(body, input, tokens, operatorToken);
	}

	private static Expression BuildRegularBinaryExpression(Body body, ReadOnlySpan<char> input,
		Stack<Range> tokens, string operatorToken)
	{
		var right = GetUnaryOrBuildNestedBinary(body, input, tokens);
		var left = GetUnaryOrBuildNestedBinary(body, input, tokens);
		if (operatorToken == BinaryOperator.Multiply && HasIncompatibleDimensions(left, right))
			throw new ListsHaveDifferentDimensions(body, left + " " + right);
		var arguments = new[] { right };
		return new Binary(left,
			left.ReturnType.GetMethod(operatorToken, arguments, body.Method.Parser), arguments);
	}

	private static Expression GetUnaryOrBuildNestedBinary(Body body, ReadOnlySpan<char> input,
		Stack<Range> tokens)
	{
		var nextTokenRange = tokens.Pop();
		var expression = input[nextTokenRange.Start.Value].IsSingleCharacterOperator() ||
			input[nextTokenRange].IsMultiCharacterOperator()
				? BuildBinaryExpression(body, input, nextTokenRange, tokens)
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

	public sealed class ListsHaveDifferentDimensions : ParsingFailed
	{
		public ListsHaveDifferentDimensions(Body body, string error) : base(body, error) { }
	}
}