//#define LOG_OPERATORS_PARSING
using Strict.Language;

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
		child is MethodCall binaryOrUnary && BinaryOperator.GetPrecedence(binaryOrUnary.Method.Name) <
		BinaryOperator.GetPrecedence(Method.Name) || child is If
			? $"({child})"
			: child.ToString();

	public static Expression
		Parse(Body body, ReadOnlySpan<char> input, Stack<Range> postfixTokens)
	{
#if LOG_OPERATORS_PARSING
		Console.WriteLine();
		Console.WriteLine("Binary.Parse " + input.ToString() + ", postfixTokens=" + postfixTokens.Count);
#endif
		return postfixTokens.Count < 3
			? throw new IncompleteTokensForBinaryExpression(body, input, postfixTokens)
			: BuildBinaryExpression(body, input, postfixTokens.Pop(), postfixTokens);
	}

	public sealed class IncompleteTokensForBinaryExpression(Body body, ReadOnlySpan<char> input,
		IEnumerable<Range> postfixTokens) : ParsingFailed(body, //ncrunch: no coverage
		input.GetTextsFromRanges(postfixTokens).Reverse().ToWordList());

	private static Expression BuildBinaryExpression(Body body, ReadOnlySpan<char> input,
		Range operatorTokenRange, Stack<Range> tokens)
	{
		// If just "in" was passed, check if it was preceded by an "is" or "is not" (or be in for)
		if (input.Contains(" in ", StringComparison.Ordinal) &&
			!input.Contains(" is in ", StringComparison.Ordinal) &&
			!input.Contains(" is not in ", StringComparison.Ordinal))
			throw new InMustAlwaysBePrecededByIsOrIsNot(input.ToString());
		var operatorToken = input[operatorTokenRange].ToString();
#if LOG_OPERATORS_PARSING
		Console.WriteLine("BuildBinaryExpression operator=" + operatorToken + ", remaining tokens=" +
			tokens.Count);
#endif
		return operatorToken switch
		{
			BinaryOperator.To => To.Parse(body, input[tokens.Pop()],
				GetUnaryOrBuildNestedBinary(body, input, tokens)),
			UnaryOperator.Not => BuildNotBinaryExpression(body, input, tokens),
			_ => BuildRegularBinaryExpression(body, input, tokens, operatorToken)
		};
	}

	public sealed class InMustAlwaysBePrecededByIsOrIsNot(string input) : Exception(input);

	private static Expression BuildNotBinaryExpression(Body body, ReadOnlySpan<char> input,
		Stack<Range> tokens) =>
		BuildNot(tokens.Count == 1
			? GetUnaryOrBuildNestedBinary(body, input, tokens)
			: BuildRegularBinaryExpression(body, input, tokens, input[tokens.Pop()].ToString()));

	private static Binary BuildRegularBinaryExpression(Body body, ReadOnlySpan<char> input,
		Stack<Range> tokens, string operatorToken)
	{
#if LOG_OPERATORS_PARSING
		Console.WriteLine("BuildRegularBinaryExpression operatorToken=" + operatorToken + ", next token=" +
			input[tokens.Peek()].ToString() + ", remaining tokens=" + tokens.Count);
#endif
		var right = GetUnaryOrBuildNestedBinary(body, input, tokens,
			operatorToken is BinaryOperator.Is);
#if LOG_OPERATORS_PARSING
		Console.WriteLine("BuildRegularBinaryExpression right=" + right + ", next token=" +
			input[tokens.Peek()].ToString() + ", remaining tokens=" + tokens.Count);
#endif
		var left = GetUnaryOrBuildNestedBinary(body, input, tokens);
#if LOG_OPERATORS_PARSING
		Console.WriteLine("BuildRegularBinaryExpression left=" + left + ", next token=" +
			(tokens.Count > 0
				? input[tokens.Peek()].ToString()
				: "<empty>") + ", remaining tokens=" + tokens.Count);
#endif
		// Any incompatibility is checked at runtime when the Executor runs on this
		return operatorToken is BinaryOperator.In
			? new Binary(right, right.ReturnType.GetMethod(BinaryOperator.In, [left]), [left])
			: new Binary(left, left.ReturnType.GetMethod(operatorToken, [right]), [right]);
	}

	private static Expression GetUnaryOrBuildNestedBinary(Body body, ReadOnlySpan<char> input,
		Stack<Range> tokens, bool checkRightForIsTypeComparison = false)
	{
		var nextTokenRange = tokens.Pop();
#if LOG_OPERATORS_PARSING
		Console.WriteLine("GetUnaryOrBuildNestedBinary token=" +
			input[nextTokenRange].ToString() + ", remaining tokens=" + tokens.Count);
#endif
		return input[nextTokenRange].IsNot()
			? BuildNot(GetUnaryOrBuildNestedBinary(body, input, tokens, checkRightForIsTypeComparison))
			: nextTokenRange.End.Value == nextTokenRange.Start.Value + 1 &&
			input[nextTokenRange.Start.Value].IsSingleCharacterOperator() ||
			input[nextTokenRange].IsMultiCharacterOperator()
				? BuildBinaryExpression(body, input, nextTokenRange, tokens)
				: checkRightForIsTypeComparison
					? TypeComparison.Parse(body, input, nextTokenRange)
					: body.Method.ParseExpression(body, input[nextTokenRange]);
	}

	private static Expression BuildNot(Expression expression) =>
		new Not(expression.ReturnType.GetMethod(UnaryOperator.Not, []), expression);
}