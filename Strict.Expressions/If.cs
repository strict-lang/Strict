using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict.Expressions;

/// <summary>
/// If expressions are used for branching, can also be used as an input for any other expression
/// like method arguments, other conditions, etc. like conditional operators.
/// </summary>
public sealed class If(
	Expression condition,
	Expression then,
	int lineNumber = 0,
	Expression? optionalElse = null,
	Body? bodyForErrorMessage = null) : Expression(
	CheckExpressionAndGetMatchingType(then, optionalElse, bodyForErrorMessage), lineNumber)
{
	private static Type CheckExpressionAndGetMatchingType(Expression then, Expression? optionalElse,
		Body? bodyForErrorMessage) =>
		then is Declaration || optionalElse is Declaration
			? then.ReturnType
			: GetMatchingType(then.ReturnType, optionalElse?.ReturnType, bodyForErrorMessage);

	/// <summary>
	/// The return type of the whole if expression must be a type compatible to both what the then
	/// expression and the else expression return (if used). This is a common problem in static
	/// programming languages, and we can fix it here by evaluating both types and finding a common
	/// base type. If that is not possible, there is a compilation error here.
	/// </summary>
	private static Type GetMatchingType(Type thenType, Type? elseType, Body? bodyForErrorMessage) =>
		elseType == null || elseType.IsSameOrCanBeUsedAs(thenType, false) || elseType.IsError
			? thenType
			: thenType.IsSameOrCanBeUsedAs(elseType, false) || thenType.IsError
				? elseType
				: thenType.FindFirstUnionType(elseType) ??
				throw new ReturnTypeOfThenAndElseMustHaveMatchingType(
					bodyForErrorMessage ?? new Body(thenType.Methods[0]), thenType, elseType);

	public class ReturnTypeOfThenAndElseMustHaveMatchingType(
		Body body,
		Type thenReturnType,
		Type optionalElseReturnType) : ParsingFailed(body,
		"The Then type: " + thenReturnType + " is not same as the Else type: " +
		optionalElseReturnType);

	public Expression Condition { get; } = condition;
	public Expression Then { get; } = then;
	/// <summary>
	/// Rarely used as most of the time Then will return, and anything after is automatically the
	/// else (no else needed). For multiple if/else or when not returning else might be useful.
	/// </summary>
	public Expression? OptionalElse { get; } = optionalElse;

	public override int GetHashCode() =>
		Condition.GetHashCode() ^ Then.GetHashCode() ^ (OptionalElse?.GetHashCode() ?? 0);

	public override string ToString() =>
		OptionalElse != null && (OptionalElse.ReturnType.IsSameOrCanBeUsedAs(Then.ReturnType) ||
			Then.ReturnType.IsError || OptionalElse.ReturnType.IsError) && Then is not Body &&
		OptionalElse is not Body && OptionalElse is not If
			? Condition + ThenSeparator + Then + ElseSeparator + OptionalElse
			: "if " + Condition + Environment.NewLine + IndentExpression(Then) + (OptionalElse != null
				? OptionalElse is If
					? Environment.NewLine + "else " + OptionalElse
					: Environment.NewLine + "else" + Environment.NewLine + IndentExpression(OptionalElse)
				: "");

	public override bool IsConstant =>
		Condition.IsConstant && Then.IsConstant && (OptionalElse?.IsConstant ?? true);

	public override bool Equals(Expression? other) =>
		other is If otherIf && Condition.Equals(otherIf.Condition) && Then.Equals(otherIf.Then) &&
		(OptionalElse?.Equals(otherIf.OptionalElse) ?? otherIf.OptionalElse == null);

	public static Expression? TryParse(Body body, ReadOnlySpan<char> line) =>
		line.Equals("if", StringComparison.Ordinal)
			? throw new MissingCondition(body)
			: line.Equals("else", StringComparison.Ordinal) ||
			line.StartsWith("else if", StringComparison.Ordinal)
				? throw new UnexpectedElse(body)
				: line.StartsWith("if ", StringComparison.Ordinal)
					? TryParseIf(body, line)
					: null;

	public sealed class MissingCondition(Body body) : ParsingFailed(body);
	public sealed class UnexpectedElse(Body body) : ParsingFailed(body);

	private static Expression TryParseIf(Body body, ReadOnlySpan<char> line)
	{
		var condition = GetConditionExpression(body, line[3..]);
		var thenBody = body.FindCurrentChild();
		if (thenBody == null)
			throw new MissingThen(body);
		var then = thenBody.Parse();
		return new If(condition, then, body.CurrentFileLineNumber, HasRemainingBody(body)
			? CreateElseIfOrElse(body, body.GetLine(body.ParsingLineNumber + 1).AsSpan(body.Tabs))
			: null, body);
	}

	private static Expression GetConditionExpression(Body body, ReadOnlySpan<char> line)
	{
		var condition = body.Method.ParseExpression(body, line);
		var booleanType = condition.ReturnType.GetType(Base.Boolean);
		if (condition.ReturnType == booleanType ||
			booleanType.IsSameOrCanBeUsedAs(condition.ReturnType, false))
			return condition;
		throw new InvalidCondition(body, condition.ReturnType);
	}

	public sealed class InvalidCondition(Body body, Type? conditionReturnType = null)
		: ParsingFailed(body, conditionReturnType != null
			? body.Method.FullName + "\n Return type " + conditionReturnType + " is not " + Base.Boolean
			: null);

	public sealed class MissingThen(Body body) : ParsingFailed(body);

	private static bool HasRemainingBody(Body body) =>
		body.ParsingLineNumber + 1 < body.LineRange.End.Value;

	private static Expression? CreateElseIfOrElse(Body body, ReadOnlySpan<char> line) =>
		HasElseIf(line)
			? TryParseIf(body, body.GetLine(body.ParsingLineNumber++ + 1).AsSpan(body.Tabs + 5))
			: HasOnlyElse(line)
				? CreateElse(body)
				: null;

	private static bool HasElseIf(ReadOnlySpan<char> line) =>
		line.StartsWith("else if ", StringComparison.Ordinal);

	private static bool HasOnlyElse(ReadOnlySpan<char> line) =>
		line.Equals("else", StringComparison.Ordinal);

	private static Expression CreateElse(Body body)
	{
		body.ParsingLineNumber++;
		var elseBody = body.FindCurrentChild();
		return elseBody == null
			? throw new MissingElseExpression(body)
			: elseBody.Parse();
	}

	public static bool CanTryParseConditional(Body body, ReadOnlySpan<char> input)
	{
		var thenIndex = input.IndexOf(ThenSeparator, StringComparison.Ordinal);
		var firstBracket = input.IndexOf('(');
		if (thenIndex > 0 && NoFirstBracketOrSurroundedByIt(input, firstBracket, thenIndex))
			return CountThenSeparators(input) > 1
				? throw new ConditionalExpressionsCannotBeNested(body)
				: true;
		return false;
	}

	private static bool NoFirstBracketOrSurroundedByIt(ReadOnlySpan<char> input, int firstBracket,
		int separatorIndex) =>
		firstBracket == -1 || firstBracket > separatorIndex || input.IndexOf(')') < separatorIndex ||
		firstBracket == 0 && input[^1] == ')';

	public sealed class ConditionalExpressionsCannotBeNested(Body body) : ParsingFailed(body);

	public static Expression ParseConditional(Body body, ReadOnlySpan<char> input)
	{
		if (input[0] == '(' && input[^1] == ')')
			input = input[1..^1];
		var thenIndex = input.IndexOf(ThenSeparator, StringComparison.Ordinal);
		if (thenIndex < 1)
			throw new InvalidCondition(body); //ncrunch: no coverage
		var elseIndex = input.IndexOf(ElseSeparator, StringComparison.Ordinal);
		if (elseIndex <= thenIndex)
			throw new MissingElseExpression(body);
		return new If(GetConditionExpression(body, input[..thenIndex]),
			body.Method.ParseExpression(body, input[(thenIndex + ThenSeparator.Length)..elseIndex]),
			body.CurrentFileLineNumber,
			body.Method.ParseExpression(body, input[(elseIndex + ElseSeparator.Length)..]));
	}

	private static int CountThenSeparators(ReadOnlySpan<char> input)
	{
		var count = 0;
		for (var index = 0; index <= input.Length - ThenSeparator.Length; index++)
			if (input[index..].StartsWith(ThenSeparator, StringComparison.Ordinal))
			{
				count++;
				index += ThenSeparator.Length - 1;
			}
		return count;
	}

	public const string ThenSeparator = " then ";
	private const string ElseSeparator = " else ";

	public sealed class MissingElseExpression(Body body) : ParsingFailed(body);
}