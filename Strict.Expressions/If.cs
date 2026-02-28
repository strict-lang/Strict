using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict.Expressions;

/// <summary>
/// If expressions are used for branching, can also be used as an input for any other expression
/// like method arguments, other conditions, etc. like conditional operators.
/// </summary>
public sealed class If(Expression condition, Expression then, int lineNumber = 0,
	Expression? optionalElse = null, Body? bodyForErrorMessage = null) : Expression(
	CheckExpressionAndGetMatchingType(then, optionalElse, bodyForErrorMessage), lineNumber)
{
	private const string IfPrefix = "if ";
	private const string IfKeyword = "if";
	public const string ThenSeparator = " then ";
	private const string ElseSeparator = " else ";
	private const string ElsePrefix = "else ";
	private const string ElseKeyword = "else";
	private const string SelectorSuffix = " is";

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
			: IfPrefix + Condition + Environment.NewLine + IndentExpression(Then) + (OptionalElse != null
				? OptionalElse is If
					? Environment.NewLine + ElsePrefix + OptionalElse
					: Environment.NewLine + ElseKeyword + Environment.NewLine + IndentExpression(OptionalElse)
				: "");

	public override bool IsConstant =>
		Condition.IsConstant && Then.IsConstant && (OptionalElse?.IsConstant ?? true);

	public override bool Equals(Expression? other) =>
		other is If otherIf && Condition.Equals(otherIf.Condition) && Then.Equals(otherIf.Then) &&
		(OptionalElse?.Equals(otherIf.OptionalElse) ?? otherIf.OptionalElse == null);

	public static Expression? TryParse(Body body, ReadOnlySpan<char> line) =>
		line.Equals(IfKeyword, StringComparison.Ordinal)
			? throw new MissingCondition(body)
			: line.Equals(ElseKeyword, StringComparison.Ordinal) ||
			line.StartsWith("else if", StringComparison.Ordinal)
				? throw new UnexpectedElse(body)
				: line.StartsWith(IfPrefix, StringComparison.Ordinal)
					? TryParseIf(body, line)
					: null;

	public sealed class MissingCondition(Body body) : ParsingFailed(body);
	public sealed class UnexpectedElse(Body body) : ParsingFailed(body);

	private static Expression TryParseIf(Body body, ReadOnlySpan<char> line)
	{
		var trimmedLine = line.TrimEnd();
		if (trimmedLine.EndsWith(SelectorSuffix, StringComparison.Ordinal))
			return ParseSelectorIf(body, trimmedLine);
		var condition = GetConditionExpression(body, line[3..]);
		var thenBody = body.FindCurrentChild();
		if (thenBody == null)
			throw new MissingThen(body);
		var then = thenBody.Parse();
		return new If(condition, then, body.CurrentFileLineNumber, HasRemainingBody(body)
			? CreateElseIfOrElse(body, body.GetLine(body.ParsingLineNumber + 1).AsSpan(body.Tabs))
			: null, body);
	}

	private static Expression ParseSelectorIf(Body body, ReadOnlySpan<char> line)
	{
		var trimmedLine = line.TrimEnd();
		var selectorText = trimmedLine[3..^SelectorSuffix.Length];
		if (selectorText.IsEmpty)
			throw new MissingCondition(body); //ncrunch: no coverage
		var selector = body.Method.ParseExpression(body, selectorText);
		var thenBody = body.FindCurrentChild();
		if (thenBody == null)
			throw new MissingThen(body); //ncrunch: no coverage
		var cases = ParseSelectorCases(thenBody, selector, out var optionalElse);
		return new SelectorIf(selector, cases, body.CurrentFileLineNumber, optionalElse, body);
	}

	private static IReadOnlyList<SelectorIf.Case> ParseSelectorCases(Body body, Expression selector,
		out Expression? optionalElse)
	{
		optionalElse = null;
		var cases = new List<SelectorIf.Case>();
		for (var lineNumber = body.LineRange.Start.Value; lineNumber < body.LineRange.End.Value;
			lineNumber++)
		{
			var line = body.GetLine(lineNumber).AsSpan(body.Tabs).TrimStart();
			if (line.StartsWith(ElsePrefix, StringComparison.Ordinal) ||
				line.Equals(ElseKeyword, StringComparison.Ordinal))
			{
				if (optionalElse != null || lineNumber + 1 < body.LineRange.End.Value)
					throw new UnexpectedElse(body); //ncrunch: no coverage
				var elseText = line.Length > ElsePrefix.Length
					? line[ElsePrefix.Length..].TrimStart()
					: ReadOnlySpan<char>.Empty;
				if (elseText.IsEmpty)
					throw new MissingElseExpression(body); //ncrunch: no coverage
				optionalElse = body.Method.ParseExpression(body, elseText);
				break;
			}
			var thenIndex = line.IndexOf(ThenSeparator, StringComparison.Ordinal);
			if (thenIndex < 0)
				throw new MissingThen(body); //ncrunch: no coverage
			var patternText = line[..thenIndex].TrimEnd();
			var thenText = line[(thenIndex + ThenSeparator.Length)..].TrimStart();
			var pattern = body.Method.ParseExpression(body, patternText);
			var thenExpression = body.Method.ParseExpression(body, thenText);
			var condition = CreateSelectorCondition(selector, pattern);
			cases.Add(new SelectorIf.Case(pattern, thenExpression, condition));
		}
		return cases;
	}

	private static Expression CreateSelectorCondition(Expression selector, Expression pattern)
	{
		var arguments = new[] { pattern };
		return new Binary(selector, selector.ReturnType.GetMethod(BinaryOperator.Is, arguments),
			arguments);
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
			? body.Method.FolderName + "\n Return type " + conditionReturnType + " is not " + Base.Boolean
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

	public sealed class MissingElseExpression(Body body) : ParsingFailed(body);
}