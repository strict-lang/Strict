namespace Strict.Language.Expressions;

/// <summary>
/// If expressions are used for branching, can also be used as an input for any other expression
/// like method arguments, other conditions, etc. like conditional operators.
/// </summary>
public sealed class If(Expression condition, Expression then,	Expression? optionalElse = null,
	Body? bodyForErrorMessage = null)
	: Expression(CheckExpressionAndGetMatchingType(then, optionalElse, bodyForErrorMessage))
{
	private static Type CheckExpressionAndGetMatchingType(Expression then, Expression? optionalElse,
		Body? bodyForErrorMessage) =>
		then is ConstantDeclaration || optionalElse is ConstantDeclaration
			? then.ReturnType
			: GetMatchingType(then.ReturnType, optionalElse?.ReturnType, bodyForErrorMessage);

	/// <summary>
	/// The return type of the whole if expression must be a type compatible to both what the then
	/// expression and the else expression return (if used). This is a common problem in static
	/// programming languages, and we can fix it here by evaluating both types and find a common base
	/// type. If that is not possible there is a compilation error here.
	/// </summary>
	private static Type GetMatchingType(Type thenType, Type? elseType, Body? bodyForErrorMessage) =>
		elseType == null || elseType.IsCompatible(thenType) || elseType.Name == Base.Error
			? thenType
			: thenType.IsCompatible(elseType) || thenType.Name == Base.Error
				? elseType
				: thenType.FindFirstUnionType(elseType) ??
				throw new ReturnTypeOfThenAndElseMustHaveMatchingType(
					bodyForErrorMessage ?? new Body(thenType.Methods[0]), thenType, elseType);

	public class ReturnTypeOfThenAndElseMustHaveMatchingType(Body body,
		Type thenReturnType, Type optionalElseReturnType) : ParsingFailed(body,
		"The Then type: " + thenReturnType + " is not same as the Else type: " +
		optionalElseReturnType);

	public Expression Condition { get; } = condition;
	public Expression Then { get; } = then;
	/// <summary>
	/// Rarely used as most of the time Then will return and anything after is automatically the else
	/// (else is not allowed then). For multiple if/else or when not returning else might be useful.
	/// </summary>
	public Expression? OptionalElse { get; } = optionalElse;

	public override int GetHashCode() =>
		Condition.GetHashCode() ^ Then.GetHashCode() ^ (OptionalElse?.GetHashCode() ?? 0);

	public override string ToString() =>
		OptionalElse != null && Then.ReturnType == OptionalElse.ReturnType && Then is not Body &&
		OptionalElse is not Body && OptionalElse is not If
			? Condition + " ? " + Then + " else " + OptionalElse
			: "if " + Condition + Environment.NewLine + "\t" + (Then is Body thenBody
				? string.Join(Environment.NewLine + "\t", thenBody.Expressions)
				: Then) + (OptionalElse != null
				? Environment.NewLine + "else" + (OptionalElse is If
					? " "
					: Environment.NewLine + "\t") + (OptionalElse is Body elseBody
					? string.Join(Environment.NewLine + "\t", elseBody.Expressions)
					: OptionalElse)
				: "");

	public override bool Equals(Expression? other) =>
		other is If a && Equals(Condition, a.Condition) && Then.Equals(a.Then) &&
		(OptionalElse?.Equals(a.OptionalElse) ?? a.OptionalElse == null);

	public static Expression? TryParse(Body body, ReadOnlySpan<char> line) =>
		line.Equals("if", StringComparison.Ordinal)
			? throw new MissingCondition(body)
			: line.Equals("else", StringComparison.Ordinal) || line.StartsWith("else if", StringComparison.Ordinal)
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
		return new If(condition, then, HasRemainingBody(body)
			? CreateElseIfOrElse(body, body.GetLine(body.ParsingLineNumber + 1).AsSpan(body.Tabs))
			: null, body);
	}

	private static Expression GetConditionExpression(Body body, ReadOnlySpan<char> line)
	{
		var condition = body.Method.ParseExpression(body, line);
		var booleanType = condition.ReturnType.GetType(Base.Boolean);
		if (condition.ReturnType == booleanType || booleanType.IsCompatible(condition.ReturnType))
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

	private static bool HasOnlyElse(ReadOnlySpan<char> line) => line.Equals("else", StringComparison.Ordinal);

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
		var questionMarkIndex = input.IndexOf('?');
		var firstBracket = input.IndexOf('(');
		if (questionMarkIndex > 2 &&
			NoFirstBracketOrSurroundedByIt(input, firstBracket, questionMarkIndex))
			return input.Count('?') > 1
				? throw new ConditionalExpressionsCannotBeNested(body)
				: true;
		return false;
	}

	private static bool NoFirstBracketOrSurroundedByIt(ReadOnlySpan<char> input, int firstBracket,
		int questionMarkIndex) =>
		firstBracket == -1 || firstBracket > questionMarkIndex || input.IndexOf(')') < questionMarkIndex ||
		firstBracket == 0 && input[^1] == ')';

	public sealed class ConditionalExpressionsCannotBeNested(Body body) : ParsingFailed(body);

	public static Expression ParseConditional(Body body, ReadOnlySpan<char> input)
	{
		if (input[0] == '(' && input[^1] == ')')
			input = input[1..^1];
		var questionMarkIndex = input.IndexOf('?');
		if (questionMarkIndex < 2)
			throw new InvalidCondition(body); //ncrunch: no coverage
		var elseIndex = input.IndexOf(" else ");
		if (elseIndex <= 5)
			throw new MissingElseExpression(body);
		return new If(GetConditionExpression(body, input[..(questionMarkIndex - 1)]),
			body.Method.ParseExpression(body, input[(questionMarkIndex + 2)..elseIndex]), body.Method.ParseExpression(body, input[(elseIndex + 6)..]));
	}

	public sealed class MissingElseExpression(Body body) : ParsingFailed(body);
}