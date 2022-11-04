using System;
using System.Linq;

namespace Strict.Language.Expressions;

/// <summary>
/// If expressions are used for branching, can also be used as an input for any other expression
/// like method arguments, other conditions, etc. like conditional operators.
/// </summary>
public sealed class If : Expression
{
	public If(Expression condition, Expression then, Expression? optionalElse = null,
		Body? bodyForErrorMessage = null) : base(GetMatchingType(then.ReturnType,
		optionalElse?.ReturnType, bodyForErrorMessage))
	{
		Condition = condition;
		Then = then;
		OptionalElse = optionalElse;
	}

	/// <summary>
	/// The return type of the whole if expression must be a type compatible to both what the then
	/// expression and the else expression return (if used). This is a common problem in static
	/// programming languages and we can fix it here by evaluating both types and find a common base
	/// type. If that is not possible there is a compilation error here.
	/// </summary>
	private static Type GetMatchingType(Type thenType, Type? elseType,
		Body? bodyForErrorMessage) =>
		elseType == null || thenType == elseType || elseType.Implements.Contains(thenType)
			? thenType
			: thenType.Implements.Contains(elseType)
				? elseType
				: thenType.Implements.Union(elseType.Implements).FirstOrDefault() ??
				throw new ReturnTypeOfThenAndElseMustHaveMatchingType(
					bodyForErrorMessage ?? new Body(thenType.Methods[0]), thenType,
					elseType);

	public class ReturnTypeOfThenAndElseMustHaveMatchingType : ParsingFailed
	{
		public ReturnTypeOfThenAndElseMustHaveMatchingType(Body body, Type thenReturnType,
			Type optionalElseReturnType) : base(body,
			"The Then type: " + thenReturnType + " is not same as the Else type: " +
			optionalElseReturnType) { }
	}

	public Expression Condition { get; }
	public Expression Then { get; }
	/// <summary>
	/// Rarely used as most of the time Then will return and anything after is automatically the else
	/// (else is not allowed then). For multiple if/else or when not returning else might be useful.
	/// </summary>
	public Expression? OptionalElse { get; }

	public override int GetHashCode() =>
		Condition.GetHashCode() ^ Then.GetHashCode() ^ (OptionalElse?.GetHashCode() ?? 0);

	public override string ToString() =>
		OptionalElse != null && Then.ReturnType == OptionalElse.ReturnType &&
		Then is not Body && OptionalElse is not Body
			? Condition + " ? " + Then + " else " + OptionalElse
			: "if " + Condition + Environment.NewLine + "\t" + Then + (OptionalElse != null
				? Environment.NewLine + "else" + Environment.NewLine + "\t" + OptionalElse
				: "");

	public override bool Equals(Expression? other) =>
		other is If a && Equals(Condition, a.Condition) && Then.Equals(a.Then) &&
		(OptionalElse?.Equals(a.OptionalElse) ?? a.OptionalElse == null);

	public static Expression? TryParse(Body body, ReadOnlySpan<char> line) =>
		line.Equals("if", StringComparison.Ordinal)
			? throw new MissingCondition(body)
			: line.Equals("else", StringComparison.Ordinal)
				? throw new UnexpectedElse(body)
				: line.StartsWith("if ", StringComparison.Ordinal)
					? TryParseIf(body, line)
					: null;

	public sealed class MissingCondition : ParsingFailed
	{
		public MissingCondition(Body body) : base(body) { }
	}

	public sealed class UnexpectedElse : ParsingFailed
	{
		public UnexpectedElse(Body body) : base(body) { }
	}

	private static Expression TryParseIf(Body body, ReadOnlySpan<char> line)
	{
		var condition = GetConditionExpression(body, line[3..]);
		var thenBody = body.FindCurrentChild();
		if (thenBody == null)
			throw new MissingThen(body);
		var then = thenBody.Parse();
		return HasElse(body)
			? CreateIfWithElse(body, condition, then)
			: new If(condition, then, null, body);
	}

	private static bool HasElse(Body body) =>
		body.ParsingLineNumber + 2 < body.LineRange.End.Value && body.
			GetLine(body.ParsingLineNumber + 1).AsSpan(body.Tabs).
			Equals("else", StringComparison.Ordinal);

	private static Expression CreateIfWithElse(Body body, Expression condition, Expression then)
	{
		body.ParsingLineNumber++;
		var elseBody = body.FindCurrentChild();
		return elseBody == null
			? throw new MissingElseExpression(body)
			: new If(condition, then, elseBody.Parse(), body);
	}

	private static Expression GetConditionExpression(Body body, ReadOnlySpan<char> line)
	{
		var condition = body.Method.ParseExpression(body, line);
		if (condition.ReturnType.Name != Base.Boolean)
			throw new InvalidCondition(body, condition.ReturnType);
		return condition;
	}

	public sealed class InvalidCondition : ParsingFailed
	{
		public InvalidCondition(Body body, Type? conditionReturnType = null) : base(body,
			conditionReturnType != null
				? body.Method.FullName + "\n Return type " + conditionReturnType + " is not " + Base.Boolean
				: null) { }
	}

	public sealed class MissingThen : ParsingFailed
	{
		public MissingThen(Body body) : base(body) { }
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

	public sealed class ConditionalExpressionsCannotBeNested : ParsingFailed
	{
		public ConditionalExpressionsCannotBeNested(Body body) : base(body) { }
	}

	public static Expression ParseConditional(Body body, ReadOnlySpan<char> input)
	{
#if LOG_DETAILS
		Logger.Info(nameof(ParseConditional) + " " + input.ToString());
#endif
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

	public sealed class MissingElseExpression : ParsingFailed
	{
		public MissingElseExpression(Body body) : base(body) { }
	}
}