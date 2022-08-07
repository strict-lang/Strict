using System;
using System.Linq;

namespace Strict.Language.Expressions;

/// <summary>
/// If expressions are used for branching, can also be used as an input for any other expression
/// like method arguments, other conditions, etc. like conditional operators.
/// </summary>
public sealed class If : BlockExpression
{
	public If(Expression condition, Expression then, Expression? optionalElse = null,
		Method.Line? lineForErrorMessage = null) : base(GetMatchingType(then.ReturnType,
		optionalElse?.ReturnType, lineForErrorMessage))
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
		Method.Line? lineForErrorMessage) =>
		elseType == null || thenType == elseType || elseType.Implements.Contains(thenType)
			? thenType
			: thenType.Implements.Contains(elseType)
				? elseType
				: thenType.Implements.Union(elseType.Implements).FirstOrDefault() ??
				throw new ReturnTypeOfThenAndElseMustHaveMatchingType(
					lineForErrorMessage ?? new Method.Line(thenType.Methods[0], 0, "", 0), thenType,
					elseType);

	// ReSharper disable once HollowTypeName
	public class ReturnTypeOfThenAndElseMustHaveMatchingType : ParsingFailed
	{
		public ReturnTypeOfThenAndElseMustHaveMatchingType(Method.Line line, Type thenReturnType, Type optionalElseReturnType) : base(line, "The Then type: " + thenReturnType + " is not same as the Else type: " + optionalElseReturnType) { }
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
		Then is not BlockExpression && OptionalElse is not BlockExpression
			? Condition + " ? " + Then + " else " + OptionalElse
			: "if " + Condition + Environment.NewLine + "\t" + Then + (OptionalElse != null
				? Environment.NewLine + "else" + Environment.NewLine + "\t" + OptionalElse
				: "");

	public override bool Equals(Expression? other) =>
		other is If a && Equals(Condition, a.Condition) && Then.Equals(a.Then) &&
		(OptionalElse?.Equals(a.OptionalElse) ?? a.OptionalElse == null);

	public static Expression? TryParse(Method.Line line, ref int methodLineNumber) =>
		line.Text == "if"
			? throw new MissingCondition(line)
			: line.Text == "else"
				? throw new UnexpectedElse(line)
				: line.Text.StartsWith("if ", StringComparison.Ordinal)
					? TryParseIf(line, ref methodLineNumber)
					: null;

	public sealed class MissingCondition : ParsingFailed
	{
		public MissingCondition(Method.Line line) : base(line) { }
	}

	public sealed class UnexpectedElse : ParsingFailed
	{
		public UnexpectedElse(Method.Line line) : base(line) { }
	}

	private static Expression TryParseIf(Method.Line line, ref int methodLineNumber)
	{
		var condition = GetConditionExpression(line, 3..);
		methodLineNumber++;
		var then = GetThenExpression(line.Method, ref methodLineNumber);
		if (methodLineNumber + 2 >= line.Method.bodyLines.Count ||
			line.Method.bodyLines[methodLineNumber + 1].Text != "else")
			return new If(condition, then, null, line);
		methodLineNumber += 2;
		return new If(condition, then,
			line.Method.ParseMethodLine(line.Method.bodyLines[methodLineNumber], ref methodLineNumber),
			line);
	}

	private static Expression GetConditionExpression(Method.Line line, Range conditionRange)
	{
		var condition = line.Method.ParseExpression(line, conditionRange);
		if (condition.ReturnType.Name != Base.Boolean)
			throw new InvalidCondition(line, condition.ReturnType);
		return condition;
	}

	public class InvalidCondition : ParsingFailed
	{
		public InvalidCondition(Method.Line line, Type? conditionReturnType = null) : base(line,
			conditionReturnType != null
				? line.Text + "\n Return type " + conditionReturnType + " is not " + Base.Boolean
				: null) { }
	}

	private static Expression GetThenExpression(Method method, ref int methodLineNumber)
	{
		if (methodLineNumber >= method.bodyLines.Count)
			throw new MissingThen(method.bodyLines[methodLineNumber - 1]);
		if (method.bodyLines[methodLineNumber].Tabs !=
			method.bodyLines[methodLineNumber - 1].Tabs + 1)
			throw new Method.InvalidIndentation(method.Type, method.TypeLineNumber + methodLineNumber,
				string.Join('\n', method.bodyLines.ToWordList()), method.Name);
		//https://deltaengine.fogbugz.com/f/cases/25323
		return method.ParseMethodLine(method.bodyLines[methodLineNumber], ref methodLineNumber);
	}

	public sealed class MissingThen : ParsingFailed
	{
		public MissingThen(Method.Line line) : base(line) { }
	}

	public static bool CanTryParseConditional(Method.Line line, Range range)
	{
		var input = line.Text.GetSpanFromRange(range);
		var questionMarkIndex = input.IndexOf('?');
		var firstBracket = input.IndexOf('(');
		if (questionMarkIndex > 2 && (firstBracket == -1 || firstBracket > questionMarkIndex))
			return input.Count('?') > 1
				? throw new ConditionalExpressionsCannotBeNested(line)
				: true;
		return false;
	}

	public static If ParseConditional(Method.Line line, Range range)
	{
		var input = line.Text.GetSpanFromRange(range);
#if LOG_DETAILS
		Logger.Info(nameof(ParseConditional) + " " + input.ToString());
#endif
		var questionMarkIndex = input.IndexOf('?');
		if (questionMarkIndex < 2)
			throw new InvalidCondition(line);
		var elseIndex = input.IndexOf(" else ");
		if (elseIndex <= 5)
			throw new MissingElseExpression(line);
		var start = range.Start.Value;
		var conditionRange = start..(start + questionMarkIndex - 1);
		var thenRange = (start + questionMarkIndex + 2)..(start + elseIndex);
		var elseRange = (start + elseIndex + 6)..(start + input.Length);
		return new If(GetConditionExpression(line, conditionRange),
			line.Method.ParseExpression(line, thenRange), line.Method.ParseExpression(line, elseRange));
	}

	public sealed class MissingElseExpression : ParsingFailed
	{
		public MissingElseExpression(Method.Line line) : base(line) { }
	}

	public sealed class ConditionalExpressionsCannotBeNested : ParsingFailed
	{
		public ConditionalExpressionsCannotBeNested(Method.Line line) : base(line) { }
	}
}