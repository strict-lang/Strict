using System;

namespace Strict.Language.Expressions;

public sealed class If : BlockExpression
{
	public If(Expression condition, Expression then, Expression? optionalElse) : base(then.ReturnType)
	{
		Condition = condition;
		Then = then;
		OptionalElse = optionalElse;
	}

	public Expression Condition { get; }
	public Expression Then { get; }
	public Expression? OptionalElse { get; }

	public override int GetHashCode() =>
		Condition.GetHashCode() ^ Then.GetHashCode() ^ (OptionalElse?.GetHashCode() ?? 0);

	public override string ToString() =>
		"if " + Condition + Environment.NewLine + "\t" + Then + (OptionalElse != null
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
		var condition = line.Method.TryParseExpression(line, line.Text["if ".Length..]) ??
			throw new MissingCondition(line);
		methodLineNumber++;
		var then = GetThenExpression(line.Method, ref methodLineNumber);
		if (methodLineNumber + 2 >= line.Method.bodyLines.Count ||
			line.Method.bodyLines[methodLineNumber + 1].Text != "else")
			return new If(condition, then, null);
		methodLineNumber += 2;
		return new If(condition, then,
			line.Method.ParseMethodLine(line.Method.bodyLines[methodLineNumber], ref methodLineNumber));
	}

	private static Expression GetThenExpression(Method method, ref int methodLineNumber)
	{
		if (methodLineNumber >= method.bodyLines.Count)
			throw new MissingThen(method.bodyLines[methodLineNumber - 1]);
		if (method.bodyLines[methodLineNumber].Tabs !=
			method.bodyLines[methodLineNumber - 1].Tabs + 1)
			throw new Method.InvalidIndentation(method.Type, method.TypeLineNumber + methodLineNumber,
				string.Join('\n', method.bodyLines.ToWordList()), method.Name);
		return method.ParseMethodLine(method.bodyLines[methodLineNumber], ref methodLineNumber);
	}

	public sealed class MissingThen : ParsingFailed
	{
		public MissingThen(Method.Line line) : base(line) { }
	}
}