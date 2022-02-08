using System;

namespace Strict.Language.Expressions;

public sealed class If : Expression
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
		Condition.GetHashCode() ^ Then.GetHashCode() ^ OptionalElse?.GetHashCode() ?? 0;

	public override string ToString() =>
		"if " + Condition + Environment.NewLine + "\t" + Then + (OptionalElse != null
			? Environment.NewLine + "else" + Environment.NewLine + "\t" + OptionalElse
			: "");

	public override bool Equals(Expression? other) =>
		other is If a && Equals(Condition, a.Condition) && Then.Equals(a.Then) && (OptionalElse?.Equals(a.OptionalElse) ?? true);

	public static Expression? TryParse(Method method, ref int lineNumber) =>
		method.bodyLines[lineNumber].Text.StartsWith("if ", StringComparison.Ordinal)
			? TryParseIf(method, ref lineNumber)
			: null;

	private static Expression TryParseIf(Method method, ref int lineNumber)
	{
		var condition = method.TryParse(method.bodyLines[lineNumber].Text["if ".Length..]) ??
			throw new MissingCondition(method.bodyLines[lineNumber].Text);
		lineNumber++;
		if (lineNumber >= method.bodyLines.Count)
			throw new MissingThen(method.bodyLines[lineNumber - 1].Text);
		if (method.bodyLines[lineNumber].Tabs != method.bodyLines[lineNumber - 1].Tabs + 1)
			throw new Method.InvalidIndentation(string.Join('\n', method.bodyLines.ToWordListString()),
				lineNumber, method.Name);
		var then = method.TryParse(method.bodyLines[lineNumber].Text, ref lineNumber) ??
			throw new MissingThen(method.bodyLines[lineNumber].Text);
		Expression? optionalElse = null;
		lineNumber++;
		if (lineNumber < method.bodyLines.Count)
			optionalElse = method.TryParse(method.bodyLines[lineNumber].Text, ref lineNumber);
		return new If(condition, then, optionalElse);
	}

	public sealed class MissingCondition : Exception
	{
		public MissingCondition(string input) : base(input) { }
	}

	public sealed class MissingThen : Exception
	{
		public MissingThen(string otherLines) : base(otherLines) { }
	}
}