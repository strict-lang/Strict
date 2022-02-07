using System;
using System.Collections.Generic;
using System.Text;

namespace Strict.Language.Expressions;

/// <summary>
/// Parses method bodies by splitting into main lines (lines starting without tabs)
/// and getting the expression recursively via parser combinator logic in each expression.
/// </summary>
public class MethodExpressionParser : ExpressionParser
{
	/// <summary>
	/// Called lazily by Method.Body and only if needed for execution (context should be over there
	/// as parsing is done in parallel, we should not keep any state here).
	/// </summary>
	public override MethodBody Parse(Method method)
	{
		var expressions = new List<Expression>();
		for (var lineNumber = 0; lineNumber < method.bodyLines.Count; lineNumber++)
			expressions.Add(TryParse(method, ref lineNumber) ??
				throw new UnknownExpression(method, method.bodyLines[lineNumber].Text, lineNumber + 1));
		return new MethodBody(method, expressions);
	}

	public override MethodBody Parse(Type type, string initializationLine)
	{
		var constructor = type.Methods[0];
		var lineNumber = 0;
		return new MethodBody(constructor,
			new[] { TryParse(constructor, initializationLine, ref lineNumber)! });
	}

	public override Expression? TryParse(Method method, ref int lineNumber) =>
		TryParse(method, method.bodyLines[lineNumber].Text, ref lineNumber);

	public override Expression? TryParse(Method method, string line, ref int lineNumber)
	{
		if (string.IsNullOrEmpty(line))
			throw new EmptyExpression(method);
		return Assignment.TryParse(method, line) ?? If.TryParse(method, ref lineNumber) ??
			Number.TryParse(method, line) ?? Boolean.TryParse(method, line) ??
			Text.TryParse(method, line) ?? Binary.TryParse(method, line) ??
			MethodCall.TryParse(method, line) ?? MemberCall.TryParse(method, line);
	}

	public class UnknownExpression : Exception
	{
		public UnknownExpression(Method context, string input, int lineNumber = 0) : base(input +
			"\n in " + context + (lineNumber > 0
				? ":" + lineNumber
				: "")) { }
	}

	protected class EmptyExpression : Exception
	{
		public EmptyExpression(Method context) : base(context.ToString()) { }
	}
}

public class If : Expression
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
			throw new InvalidIndentation(string.Join('\n', method.bodyLines.ToWordListString()));
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

	public sealed class InvalidIndentation : Exception
	{
		public InvalidIndentation(string lines) : base(lines) { }
	}
}