using System;
using System.Collections.Generic;

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
			expressions.Add(TryParse(method, method.bodyLines[lineNumber].Text, ref lineNumber) ??
				throw new UnknownExpression(method, method.bodyLines[lineNumber].Text, lineNumber));
		return new MethodBody(method, expressions);
	}

	public override Expression ParseMethodCall(Type type, string initializationLine)
	{
		var constructor = type.Methods[0];
		var lineNumber = 0;
		return new MethodCall(new Value(type, type), constructor,
			TryParse(constructor, initializationLine, ref lineNumber)!);
	}

	public override Expression? TryParse(Method method, string line, ref int lineNumber) =>
		Assignment.TryParse(method, line) ?? If.TryParse(method, line, ref lineNumber) ??
		Return.TryParse(method, line) ?? Number.TryParse(method, line) ??
		Boolean.TryParse(method, line) ?? Text.TryParse(method, line) ??
		Binary.TryParse(method, line) ??
		MemberCall.TryParse(method, line) ?? MethodCall.TryParse(method, line);

	public class UnknownExpression : Exception
	{
		public UnknownExpression(Method context, string input, int lineNumber = -1) : base(input +
			"\n in " + context + (lineNumber >= 0
				? ":" + (lineNumber + 1)
				: "")) { }
	}
}