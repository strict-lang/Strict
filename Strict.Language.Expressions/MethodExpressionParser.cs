using System.Collections.Generic;
using System.Linq;

namespace Strict.Language.Expressions;

/// <summary>
/// Parses method bodies by splitting into main lines (lines starting without tabs)
/// and getting the expression recursively via parser combinator logic in each expression.
/// </summary>
public class MethodExpressionParser : ExpressionParser
{
	public override Expression ParseAssignmentExpression(Type type, string initializationLine,
		int fileLineNumber)
	{
		var line = new Method.Line(new Method(type, 0, this, new[] { Method.From }), 0,
			initializationLine, 0);
		var singleArgumentOrMethodCall = TryParseExpression(line, initializationLine) ??
			throw new UnknownExpression(line);
		Expression[] arguments;
		if (singleArgumentOrMethodCall is MethodCall methodCall)
			arguments = methodCall.Arguments.ToArray();
		else
			arguments = new[] { singleArgumentOrMethodCall };
		foreach (var method in type.Methods)
			if (method.Name == Method.From && method.Parameters.Count == arguments.Length)
				return new MethodCall(new Value(type, type), method, arguments);
		throw new UnknownExpression(line); //ncrunch: no coverage
	}

	public override Expression? TryParseExpression(Method.Line line, string partToParse) =>
		Number.TryParse(line, partToParse) ?? Boolean.TryParse(line, partToParse) ??
		Text.TryParse(line, partToParse) ?? Binary.TryParse(line, partToParse) ??
		MemberCall.TryParse(line, partToParse) ?? MethodCall.TryParse(line, partToParse);

	public class UnknownExpression : ParsingFailed
	{
		public UnknownExpression(Method.Line line, string error = "") : base(line, error) { }
	}

	/// <summary>
	/// Called lazily by Method.Body and only if needed for execution (context should be over there
	/// as parsing is done in parallel, we should not keep any state here).
	/// </summary>
	public override Expression ParseMethodBody(Method method)
	{
		var expressions = new List<Expression>();
		for (var lineNumber = 0; lineNumber < method.bodyLines.Count; lineNumber++)
			expressions.Add(ParseMethodLine(method.bodyLines[lineNumber], ref lineNumber));
		return new MethodBody(method, expressions);
	}

	public override Expression ParseMethodLine(Method.Line line, ref int methodLineNumber) =>
		Assignment.TryParse(line) ?? If.TryParse(line, ref methodLineNumber) ??
		Return.TryParse(line) ??
		TryParseExpression(line, line.Text) ?? throw new UnknownExpression(line);
}