using System;
using System.Collections.Generic;

namespace Strict.Language.Expressions;

/// <summary>
/// Parses method bodies by splitting into main lines (lines starting without tabs)
/// and getting the expression recursively via parser combinator logic in each expression.
/// </summary>
public class MethodExpressionParser : ExpressionParser
{
	public override Expression ParseAssignmentExpression(Type type, string initializationLine, int fileLineNumber)
	{
		var constructor = type.Methods[0];
		var line = new Method.Line(constructor, 0, initializationLine, fileLineNumber);
		return new MethodCall(new Value(type, type), constructor,
			TryParseExpression(line, ..) ?? throw new UnknownExpression(line));
	}

	public override Expression? TryParseExpression(Method.Line line, Range partToParse)
	{
		if (!line.Text.GetSpanFromRange(partToParse).Contains(' '))
			return Boolean.TryParse(line, partToParse) ?? Text.TryParse(line, partToParse) ??
				List.TryParse(line, partToParse) ?? Constructor.TryParse(line, partToParse) ??
				MemberCall.TryParse(line, partToParse) ?? MethodCall.TryParse(line, partToParse) ??
				Number.TryParse(line, partToParse);
		var postfixTokens = new ShuntingYard(line.Text, partToParse).Output;
		if (postfixTokens.Count == 0)
			throw new NotSupportedException(
				"Something really bad went wrong, delete this when everything works!");
		else if (postfixTokens.Count == 1)
		{
			var range = postfixTokens.Pop();
			return Text.TryParse(line, range) ?? List.TryParse(line, range);
		}
		else if (postfixTokens.Count == 2)
		{
			// TODO: Unary goes here
			throw new NotSupportedException("Feed Murali more to get Unary done");
		}
		return Binary.TryParse(line, postfixTokens);
	}

	public sealed class UnknownExpression : ParsingFailed
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
		{
			var expression = ParseMethodLine(method.bodyLines[lineNumber], ref lineNumber);
			if (expression is Assignment assignment)
				method.Variables.Add(assignment);
			expressions.Add(expression);
		}
		//TODO: to clear link to original memory: method.bodyLines = Memory<char>.Empty; //ArraySegment<Method.Line>.Empty;
		return new MethodBody(method, expressions);
	}

	//private readonly PhraseTokenizer tokenizer = new();

	public override Expression ParseMethodLine(Method.Line line, ref int methodLineNumber) =>
		Assignment.TryParse(line) ?? If.TryParse(line, ref methodLineNumber) ??
		//TODO: for loop
		Return.TryParse(line) ??
		//TODO: error
		TryParseExpression(line, ..) ?? throw new UnknownExpression(line);
}