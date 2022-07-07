﻿using System.Collections.Generic;

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
			TryParseExpression(line, initializationLine) ?? throw new UnknownExpression(line));
	}

	public override Expression? TryParseExpression(Method.Line line, string partToParse) =>
		Number.TryParse(line, partToParse) ?? Boolean.TryParse(line, partToParse) ??
		Text.TryParse(line, partToParse) ?? Binary.TryParse(line, partToParse) ?? List.TryParse(line, partToParse) ?? Constructor.TryParse(line, partToParse) ??
		MemberCall.TryParse(line, partToParse) ?? MethodCall.TryParse(line, partToParse);

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
		return new MethodBody(method, expressions);
	}

	public override Expression ParseMethodLine(Method.Line line, ref int methodLineNumber) =>
		Assignment.TryParse(line) ?? If.TryParse(line, ref methodLineNumber) ??
		//TODO: for loop
		Return.TryParse(line) ??
		//TODO: error
		TryParseExpression(line, line.Text) ?? throw new UnknownExpression(line);
}

//public class Group
//{
//	public static Expression? TryParse(Method.Line line, string input)
//	{
//		if (input.Contains('(') && input.Contains(')') &&
//			input[(input.IndexOf('(') + 1)..input.IndexOf(')')].HasOperator(out _))
//		{
//			var left = line.Method.TryParseExpression(line, input[..1]);
//			var right = line.Method.TryParseExpression(line, input[(input.IndexOf('(') + 1)..input.IndexOf(')')]);
//			return new Binary(left!, left?.ReturnType.Methods.FirstOrDefault(o => o.Name == input[1..3].Trim()),
//				right!);
//		}
//		return null;
//	}
//}