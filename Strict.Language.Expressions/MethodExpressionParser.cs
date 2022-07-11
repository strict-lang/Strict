using System.Collections.Generic;
using System.Linq;

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

	public override Expression ParseMethodLine(Method.Line line, ref int methodLineNumber)
	{
		try
		{
			return Assignment.TryParse(line) ?? If.TryParse(line, ref methodLineNumber) ??
				//TODO: for loop
				Return.TryParse(line) ??
				//TODO: error
				TryParseExpression(line, line.Text) ?? throw new UnknownExpression(line);
		}
		catch (UnknownExpression e)
		{
			return line.Groups.Count > 0
				? ParseGroupExpressions(line, 0)
				: throw e;
		}
	}

	private Expression ParseGroupExpressions(Method.Line line, int groupIndex)
	{
		var currentGroup = line.Groups[groupIndex];
		var left =
			TryParseExpression(line, line.Text.Substring(currentGroup.Start, currentGroup.Length)) ??
			throw new UnknownExpression(line);
		if (groupIndex == line.Groups.Count - 1)
			return line.Text.Length <= currentGroup.Start + currentGroup.Length + 1
				? left
				: ParseRemainingText(line, left, currentGroup.Start + currentGroup.Length);
		var nextGroupIndex = FindNextGroupIndex(line, groupIndex, currentGroup);
		return nextGroupIndex > 0
			? new Binary(left, FindOperatorMethod(left, GetOperatorText(line, currentGroup, nextGroupIndex)), ParseGroupExpressions(line, nextGroupIndex))
			: left;
	}

	private Expression ParseRemainingText(Method.Line line, Expression left, int startIndex)
	{
		var operatorMethod =
			left.ReturnType.Methods.FirstOrDefault(m =>
				m.Name == line.Text.Substring(startIndex, 3).Replace("(", "").
					Replace(")", "").Trim()) ?? throw new Binary.NoMatchingOperatorFound(left.ReturnType,
				line.Text.Substring(startIndex, 3));
		var remainingExpression =
			TryParseExpression(line,
				line.Text[(startIndex + 3)..].Replace("(", "").
					Replace(")", "").Trim()) ?? throw new UnknownExpression(line);
		return new Binary(left, operatorMethod, remainingExpression);
	}

	private static int FindNextGroupIndex(Method.Line line, int groupIndex, Group currentGroup)
	{
		var index = 1;
		while (groupIndex + index < line.Groups.Count)
		{
			if (line.Groups[groupIndex + index].Start > currentGroup.Start)
				return groupIndex + index;
			index++;
		}
		return -1;
	}

	private static Method FindOperatorMethod(Expression left, string operatorText) =>
		left.ReturnType.Methods.FirstOrDefault(m => m.Name == operatorText) ??
		throw new Binary.NoMatchingOperatorFound(left.ReturnType, operatorText);

	private static string GetOperatorText(Method.Line line, Group currentGroup, int nextGroupIndex) =>
		line.
			Text[(currentGroup.Start + currentGroup.Length)..(line.Groups[nextGroupIndex].Start - 1)].
			Replace("(", "").Replace(")", "").Trim();
}