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
	//TODO: change to MethodBody
	public override Expression Parse(Method context, string lines)
	{
		var mainLines = GetMainLines(lines);
		var expressions = new Expression[mainLines.Count];
		for (var number = 0; number < mainLines.Count; number++)
			expressions[number] = TryParse(context, mainLines[number]) ??
				throw new UnknownExpression(context, mainLines[number], number + 1);
		return new MethodBody(context, expressions);
	}

	public static IReadOnlyList<string> GetMainLines(string lines)
	{
		var mainLines = new List<string>();
		var currentLine = new StringBuilder(40);
		for (var index = 0; index < lines.Length; index++)
			if (lines[index] == '\n' && NextLineIsNotExtraIndented(index, lines))
			{
				mainLines.Add(currentLine.ToString());
				currentLine.Clear();
			}
			else if (lines[index] != '\r' && (index < lines.Length - 1 || lines[index] != '\n'))
				currentLine.Append(lines[index]);
		if (currentLine.Length > 0)
			mainLines.Add(currentLine.ToString());
		return mainLines;
	}

	private static bool NextLineIsNotExtraIndented(int index, string lines) =>
		index + 1 < lines.Length && lines[index + 1] != '\t';

	public static Expression? TryParse(Method context, string input)
	{
		if (string.IsNullOrEmpty(input))
			throw new IncompleteExpression(context);
		return Assignment.TryParse(context, input) ?? Number.TryParse(context, input) ??
			Boolean.TryParse(context, input) ?? Text.TryParse(context, input) ??
			Binary.TryParse(context, input) ??
			MethodCall.TryParse(context, input); //TODO: still need member call!
	}

	public class UnknownExpression : Exception
	{
		public UnknownExpression(Method context, string input, int lineNumber = 0) : base(input +
			"\n in " + context + (lineNumber > 0
				? ":" + lineNumber
				: "")) { }
	}

	protected class IncompleteExpression : Exception
	{
		public IncompleteExpression(Method context) : base(context.ToString()) { }
	}
}