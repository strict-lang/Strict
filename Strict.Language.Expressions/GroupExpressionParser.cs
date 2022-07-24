/*TODO: remove, not longer used
using System;
using System.Linq;

namespace Strict.Language.Expressions;

public sealed class GroupExpressionParser
{
	public static Expression TryParse(Method.Line line, int groupIndex)
	{
		var currentGroup = line.Groups[groupIndex];
		var left = line.Method.TryParseExpression(line, line.Text.Substring(currentGroup.Start, currentGroup.Length)) ??
			throw new MethodExpressionParser.UnknownExpression(line);
		if (HasAnyTextBeforeFirstGroup(line, groupIndex, currentGroup))
			left = ParseBeginningNonGroupExpression(line, currentGroup, left);
		return IsLastGroup(line, groupIndex)
			? HasNoTextAfterLastGroup(line, currentGroup)
				? left
				: ParseEndNonGroupExpressions(line, left, currentGroup.Start + currentGroup.Length)
			: ParseMiddleNonGroupExpressions(line, left, groupIndex);
	}

	private static bool HasAnyTextBeforeFirstGroup(Method.Line line, int groupIndex,
		Group currentGroup) =>
		groupIndex == 0 && line.Text[..currentGroup.Start].Any(c => c != '(');

	private static Expression ParseBeginningNonGroupExpression(Method.Line line, Group currentGroup, Expression right)
	{
		var partToParse = line.Text[..(currentGroup.Start - 1)].Split(" ");
		var left = line.Method.TryParseExpression(line, partToParse[0]) ?? throw new MethodExpressionParser.UnknownExpression(line);
		return new Binary(left, FindOperatorMethod(left, partToParse[1]), right);
	}

	private static bool IsLastGroup(Method.Line line, int groupIndex) =>
		groupIndex == line.Groups.Count - 1;

	private static bool HasNoTextAfterLastGroup(Method.Line line, Group currentGroup) =>
		line.Text.Length <= currentGroup.Start + currentGroup.Length + 1;

	private static Expression ParseEndNonGroupExpressions(Method.Line line, Expression left, int startIndex)
	{
		var operatorText =
			ReplaceParenthesisAndTrim(line.Text.Substring(startIndex, 3));
		var operatorMethod = left.ReturnType.Methods.FirstOrDefault(m => m.Name == operatorText) ??
			throw new Binary.NoMatchingOperatorFound(left.ReturnType, operatorText);
		var remainingExpression = line.Method.TryParseExpression(line,
			line.Text.AsSpan(startIndex + 3)) ?? throw new MethodExpressionParser.UnknownExpression(line);
		return new Binary(left, operatorMethod, remainingExpression);
	}

	private static string ReplaceParenthesisAndTrim(string text) =>
		text.Replace("(", "").Replace(")", "").Trim();

	private static Expression ParseMiddleNonGroupExpressions(Method.Line line, Expression left, int groupIndex)
	{
		var nextGroupIndex = FindNextGroupIndex(line, groupIndex, line.Groups[groupIndex]);
		if (nextGroupIndex <= 0)
			return left;
		var operatorText = GetOperatorText(line, line.Groups[groupIndex], nextGroupIndex);
		var partsToParse = operatorText.Split(" ");
		if (partsToParse.Length <= 1)
			return new Binary(left, FindOperatorMethod(left, operatorText),
				TryParse(line, nextGroupIndex));
		left = new Binary(left, FindOperatorMethod(left, partsToParse[0]), line.Method.TryParseExpression(line, partsToParse[1]) ?? throw new MethodExpressionParser.UnknownExpression(line));
		return new Binary(left, FindOperatorMethod(left, partsToParse[2]),
			TryParse(line, nextGroupIndex));
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

	private static string
		GetOperatorText(Method.Line line, Group currentGroup, int nextGroupIndex) =>
		ReplaceParenthesisAndTrim(line.Text[
			(currentGroup.Start + currentGroup.Length)..(line.Groups[nextGroupIndex].Start - 1)]);

	private static Method FindOperatorMethod(Expression left, string operatorText) =>
		left.ReturnType.Methods.FirstOrDefault(m => m.Name == operatorText) ??
		throw new Binary.NoMatchingOperatorFound(left.ReturnType, operatorText);
}
*/