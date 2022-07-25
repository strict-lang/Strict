using System;
using System.Collections.Generic;
using System.Linq;

namespace Strict.Language.Expressions;

public sealed class List : Expression
{
	public List(Context context, List<Expression> values) : base(context.GetType(Base.List)) => Values = values;
	public List<Expression> Values { get; }
	public override string ToString() => "(" + ValuesToString() + ")";

	private string ValuesToString() =>
		Values.Aggregate("", (current, expression) => current + (expression + ", ")).Trim()[..^1];

	public static Expression? TryParse(Method.Line line, Range range)
	{
		// (1, 2, 3) + (3, 4)
		// ^l1       ^op ^l2
		// ((1, 2), (3, 4))
		// ^l1 -> this is problematic, add some tests, probably some grouping needed, ask Ben if you need some new grouping code, or use your own ..
		var input = line.Text.GetSpanFromRange(range);
		if (input.Length >= 2 && input[0] == '(' && input[^1] == ')')
		{
			var innerSpan = input[1..^1];
			if (innerSpan.IsEmpty)
				throw new EmptyListNotAllowed(line, input.ToString());
			var start = range.Start.Value + 1;
			if (!innerSpan.Contains('(') && !innerSpan.Contains('"'))
				return TryParseListFast(line, (start, innerSpan.Length),
					innerSpan.SplitIntoRanges(',', true));
			var expressions = new List<Expression>();
			new PhraseTokenizer(line.Text, new Range(start, start + innerSpan.Length)).ProcessEachToken(
				tokenRange =>
				{
					if (line.Text[tokenRange.Start.Value] != ',')
						expressions.Add(line.Method.TryParseExpression(line, tokenRange) ??
							throw new MethodExpressionParser.UnknownExpression(line, line.Text[tokenRange]));
				});
			return new List(line.Method, expressions);
		}
		else
			return null;
	}

	private static Expression TryParseListFast(Method.Line line, (int, int) offsetAndInnerSpanLength, RangeEnumerator elements)
	{
		var expressions = new List<Expression>();
		foreach (var element in elements)
			expressions.Add(line.Method.TryParseExpression(line, element.GetOuterRange(offsetAndInnerSpanLength)) ??
				throw new MethodExpressionParser.UnknownExpression(line, line.Text[element.GetOuterRange(offsetAndInnerSpanLength)]));
		return new List(line.Method, expressions);
	}

	//TODO: Probably not needed
	public static bool HasIncompatibleDimensions(Expression left, Expression right) =>
		left is List leftList && right is List rightList &&
		leftList.Values.Count != rightList.Values.Count;

	//TODO: as discussed in meeting, we use generics and always check if the right side is castable into the left side (via from), e.g. make a test where we add a Count to a list of Texts -> output list of texts (always from left side), we never change the left side type
	public static bool HasMismatchingTypes(Expression left, Expression right) =>
		left is List leftList && !leftList.IsFirstType<Text>() && right switch
		{
			List rightList when rightList.IsFirstType<Text>() => true,
			Binary { Left: List rightBinaryLeftList } when rightBinaryLeftList.IsFirstType<Text>() =>
				true,
			Binary { Left: Text } => true,
			_ => !leftList.IsFirstType<Text>() && right is Text
		};

	private bool IsFirstType<T>() => Values.First() is T;

	public sealed class ListsHaveDifferentDimensions : ParsingFailed
	{
		public ListsHaveDifferentDimensions(Method.Line line, string error) : base(line, error) { }
	}

	public class EmptyListNotAllowed : ParsingFailed
	{
		public EmptyListNotAllowed(Method.Line line, string error) : base(line, error) { }
	}
}