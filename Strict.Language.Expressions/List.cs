using System;
using System.Collections.Generic;
using System.Linq;

namespace Strict.Language.Expressions;

public sealed class List : Value
{
	public List(Context context, List<Expression> values) : base(context.GetType(Base.List),
		//TODO: we should find the common base type for the whole list, see BinarySaveExtension and also how If handles this
		values[0].ReturnType) => Values = values;
	public List<Expression> Values { get; }
	public override string ToString() => Values.ToBrackets();

	/// <summary>
	/// Since there was no space found we can check much quicker what is inside the list
	/// </summary>
	public static Expression? TryParseWithSingleElement(Method.Line line, Range range)
	{
		var input = line.Text.GetSpanFromRange(range);
		return input.Length < 2 || input[0] != '(' || input[^1] != ')'
			? null
			: input.Length == 2
				? throw new EmptyListNotAllowed(line, input.ToString())
				: new List(line.Method,
					new List<Expression>
					{
						line.Method.TryParseExpression(line, range.RemoveFirstAndLast(line.Text.Length)) ??
						throw new MethodExpressionParser.UnknownExpression(line, input.ToString())
					});
	}

	//TODO: need more complex tests, seems to do the basics correct, but check all nested cases as well
	// (1, 2, 3) + (3, 4)
	// ^l1       ^op ^l2
	// ((1, 2), (3, 4))
	// ^l1 -> this is problematic, add some tests, probably some grouping needed, ask Ben if you need some new grouping code, or use your own ..
	public static Expression? TryParseWithMultipleOrNestedElements(Method.Line line, Range range)
	{
		var input = line.Text.GetSpanFromRange(range);
		if (input.Length <= 2 || input[0] != '(' || input[^1] != ')')
			return null;
		Console.WriteLine("TryParseWithMultipleOrNestedElements: "+input[1..^1].ToString());
		return new List(line.Method, line.Method.ParseListArguments(line, range.Start.Value + 1,
			range.Start.Value + input.Length - 1));
	}

	public class EmptyListNotAllowed : ParsingFailed
	{
		public EmptyListNotAllowed(Method.Line line, string error) : base(line, error) { }
	}

	internal bool IsFirstType<T>() => Values.First() is T;
}