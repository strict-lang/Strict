using System;
using System.Collections.Generic;
using System.Linq;

namespace Strict.Language.Expressions;

public sealed class List : Value
{
	public List(Context context, List<Expression> values) : base(context.GetType(Base.List),
		//TODO: we should find the common base type for the whole list, see BinarySaveExtension and also how If handles this
		values[0].ReturnType) =>
		Values = values;

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
				? throw new EmptyListNotAllowed(line)
				: new List(line.Method,
					new List<Expression>
					{
						line.Method.ParseExpression(line, range.RemoveFirstAndLast(line.Text.Length))
					});
	}

	public static Expression? TryParseWithMultipleOrNestedElements(Method.Line line, Range range)
	{
		var input = line.Text.GetSpanFromRange(range);
		return input.Length > 2 && input[0] == '(' && input[^1] == ')'
			? new List(line.Method,
				line.Method.ParseListArguments(line, range.RemoveFirstAndLast(line.Text.Length)))
			: null;
	}

	public class EmptyListNotAllowed : ParsingFailed
	{
		public EmptyListNotAllowed(Method.Line line) : base(line, "()") { }
	}

	internal bool IsFirstType<T>() => Values.First() is T;
}