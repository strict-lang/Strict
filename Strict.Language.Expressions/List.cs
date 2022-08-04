using System;
using System.Collections.Generic;
using System.Linq;

namespace Strict.Language.Expressions;

public sealed class List : Value
{
	public List(Method.Line? lineForErrorMessage, List<Expression> values) : base(
		GetCommonBaseType(values.Select(v => v.ReturnType).ToList(),
			lineForErrorMessage), //Any other better approach other than ToList?
		values[0].ReturnType) =>
		Values = values;

	private static Type
		GetCommonBaseType(IReadOnlyList<Type> returnTypes, Method.Line? lineForErrorMessage) =>
		returnTypes.Count == 1 || returnTypes.All(t => t == returnTypes[0]) ||
		returnTypes.Any(t => t.Implements.Contains(returnTypes[0]))
			? returnTypes[0]
			: returnTypes.FirstOrDefault(t => returnTypes[0].Implements.Contains(t)) ??
			throw new ListElementsMustHaveMatchingType(
				lineForErrorMessage ?? new Method.Line(returnTypes[0].Methods[0], 0, "", 0), returnTypes);

	public sealed class ListElementsMustHaveMatchingType : ParsingFailed
	{
		public ListElementsMustHaveMatchingType(Method.Line line, IEnumerable<Type> returnTypes) : base(line, "List has one or many mismatching types " + string.Join(", ", returnTypes)) { }
	}

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
				: new List(line,
					new List<Expression>
					{
						line.Method.ParseExpression(line, range.RemoveFirstAndLast(line.Text.Length))
					});
	}

	public static Expression? TryParseWithMultipleOrNestedElements(Method.Line line, Range range)
	{
		var input = line.Text.GetSpanFromRange(range);
		return input.Length > 2 && input[0] == '(' && input[^1] == ')'
			? new List(line,
				line.Method.ParseListArguments(line, range.RemoveFirstAndLast(line.Text.Length)))
			: null;
	}

	public class EmptyListNotAllowed : ParsingFailed
	{
		public EmptyListNotAllowed(Method.Line line) : base(line, "()") { }
	}

	internal bool IsFirstType<T>() => Values.First() is T;
}