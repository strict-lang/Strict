﻿namespace Strict.Language.Expressions;

public sealed class List : Value
{
	public List(Body bodyForErrorMessage, List<Expression> values) : base(
		values[0].ReturnType.GetListImplementationType(GetCommonBaseType(values.Select(v => v.ReturnType).ToList(),
			bodyForErrorMessage)),
		values) =>
		Values = values;

	public List(Type type) : base(type, Array.Empty<Expression>()) =>
		Values = new List<Expression>();

	private static Type
		GetCommonBaseType(IReadOnlyList<Type> returnTypes, Body bodyForErrorMessage) =>
		returnTypes.Count == 1 || returnTypes.All(t => t == returnTypes[0]) ||
		returnTypes.Any(t => t.Members.Any(member => member.Type == returnTypes[0]))
			? returnTypes[0]
			: returnTypes.FirstOrDefault(t => returnTypes[0].Members.Any(m => m.Type == t)) ??
			throw new ListElementsMustHaveMatchingType(bodyForErrorMessage, returnTypes);

	public sealed class ListElementsMustHaveMatchingType(Body body, IEnumerable<Type> returnTypes)
		: ParsingFailed(body,
			"List has one or many mismatching types " + string.Join(", ", returnTypes));

	public List<Expression> Values { get; private set; }

	public override string ToString()
	{
		if (Values.Count == 0)
			return ReturnType.Name;
		var result = Values.ToBrackets();
		return result.Length > Limit.MultiLineCharacterCount
			? result.Replace(", ", ",\n\t")
			: result;
	}

	/// <summary>
	/// Since there was no space found we can check much quicker what is inside the list
	/// </summary>
	public static Expression? TryParseWithSingleElement(Body body, ReadOnlySpan<char> input) =>
		input.Length < 2 || input[0] != '(' || input[^1] != ')'
			? null
			: input.Length == 2
				? throw new EmptyListNotAllowed(body)
				: new List(body,
					new List<Expression>
					{
						body.Method.ParseExpression(body, input[1..^1])
					});

	public static Expression? TryParseWithMultipleOrNestedElements(Body body, ReadOnlySpan<char> input) =>
		input.Length > 2 && input[0] == '(' && input[^1] == ')'
			? new List(body,
				body.Method.ParseListArguments(body, input[1..^1]))
			: null;

	public sealed class EmptyListNotAllowed(Body body) : ParsingFailed(body, "()");

	public void UpdateValue(Body bodyForErrorMessage, Expression index, Expression newExpression)
	{
		if (Values.Count == 0)
			Values = new List<Expression> { newExpression };
		if (index is Number number && int.TryParse(number.Data.ToString(), out var indexNumber))
			if (Values.Count - 1 >= indexNumber)
				Values[indexNumber] = newExpression;
			else
				throw new IndexOutOfRangeInListExpressions(bodyForErrorMessage, indexNumber, Values.Count);
	}

	public class IndexOutOfRangeInListExpressions(Body body, int index, int listExpressionsCount)
		: ParsingFailed(body,
			$"Given index {index} is not within the List Expressions count {listExpressionsCount}");
}