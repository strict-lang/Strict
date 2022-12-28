using System;
using System.Collections.Generic;
using System.Linq;

namespace Strict.Language.Expressions;

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

	public sealed class ListElementsMustHaveMatchingType : ParsingFailed
	{
		public ListElementsMustHaveMatchingType(Body body, IEnumerable<Type> returnTypes) :
			base(body, "List has one or many mismatching types " + string.Join(", ", returnTypes)) { }
	}

	public List<Expression> Values { get; }

	public override string ToString()
	{
		if (Values.Count == 0)
			return ReturnType.Name;
		var result = Values.ToBrackets();
		return result.Length > Limit.ListCharacterCount
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

	public class EmptyListNotAllowed : ParsingFailed
	{
		public EmptyListNotAllowed(Body body) : base(body, "()") { }
	}
}