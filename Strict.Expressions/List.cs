using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict.Expressions;

public sealed class List : Value
{
	public List(Body bodyForErrorMessage, List<Expression> values, bool isMutable = false) : base(
		values[0].ReturnType.
			GetListImplementationType(GetCommonBaseType(values, bodyForErrorMessage)), [],
		values[0].LineNumber, isMutable) =>
		Values = values;

	public List(Type type, int lineNumber = 0) : base(type, [], lineNumber, true) => Values = [];

	private static Type GetCommonBaseType(IReadOnlyList<Expression> values, Body bodyForErrorMessage)
	{
		var firstType = values[0].ReturnType;
		if (values.Count == 1)
			return firstType;
		var allSameType = true;
		for (var i = 1; i < values.Count; i++)
			if (values[i].ReturnType != firstType)
			{
				allSameType = false;
				break;
			}
		if (allSameType)
			return firstType;
		for (var i = 0; i < values.Count; i++)
		{
			var members = values[i].ReturnType.Members;
			for (var j = 0; j < members.Count; j++)
				if (members[j].Type == firstType)
					return firstType;
		}
		for (var i = 0; i < values.Count; i++)
		{
			var members = firstType.Members;
			for (var j = 0; j < members.Count; j++)
				if (members[j].Type == values[i].ReturnType)
					return values[i].ReturnType; //ncrunch: no coverage
		}
		throw new ListElementsMustHaveMatchingType(bodyForErrorMessage, values);
	}

	public sealed class ListElementsMustHaveMatchingType(Body body, IReadOnlyList<Expression> values)
		: ParsingFailed(body, "List has one or many mismatching types " + string.Join(", ", values));

	public List<Expression> Values { get; }
	public override bool IsConstant
	{
		get
		{
			for (var i = 0; i < Values.Count; i++)
				if (!Values[i].IsConstant)
					return false;
			return true;
		}
	}

	public override bool Equals(Expression? other) =>
		ReferenceEquals(this, other) || other is List list && ReturnType == list.ReturnType &&
		Values.Count == list.Values.Count && ValuesEqual(list);

	private bool ValuesEqual(List other)
	{
		for (var i = 0; i < Values.Count; i++)
			if (!Values[i].Equals(other.Values[i]))
				return false; //ncrunch: no coverage
		return true;
	}

	public override int GetHashCode() =>
		Values.Count > 0 //ncrunch: no coverage
			? ReturnType.GetHashCode() ^ Values[0].GetHashCode() ^ Values.Count
			: ReturnType.GetHashCode();

	public ValueInstance? TryGetConstantData()
	{
		if (cachedData.HasValue)
			return cachedData;
		if (!IsConstant)
			return null;
		var valueInstances = new ValueInstance[Values.Count];
		for (var i = 0; i < Values.Count; i++)
			if (Values[i] is List innerList)
			{
				// Recursively evaluate nested list constants; List.Data throws so we use TryGetConstantData
				var innerData = innerList.TryGetConstantData();
				if (innerData == null)
					return null; //ncrunch: no coverage, only when we have mutable items
				valueInstances[i] = innerData.Value;
			}
			else if (Values[i] is Value constantValue)
				valueInstances[i] = constantValue.Data;
			else
				return null;
		cachedData = new ValueInstance(ReturnType, valueInstances);
		return cachedData;
	}

	private ValueInstance? cachedData;
	public new ValueInstance Data => throw new DataAccessRequiresConstantList(ReturnType);

	public sealed class DataAccessRequiresConstantList(Type returnType)
		: ParsingFailed(returnType, 0, "Use TryGetConstantData instead!");

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
	public static Expression? TryParseWithSingleElement(Body body, ReadOnlySpan<char> input,
		bool makeMutable) =>
		input.Length < 2 || input[0] != '(' || input[^1] != ')'
			? null
			: input.Length == 2
				? throw new EmptyListNotAllowed(body)
				: new List(body, [body.Method.ParseExpression(body, input[1..^1])], makeMutable);

	public static Expression? TryParseWithMultipleOrNestedElements(Body body,
		ReadOnlySpan<char> input, bool makeMutable) =>
		input.Length > 2 && input[0] == '(' && input[^1] == ')'
			? new List(body, body.Method.ParseListArguments(body, input[1..^1]), makeMutable)
			: null;

	public sealed class EmptyListNotAllowed(Body body) : ParsingFailed(body, "()");
}