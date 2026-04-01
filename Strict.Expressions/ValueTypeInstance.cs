using Type = Strict.Language.Type;

namespace Strict.Expressions;

public sealed class ValueTypeInstance(Type returnType, ValueInstance[] values)
	: IEquatable<ValueTypeInstance>
{
 private static readonly System.Runtime.CompilerServices.ConditionalWeakTable<Type,
		Dictionary<string, int>> MemberIndexes = new();

	public readonly Type ReturnType = returnType;
	public readonly ValueInstance[] Values = values;

	/// <summary>
	/// Finds the value for the named member using a positional scan of ReturnType.Members.
	/// Skips constant members so callers fall through to their InitialValue evaluation.
	/// </summary>
	public bool TryGetValue(string name, out ValueInstance value)
	{
   var memberIndexes = MemberIndexes.GetValue(ReturnType, CreateMemberIndexes);
		if (memberIndexes.TryGetValue(name, out var memberIndex) && memberIndex < Values.Length)
		{
			value = Values[memberIndex];
			return true;
		}
		value = default;
		return false;
	}

	private static Dictionary<string, int> CreateMemberIndexes(Type type)
	{
		var members = type.Members;
		var memberIndexes = new Dictionary<string, int>(members.Count,
			StringComparer.OrdinalIgnoreCase);
		for (var memberIndex = 0; memberIndex < members.Count; memberIndex++)
			if (!members[memberIndex].IsConstant)
				memberIndexes.TryAdd(members[memberIndex].Name, memberIndex);
		return memberIndexes;
	}

	public ValueInstance this[string name] =>
		TryGetValue(name, out var value)
			? value
			: throw new KeyNotFoundException("Member not found: " + name);

	public override string ToString()
	{
		var typeMembers = ReturnType.Members;
		if (Values.Length == 0)
			return ReturnType.ToString();
		var result = new string[Values.Length];
		for (var i = 0; i < Values.Length; i++)
			result[i] = typeMembers[i].Name + "=" + (Values[i].HasValue
				? Values[i].ToString()
				: "(unset)");
		return ReturnType + ": " + string.Join("; ", result);
	}

	public string ToAutomaticText()
	{
		var members = ReturnType.Members;
		if (Values.Length == 0)
			return ReturnType.Name;
		var visibleMembers = new List<(Strict.Language.Member Member, ValueInstance Value)>(Values.Length);
		var totalValueCount = 0;
		var containsCollection = false;
		for (var index = 0; index < Values.Length && index < members.Count; index++)
			if (ShouldIncludeMember(members[index], Values[index]))
			{
				visibleMembers.Add((members[index], Values[index]));
				totalValueCount += CountDisplayValues(Values[index]);
				containsCollection |= Values[index].IsList || Values[index].IsDictionary;
			}
		if (visibleMembers.Count == 0)
			return ReturnType.Name;
		return containsCollection || totalValueCount > 4
			? ReturnType.Name + "(" + string.Join(", ", visibleMembers.Select(member =>
				member.Member.Name + "=" + FormatValue(member.Value, true))) + ")"
			: "(" + string.Join(", ", visibleMembers.Select(member =>
				FormatValue(member.Value, false))) + ")";
	}

	private static string FormatValue(ValueInstance value, bool limitCollectionEntries) =>
		value.IsList
			? FormatList(value.List.Items, limitCollectionEntries)
			: value.ToExpressionCodeString();

	private static string FormatList(IReadOnlyList<ValueInstance> items, bool limitEntries)
	{
		if (items.Count == 0)
			return string.Empty;
		var displayedItemCount = limitEntries && items.Count > 3
			? 3
			: items.Count;
		var parts = new string[displayedItemCount + (displayedItemCount < items.Count
			? 1
			: 0)];
		for (var index = 0; index < displayedItemCount; index++)
			parts[index] = items[index].ToExpressionCodeString();
		if (displayedItemCount < items.Count)
			parts[^1] = "...";
		return parts.Length == 1
			? parts[0]
			: "(" + string.Join(", ", parts) + ")";
	}

	private static int CountDisplayValues(ValueInstance value) =>
		value.IsList
			? value.List.Items.Count
			: value.TryGetValueTypeInstance() is { } typeInstance
				? typeInstance.CountDisplayValues()
				: 1;

	private int CountDisplayValues()
	{
		var members = ReturnType.Members;
		var totalValueCount = 0;
		for (var index = 0; index < Values.Length && index < members.Count; index++)
			if (ShouldIncludeMember(members[index], Values[index]))
				totalValueCount += CountDisplayValues(Values[index]);
		return totalValueCount;
	}

	private static bool ShouldIncludeMember(Strict.Language.Member member, ValueInstance value) =>
		!member.IsConstant && !member.Type.IsTrait && !HasSameValueAsDefault(member, value);

	private static bool HasSameValueAsDefault(Strict.Language.Member member, ValueInstance value)
	{
		if (member.InitialValue == null)
			return false;
		return TryCreateDefaultValue(member.InitialValue, out var defaultValue)
			? value.Equals(defaultValue)
			: value.ToExpressionCodeString() == member.InitialValue.ToString();
	}

	private static bool TryCreateDefaultValue(Strict.Language.Expression expression,
		out ValueInstance value)
	{
		switch (expression)
		{
		case List list:
			if (list.TryGetConstantData() is { } listValue)
			{
				value = listValue;
				return true;
			}
			break;
		case Value constantValue:
			value = constantValue.Data;
			return true;
		case MemberCall memberCall when memberCall.Member.InitialValue != null:
			// ReSharper disable once TailRecursiveCall
			return TryCreateDefaultValue(memberCall.Member.InitialValue, out value);
		case MethodCall { Method.Name: Strict.Language.Method.From, Instance: null } methodCall:
			if (methodCall.ReturnType.IsNumber || methodCall.ReturnType.IsText ||
				methodCall.ReturnType.IsBoolean || methodCall.ReturnType.IsCharacter ||
				methodCall.ReturnType.IsEnum || methodCall.ReturnType.IsNone)
				break;
			var values = new ValueInstance[methodCall.Arguments.Count];
			for (var index = 0; index < values.Length; index++)
				if (!TryCreateDefaultValue(methodCall.Arguments[index], out values[index]))
					goto Failed;
			value = new ValueInstance(methodCall.ReturnType, values);
			return true;
		}
		Failed:
		value = default;
		return false;
	}

	public bool Equals(ValueTypeInstance? other)
	{
		if (other is null)
			return false; //ncrunch: no coverage
		if (ReferenceEquals(this, other))
			return true; //ncrunch: no coverage
		if (!other.ReturnType.IsSameOrCanBeUsedAs(ReturnType) || Values.Length != other.Values.Length)
			return false; //ncrunch: no coverage
		for (var i = 0; i < Values.Length; i++)
			if (!Values[i].Equals(other.Values[i]))
				return false; //ncrunch: no coverage
		return true;
	}
}
