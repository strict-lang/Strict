using Type = Strict.Language.Type;

namespace Strict.Expressions;

public sealed class ValueTypeInstance(Type returnType, ValueInstance[] values)
	: IEquatable<ValueTypeInstance>
{
	public readonly Type ReturnType = returnType;
	public readonly ValueInstance[] Values = values;

	/// <summary>
	/// Finds the value for the named member using a positional scan of ReturnType.Members.
	/// Skips constant members so callers fall through to their InitialValue evaluation.
	/// </summary>
	public bool TryGetValue(string name, out ValueInstance value)
	{
		var members = ReturnType.Members;
		for (var i = 0; i < members.Count; i++)
			if (!members[i].IsConstant &&
				members[i].Name.Equals(name, StringComparison.OrdinalIgnoreCase))
			{
				value = Values[i];
				return true;
			}
		value = default;
		return false;
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
		var parts = new List<string>(Values.Length);
		for (var index = 0; index < Values.Length && index < members.Count; index++)
			if (ShouldIncludeMember(members[index], Values[index]))
				parts.Add(Values[index].ToExpressionCodeString());
		return parts.Count == 0
			? ReturnType.Name
			: "(" + string.Join(", ", parts) + ")";
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
