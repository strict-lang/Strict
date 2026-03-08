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
