using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict.Expressions;

public sealed class ValueTypeInstance(Type returnType,
	ValueInstance[] values) : IEquatable<ValueTypeInstance>
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

	public ValueInstance this[string name]
	{
		get
		{
			if (TryGetValue(name, out var value))
				return value;
			throw new KeyNotFoundException("Member not found: " + name);
		}
	}

	/// <summary>
	/// Finds the index of the member with the given name (case-insensitive).
	/// </summary>
	public static int FindMemberIndex(IReadOnlyList<Member> members, string name)
	{
		for (var i = 0; i < members.Count; i++)
			if (members[i].Name.Equals(name, StringComparison.OrdinalIgnoreCase))
				return i;
		return 0;
	}

	public bool Equals(ValueTypeInstance? other)
	{
		if (other is null)
			return false;
		if (ReferenceEquals(this, other))
			return true;
		if (!other.ReturnType.IsSameOrCanBeUsedAs(ReturnType))
			return false;
		if (Values.Length != other.Values.Length)
			return false;
		for (var i = 0; i < Values.Length; i++)
			if (!Values[i].Equals(other.Values[i]))
				return false;
		return true;
	}

	public override string ToString()
	{
		var typeMembers = ReturnType.Members;
		if (Values.Length == 0)
			return ReturnType.ToString();
		var result = new string[Values.Length];
		for (var i = 0; i < Values.Length; i++)
			result[i] = typeMembers[i].Name + "=" +
				(Values[i].HasValue ? Values[i].ToString() : "(unset)");
		return ReturnType + ": " + string.Join("; ", result);
	}
}
