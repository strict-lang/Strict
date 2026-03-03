using Type = Strict.Language.Type;

namespace Strict.Expressions;

public sealed class ValueTypeInstance(Type returnType,
	Dictionary<string, ValueInstance> members) : IEquatable<ValueTypeInstance>
{
	public readonly Type ReturnType = returnType;
	public readonly Dictionary<string, ValueInstance> Members = members;

	public bool Equals(ValueTypeInstance? other) =>
		other is not null && (ReferenceEquals(this, other) ||
			other.ReturnType.IsSameOrCanBeUsedAs(ReturnType) && Members.Count == other.Members.Count &&
			Members.All(kvp =>
				other.Members.TryGetValue(kvp.Key, out var value) && kvp.Value.Equals(value)));
}