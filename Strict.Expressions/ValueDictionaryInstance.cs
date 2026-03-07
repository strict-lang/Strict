using Type = Strict.Language.Type;

namespace Strict.Expressions;

public sealed class ValueDictionaryInstance(Type returnType,
	Dictionary<ValueInstance, ValueInstance> items) : IEquatable<ValueDictionaryInstance>
{
	public readonly Type ReturnType = returnType;
	public readonly Dictionary<ValueInstance, ValueInstance> Items = items;

	public bool Equals(ValueDictionaryInstance? other)
	{
		if (ReferenceEquals(this, other))
			return true; //ncrunch: no coverage
		if (other is null || Items.Count != other.Items.Count ||
			!other.ReturnType.IsSameOrCanBeUsedAs(ReturnType))
			return false;
		foreach (var kvp in Items)
			if (!other.Items.TryGetValue(kvp.Key, out var value) || !kvp.Value.Equals(value))
				return false;
		return true;
	}

	//ncrunch: no coverage start
	public override bool Equals(object? other) => Equals(other as ValueDictionaryInstance);
	public override int GetHashCode() => HashCode.Combine(ReturnType, Items);
}