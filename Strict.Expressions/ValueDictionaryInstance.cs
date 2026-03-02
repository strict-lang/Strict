using Type = Strict.Language.Type;

namespace Strict.Expressions;

public sealed class ValueDictionaryInstance(Type returnType,
	Dictionary<ValueInstance, ValueInstance> items) : IEquatable<ValueDictionaryInstance>
{
	public readonly Type ReturnType = returnType;
	public readonly Dictionary<ValueInstance, ValueInstance> Items = items;

	public bool Equals(ValueDictionaryInstance? other) =>
		other is not null && (ReferenceEquals(this, other) ||
			other.ReturnType.IsSameOrCanBeUsedAs(ReturnType) &&
			EqualsExtensions.AreEqual(Items, other.Items));

	//ncrunch: no coverage start
	public override bool Equals(object? other) => Equals(other as ValueDictionaryInstance);
	public override int GetHashCode() => HashCode.Combine(ReturnType, Items);
}