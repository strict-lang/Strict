using Type = Strict.Language.Type;

namespace Strict.Expressions;

public sealed class ValueListInstance(Type returnType, List<ValueInstance> items) :
	IEquatable<ValueListInstance>
{
	public readonly Type ReturnType = returnType;
	public readonly List<ValueInstance> Items = items;

	public bool Equals(ValueListInstance? other) =>
		other is not null && (ReferenceEquals(this, other) ||
			other.ReturnType.IsSameOrCanBeUsedAs(ReturnType) && Items.SequenceEqual(other.Items));

	//ncrunch: no coverage start
	public override bool Equals(object? other) => Equals(other as ValueListInstance);
	public override int GetHashCode() => HashCode.Combine(ReturnType, Items);
}