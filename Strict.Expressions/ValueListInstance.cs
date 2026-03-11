using Type = Strict.Language.Type;

namespace Strict.Expressions;

public sealed class ValueListInstance : IEquatable<ValueListInstance>
{
	public ValueListInstance(Type returnType, IEnumerable<ValueInstance> items)
	{
		ReturnType = returnType;
		Items = new List<ValueInstance>(items);
	}

	public ValueListInstance(Type returnType, List<ValueInstance> items)
	{
		ReturnType = returnType;
		Items = items;
	}

	public readonly Type ReturnType;
	public readonly List<ValueInstance> Items;

	public bool Equals(ValueListInstance? other) =>
		other is not null && (ReferenceEquals(this, other) ||
			other.ReturnType.IsSameOrCanBeUsedAs(ReturnType) && Items.SequenceEqual(other.Items));

	//ncrunch: no coverage start
	public override bool Equals(object? other) => Equals(other as ValueListInstance);
	public override int GetHashCode() => HashCode.Combine(ReturnType, Items);
}