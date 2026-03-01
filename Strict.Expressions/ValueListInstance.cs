using Type = Strict.Language.Type;

namespace Strict.Expressions;

public class ValueListInstance(Type returnType, List<ValueInstance> items) :
	IEquatable<ValueListInstance>
{
	public readonly Type ReturnType = returnType;
	public readonly List<ValueInstance> Items = items;

	public bool Equals(ValueListInstance? other) =>
		other is not null && (ReferenceEquals(this, other) ||
			other.ReturnType.IsSameOrCanBeUsedAs(ReturnType) &&
			EqualsExtensions.AreEqual(Items, other.Items));
}