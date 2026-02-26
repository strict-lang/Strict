using Type = Strict.Language.Type;

namespace Strict.Expressions;

public class ValueDictionaryInstance(Type returnType, Dictionary<ValueInstance, ValueInstance> items) :
	IEquatable<ValueDictionaryInstance>
{
	public readonly Type ReturnType = returnType;
	public readonly Dictionary<ValueInstance, ValueInstance> Items = items;

	public bool Equals(ValueDictionaryInstance? other) =>
		other is not null && (ReferenceEquals(this, other) ||
			other.ReturnType.IsSameOrCanBeUsedAs(ReturnType) &&
			EqualsExtensions.AreEqual(Items, other.Items));
}