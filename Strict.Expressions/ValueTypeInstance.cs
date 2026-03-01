using Type = Strict.Language.Type;

namespace Strict.Expressions;

public sealed class ValueTypeInstance(Type returnType, Dictionary<string, ValueInstance> members) :
	IEquatable<ValueTypeInstance>
{
	public readonly Type ReturnType = returnType;
	public readonly Dictionary<string, ValueInstance> Members = members;

	public bool Equals(ValueTypeInstance? other) =>
		other is not null && (ReferenceEquals(this, other) ||
			other.ReturnType.IsSameOrCanBeUsedAs(ReturnType) &&
			EqualsExtensions.AreEqual(Members, other.Members));
}