using Strict.Language;
using System.Collections;
using Strict.Expressions;
using Type = Strict.Language.Type;

namespace Strict.HighLevelRuntime;

public sealed class ValueInstance : IEquatable<ValueInstance>
{
	public ValueInstance(Type returnType, object? value)
	{
		ReturnType = returnType;
		Value = value is Value expressionValue
			? expressionValue.Data
			: value;
		CheckIfValueMatchesReturnType(ReturnType.IsMutable
			? ((GenericTypeImplementation)ReturnType).ImplementationTypes[0]
			: ReturnType);
	}

	public Type ReturnType { get; }
	public object? Value { get; set; }

	private void CheckIfValueMatchesReturnType(Type type)
	{
		if (type.Name == Base.None)
		{
			if (Value is not null)
				throw new InvalidTypeValue(type, Value);
		}
		else if (Value is null)
			throw new InvalidTypeValue(type, Value);
		else if (type.IsBoolean)
		{
			if (Value is not bool)
				throw new InvalidTypeValue(type, Value);
		}
		else if (type.IsEnum)
		{
			if (Value is not int && Value is not string)
				throw new InvalidTypeValue(type, Value);
		}
		else if (type.Name is Base.Text or Base.Name)
		{
			if (Value is not string)
				throw new InvalidTypeValue(type, Value);
		}
		else if (type.Name is Base.Character or Base.HashCode)
		{
			if (Value is double doubleValue)
				Value = (int)doubleValue;
			if (Value is not int)
				throw new InvalidTypeValue(type, Value);
		}
		else if (type.Name == Base.List || type.Name == Base.Dictionary ||
			type is GenericTypeImplementation { Generic.Name: Base.List } ||
			type is GenericTypeImplementation { Generic.Name: Base.Dictionary })
		{
			if (Value is IList<Expression>)
				throw new InvalidTypeValue(type, Value);
			if (Value is not IList and not IDictionary and not string)
				throw new InvalidTypeValue(type, Value);
		}
		else if (type.Name == Base.Number || type.Members.Count == 1 &&
			type.IsSameOrCanBeUsedAs(type.GetType(Base.Number)))
		{
			if (Value is not double && Value is not int)
				throw new InvalidTypeValue(type, Value);
		}
		else if (Value is IDictionary<string, object?> valueDictionary)
		{
			foreach (var assignMember in valueDictionary)
				if (type.Members.All(m => !m.Name.Equals(assignMember.Key, StringComparison.OrdinalIgnoreCase)))
					throw new UnableToAssignMemberToType(assignMember, valueDictionary, type);
		}
		else if (!type.IsSameOrCanBeUsedAs(type.GetType(Base.Error)))
			throw new InvalidTypeValue(type, Value);
	}

	public sealed class UnableToAssignMemberToType(KeyValuePair<string, object?> member,
		IDictionary<string, object?> values, Type returnType) : ExecutionFailed(returnType,
		"Can't assign member " + member + " (of " + values.DictionaryToWordList() + ") to " +
		returnType + " " + returnType.Members.ToBrackets());

	public sealed class InvalidTypeValue(Type returnType, object? value) : ExecutionFailed(
		returnType, (value is IEnumerable valueEnumerable
			? valueEnumerable.EnumerableToWordList(", ", true)
			: value + "") + " (" + value?.GetType() + ") for " + returnType.Name);

	public override string ToString() =>
		ReturnType.Name == Base.Boolean
			? $"{Value}"
			: Value is IEnumerable valueEnumerable
				? $"{ReturnType.Name}: " + valueEnumerable.EnumerableToWordList(", ", true)
				: ReturnType.IsIterator
					? $"Unknown Iterator {ReturnType.Name}: {Value}"
					: $"{ReturnType.Name}:{Value}";

	public bool Equals(ValueInstance? other) =>
		ReferenceEquals(this, other) || other is not null &&
		other.ReturnType.IsSameOrCanBeUsedAs(ReturnType) &&
		EqualsExtensions.AreEqual(Value, other.Value);

	public override bool Equals(object? obj) =>
		ReferenceEquals(this, obj) || obj is ValueInstance other && Equals(other);

	public override int GetHashCode() => HashCode.Combine(ReturnType, Value);

	public object? FindInnerValue(string name)
	{
		if (Value is IDictionary<string, object?> valueDictionary)
			if (valueDictionary.TryGetValue(name, out var value))
				return value;
		return null;
	}

	public Range GetRange()
	{
		if (Value is IDictionary<string, object?> valueDictionary)
			return new Range(Convert.ToInt32(valueDictionary["Start"]),
				Convert.ToInt32(valueDictionary["ExclusiveEnd"]));
		throw new IteratorNotSupported(this);
	}

	public class IteratorNotSupported(ValueInstance instance)
		: ExecutionFailed(instance.ReturnType, instance.ToString());

	public Index GetIteratorLength() =>
		Value switch
		{
			IList list => list.Count,
			int count => count,
			double countDouble => (int)countDouble,
			string text => text.Length,
			_ => throw new IteratorNotSupported(this)
		};

	public object? GetIteratorValue(int index) =>
		ReturnType.Name is Base.Number or Base.Range
			? index
			: Value is string
				? (int)((string)Value!)[index]
				: Value is IList list
					? list[index]
					: throw new IteratorNotSupported(this);
}