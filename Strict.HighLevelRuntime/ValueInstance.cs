using Strict.Language;
using System.Collections;
using Strict.Expressions;
using Type = Strict.Language.Type;

namespace Strict.HighLevelRuntime;

public sealed class ValueInstance
{
	public ValueInstance(Type returnType, object? value)
	{
		ReturnType = returnType;
		Value = value is Value expressionValue
			? expressionValue.Data
			: value;
		if (ReturnType.IsMutable)
			throw new InvalidTypeValue(ReturnType, Value);
		if (ReturnType.Name == Base.None)
		{
			if (Value is not null)
				throw new InvalidTypeValue(ReturnType, Value);
		}
		else if (Value is null)
			throw new InvalidTypeValue(ReturnType, Value);
		else if (ReturnType.IsBoolean)
		{
			if (Value is not bool)
				throw new InvalidTypeValue(ReturnType, Value);
		}
		else if (ReturnType.IsEnum)
		{
			if (Value is not int && Value is not string)
				throw new InvalidTypeValue(ReturnType, Value);
		}
		else if (ReturnType.Name is Base.Text or Base.Name)
		{
			if (Value is not string)
				throw new InvalidTypeValue(ReturnType, Value);
		}
		else if (ReturnType.Name is Base.Character or Base.HashCode)
		{
			if (Value is double valueDouble)
				Value = (int)valueDouble;
			if (Value is not char && Value is not int)
				throw new InvalidTypeValue(ReturnType, Value);
		}
		else if (ReturnType.Name == Base.List || ReturnType.Name == Base.Dictionary ||
			ReturnType is GenericTypeImplementation { Generic.Name: Base.List } ||
			ReturnType is GenericTypeImplementation { Generic.Name: Base.Dictionary })
		{
			if (Value is not IList && Value is not IDictionary && Value is not string)
				throw new InvalidTypeValue(ReturnType, Value);
		}
		else if (ReturnType.Name == Base.Number || ReturnType.Members.Count == 1 &&
			ReturnType.IsSameOrCanBeUsedAs(ReturnType.GetType(Base.Number)))
		{
			if (Value is not double && Value is not int)
				throw new InvalidTypeValue(ReturnType, Value);
		}
		else if (Value is IDictionary<string, object?> valueDictionary)
		{
			foreach (var assignMember in valueDictionary)
				if (ReturnType.Members.All(m => !m.Name.Equals(assignMember.Key, StringComparison.OrdinalIgnoreCase)))
					throw new UnableToAssignMemberToType(assignMember, valueDictionary, ReturnType);
		}
		else if (!ReturnType.IsSameOrCanBeUsedAs(ReturnType.GetType(Base.Error)))
			throw new InvalidTypeValue(ReturnType, Value);
	}

	public Type ReturnType { get; }
	public object? Value { get; }

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
}