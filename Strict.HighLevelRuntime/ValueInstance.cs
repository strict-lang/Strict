using Strict.Language;
using System.Collections;
using System.Globalization;
using Type = Strict.Language.Type;

namespace Strict.HighLevelRuntime;

public readonly struct ValueInstance : IEquatable<ValueInstance>
{
	public readonly Type ReturnType;
	private readonly bool boolValue;
	private readonly int intValue;
	private readonly double doubleValue;
	private readonly object? objectValue;

	/// <summary>
	/// Create a None instance without storing a value.
	/// </summary>
	public ValueInstance(Type returnType, Statistics? statistics = null)
	{
		ReturnType = returnType;
		var effectiveType = GetEffectiveType(returnType);
		if (effectiveType.Name != Base.None)
			throw new InvalidTypeValue(returnType, null);
		objectValue = null;
		if (statistics != null)
			UpdateStatistics(statistics, effectiveType);
	}


	/// <summary>
	/// Create a Boolean instance without boxing.
	/// </summary>
	public ValueInstance(Type returnType, bool value, Statistics? statistics = null)
	{
		ReturnType = returnType;
		var effectiveType = GetEffectiveType(returnType);
		if (!effectiveType.IsBoolean)
			throw new InvalidTypeValue(returnType, value);
		boolValue = value;
		if (statistics != null)
			UpdateStatistics(statistics, effectiveType);
	}

	/// <summary>
	/// Create a Number instance without boxing.
	/// </summary>
	public ValueInstance(Type returnType, double number, Statistics? statistics = null)
	{
		ReturnType = returnType;
		var effectiveType = GetEffectiveType(returnType);
		if (!IsNumberType(effectiveType))
			throw new InvalidTypeValue(returnType, number);
		doubleValue = number;
		if (statistics != null)
			UpdateStatistics(statistics, effectiveType);
	}

	/// <summary>
	/// Create a Number or Integer instance from an integer value.
	/// </summary>
	public ValueInstance(Type returnType, int number, Statistics? statistics = null)
	{
		ReturnType = returnType;
		var effectiveType = GetEffectiveType(returnType);
		if (IsNumberType(effectiveType))
			doubleValue = number;
		else if (TryGetIntegerValue(effectiveType, number, out var integerValue))
			intValue = integerValue;
		else
			throw new InvalidTypeValue(returnType, number);
		if (statistics != null)
			UpdateStatistics(statistics, effectiveType);
	}

	/// <summary>
	/// Create a Text instance without boxing.
	/// </summary>
	public ValueInstance(Type returnType, string text, Statistics? statistics = null)
	{
		ReturnType = returnType;
		var effectiveType = GetEffectiveType(returnType);
		if (!IsTextType(effectiveType))
			throw new InvalidTypeValue(returnType, text);
		objectValue = text;
		if (statistics != null)
			UpdateStatistics(statistics, effectiveType);
	}

	public ValueInstance(Type returnType, object? value, Statistics? statistics = null)
	{
		if (value is Expression)
			throw new InvalidTypeValue(returnType, value);
		ReturnType = returnType;
		var effectiveType = GetEffectiveType(returnType);
		if (effectiveType.Name == Base.None || effectiveType.IsBoolean || IsNumberType(effectiveType) ||
			IsTextType(effectiveType))
			throw new InvalidTypeValue(returnType, value);
		var normalized = CheckIfValueMatchesReturnType(effectiveType, value);
		if (normalized is null)
			throw new InvalidTypeValue(returnType, value);
		if (TryGetIntegerValue(effectiveType, normalized, out var integerValue))
			intValue = integerValue;
		else
			objectValue = normalized;
		if (statistics != null)
			UpdateStatistics(statistics, effectiveType);
	}


	private static void UpdateStatistics(Statistics statistics, Type effectiveType)
	{
		statistics.ValueInstanceCount++;
		if (effectiveType.IsBoolean)
			statistics.BooleanCount++;
		else if (effectiveType.Name == Base.Number)
			statistics.NumberCount++;
		else if (effectiveType.Name is Base.Text or Base.Name)
			statistics.TextCount++;
		else if (effectiveType.Name == Base.List ||
			effectiveType is GenericTypeImplementation { Generic.Name: Base.List })
			statistics.ListCount++;
		else if (effectiveType.Name == Base.Dictionary ||
			effectiveType is GenericTypeImplementation { Generic.Name: Base.Dictionary })
			statistics.DictionaryCount++;
	}

	public object? Value
	{
		get
		{
			var effectiveType = GetEffectiveType(ReturnType);
			if (effectiveType.Name == Base.None)
				return null;
			if (effectiveType.IsBoolean)
				return boolValue;
			if (IsNumberType(effectiveType))
				return doubleValue;
			if (IsTextType(effectiveType))
				return objectValue;
			if (IsCharacterType(effectiveType) || effectiveType.IsEnum)
				return intValue;
			return objectValue;
		}
	}

	public double AsNumber()
	{
		var effectiveType = GetEffectiveType(ReturnType);
		if (IsNumberType(effectiveType))
			return doubleValue;
		if (IsCharacterType(effectiveType) || effectiveType.IsEnum)
			return intValue;
		if (effectiveType.IsBoolean)
			return boolValue ? 1d : 0d;
		return 0d;
	}

	public bool AsBool()
	{
		var effectiveType = GetEffectiveType(ReturnType);
		if (effectiveType.IsBoolean)
			return boolValue;
		return (bool)Value!;
	}


	private static Type GetEffectiveType(Type returnType) =>
		returnType.IsMutable ? ((GenericTypeImplementation)returnType).ImplementationTypes[0] : returnType;

	private static bool IsTextType(Type type) => type.Name is Base.Text or Base.Name;

	internal static bool IsNumberType(Type type) =>
		type.Name == Base.Number || type.Members.Count == 1 && type.IsSameOrCanBeUsedAs(type.GetType(Base.Number));

	private static bool IsCharacterType(Type type) => type.Name is Base.Character or Base.HashCode ||
		type.Members.Count == 1 && type.IsSameOrCanBeUsedAs(type.GetType(Base.Character));


	internal static ValueInstance Create(Type returnType, object? value, Statistics? statistics = null)
	{
		if (value is ValueInstance valueInstance)
			return Create(returnType, valueInstance, statistics);
		var effectiveType = GetEffectiveType(returnType);
		if (effectiveType.Name == Base.None)
			return new ValueInstance(returnType, statistics);
		if (effectiveType.IsBoolean)
			return new ValueInstance(returnType, (bool)value!, statistics);
		if (IsNumberType(effectiveType))
			return value is int intValue
				? new ValueInstance(returnType, intValue, statistics)
				: new ValueInstance(returnType, EqualsExtensions.NumberToDouble(value), statistics);
		if (IsTextType(effectiveType))
			return new ValueInstance(returnType, (string)value!, statistics);
		return new ValueInstance(returnType, value, statistics);
	}

	internal static ValueInstance Create(Type returnType, ValueInstance value, Statistics? statistics = null)
	{
		var effectiveType = GetEffectiveType(returnType);
		if (effectiveType.Name == Base.None)
			return new ValueInstance(returnType, statistics);
		if (effectiveType.IsBoolean)
			return new ValueInstance(returnType, value.AsBool(), statistics);
		if (IsNumberType(effectiveType))
			return new ValueInstance(returnType, value.AsNumber(), statistics);
		if (IsTextType(effectiveType))
			return new ValueInstance(returnType, (string)value.Value!, statistics);
		return new ValueInstance(returnType, value.Value, statistics);
	}

	private static bool TryGetIntegerValue(Type type, object value, out int intValue)
	{
		if (IsCharacterType(type))
		{
			if (value is int valueInt)
			{
				intValue = valueInt;
				return true;
			}
			if (value is char valueChar)
			{
				intValue = valueChar;
				return true;
			}
		}
		if (type.IsEnum && value is int enumValue)
		{
			intValue = enumValue;
			return true;
		}
		intValue = 0;
		return false;
	}

  private static object? CheckIfValueMatchesReturnType(Type type, object? value)
	{
		if (type.Name == Base.None)
		{
			if (value is not null)
				throw new InvalidTypeValue(type, value);
		}
		else if (value is null)
			throw new InvalidTypeValue(type, value);
		else if (type.IsBoolean)
		{
			if (value is not bool)
				throw new InvalidTypeValue(type, value);
		}
   else if (type.IsEnum)
		{
			if (value is not int && value is not string)
				throw new InvalidTypeValue(type, value);
		}
   else if (IsTextType(type))
		{
			if (value is not string)
				throw new InvalidTypeValue(type, value);
		}
    else if (IsCharacterType(type))
		{
			if (value is double doubleValue)
				return (int)doubleValue;
			if (value is not char && value is not int)
				throw new InvalidTypeValue(type, value);
		}
		else if (type.Name == Base.List || type.Name == Base.Dictionary ||
			type is GenericTypeImplementation { Generic.Name: Base.List } ||
			type is GenericTypeImplementation { Generic.Name: Base.Dictionary } ||
			type is GenericType { Generic.Name: Base.List } || type is GenericType
			{
				Generic.Name: Base.Dictionary
			})
		{
			if (value is IList<Expression>)
				throw new InvalidTypeValue(type, value);
			if (value is not IList and not IDictionary and not string)
				throw new InvalidTypeValue(type, value);
		}
   else if (IsNumberType(type))
		{
			if (value is char charValue)
				return (int)charValue;
			if (value is not double && value is not int)
				throw new InvalidTypeValue(type, value);
		}
		else if (value is IDictionary<string, ValueInstance> valueDictionary)
		{
			foreach (var assignMember in valueDictionary)
				if (type.Members.All(m =>
					!m.Name.Equals(assignMember.Key, StringComparison.OrdinalIgnoreCase)))
					throw new UnableToAssignMemberToType(assignMember, valueDictionary, type);
		}
		else if (!type.IsSameOrCanBeUsedAs(type.GetType(Base.Error)))
			throw new InvalidTypeValue(type, value);
		return value;
	}

	public sealed class UnableToAssignMemberToType(KeyValuePair<string, ValueInstance> member,
		IDictionary<string, ValueInstance> values,
		Type returnType) : ExecutionFailed(returnType,
		"Can't assign member " + member + " (of " + values.DictionaryToWordList() + ") to " +
		returnType + " " + returnType.Members.ToBrackets());

	public sealed class InvalidTypeValue(Type returnType, object? value) : ExecutionFailed(returnType,
		value switch
		{
			null => "null",
			Expression => "Expression " + value + " needs to be evaluated!",
			IEnumerable valueEnumerable => valueEnumerable.EnumerableToWordList(", ", true),
			_ => value + ""
		} + " (" + value?.GetType() + ") for " + returnType.Name);

	public override string ToString()
	{
		var effectiveType = GetEffectiveType(ReturnType);
		if (effectiveType.IsBoolean)
			return $"{AsBool()}";
		if (IsNumberType(effectiveType))
			return ReturnType.Name + ":" + AsNumber().ToString(CultureInfo.InvariantCulture);
		if (Value is IEnumerable valueEnumerable)
			return $"{ReturnType.Name}: " + valueEnumerable.EnumerableToWordList(", ", true);
		if (ReturnType.IsIterator)
			return $"Unknown Iterator {ReturnType.Name}: {Value}";
		return $"{ReturnType.Name}:{Value}";
	}


	public bool Equals(ValueInstance other)
	{
		if (!other.ReturnType.IsSameOrCanBeUsedAs(ReturnType))
			return false;
		var effectiveType = GetEffectiveType(ReturnType);
		var otherEffectiveType = GetEffectiveType(other.ReturnType);
		if (IsNumberType(effectiveType) && IsNumberType(otherEffectiveType))
			return AsNumber() == other.AsNumber();
		if ((IsCharacterType(effectiveType) || effectiveType.IsEnum) &&
			(IsCharacterType(otherEffectiveType) || otherEffectiveType.IsEnum))
			return intValue == other.intValue;
		return EqualsExtensions.AreEqual(Value, other.Value);
	}

	public override bool Equals(object? obj) => obj is ValueInstance other && Equals(other);

	public override int GetHashCode()
	{
		var effectiveType = GetEffectiveType(ReturnType);
		if (IsNumberType(effectiveType))
			return HashCode.Combine(ReturnType, AsNumber());
		if (IsCharacterType(effectiveType) || effectiveType.IsEnum)
			return HashCode.Combine(ReturnType, intValue);
		return HashCode.Combine(ReturnType, Value);
	}

	public object? FindInnerValue(string name)
	{
		if (objectValue is IDictionary<string, ValueInstance> valueDictionary)
			if (valueDictionary.TryGetValue(name, out var value))
				return value.Value;
		return null;
	}

	public Index GetIteratorLength()
	{
		var effectiveType = GetEffectiveType(ReturnType);
		if (IsNumberType(effectiveType))
			return (int)AsNumber();
		return objectValue switch
		{
			IList list => list.Count,
			string text => text.Length,
			_ => throw new IteratorNotSupported(this)
		};
	}

	public class IteratorNotSupported(ValueInstance instance)
		: ExecutionFailed(instance.ReturnType, instance.ToString());

	public object? GetIteratorValue(int index) =>
		ReturnType.Name is Base.Number or Base.Range
			? index
			: objectValue is string str
				? str[index]
				: objectValue is IList list
					? list[index]
					: throw new IteratorNotSupported(this);
}



