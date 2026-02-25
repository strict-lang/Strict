using Strict.Language;
using System.Collections;
using System.Globalization;
using System.Runtime.InteropServices;
using Type = Strict.Language.Type;

namespace Strict.Expressions;

[StructLayout(LayoutKind.Explicit)]
public readonly struct ValueInstance : IEquatable<ValueInstance>
{
	[FieldOffset(0)]
	public readonly Type ReturnType;
	[FieldOffset(8)]
	private readonly bool boolValue;
	[FieldOffset(8)]
	private readonly int intValue;
	[FieldOffset(8)]
	private readonly double doubleValue;
	// Note: string, List<ValueInstance>, and Dictionary<string, ValueInstance> are all reference types
	// so they can share the same memory location at offset 16
	[FieldOffset(16)]
	private readonly string? textValue;
	[FieldOffset(16)]
	private readonly List<ValueInstance>? listValue;
	[FieldOffset(16)]
	private readonly Dictionary<string, ValueInstance>? dictionaryValue;
	[FieldOffset(16)]
	private readonly Dictionary<ValueInstance, ValueInstance>? instanceDictionaryValue;

	private ValueInstance(Type returnType)
	{
		ReturnType = returnType;
		textValue = null;
	}

	private ValueInstance(Type returnType, bool value)
	{
		ReturnType = returnType;
		boolValue = value;
		textValue = null;
	}

	private ValueInstance(Type returnType, double number)
	{
		ReturnType = returnType;
		doubleValue = number;
		textValue = null;
	}

	private ValueInstance(Type returnType, int number)
	{
		ReturnType = returnType;
		intValue = number;
		textValue = null;
	}

	private ValueInstance(Type returnType, string text)
	{
		ReturnType = returnType;
		textValue = text;
	}

	private ValueInstance(Type returnType, List<ValueInstance> list)
	{
		ReturnType = returnType;
		listValue = list;
	}

	private ValueInstance(Type returnType, Dictionary<string, ValueInstance> dict)
	{
		ReturnType = returnType;
		dictionaryValue = dict;
	}

	private ValueInstance(Type returnType, Dictionary<ValueInstance, ValueInstance> dict)
	{
		ReturnType = returnType;
		instanceDictionaryValue = dict;
	}

	public static Type GetEffectiveType(Type returnType) =>
		returnType.IsMutable
			? ((GenericTypeImplementation)returnType).ImplementationTypes[0]
			: returnType;

	public static bool IsNumberType(Type type) =>
		type.Name == Base.Number || type.Members.Count == 1 && type.IsSameOrCanBeUsedAs(type.GetType(Base.Number));

	public static bool IsTextType(Type type) => type.Name is Base.Text or Base.Name;

	// NOTE: Global caching removed - ValueInstance contains Type references that are context-specific.
	// Each package/test has different Type instances, so caching causes type mismatches.
	// Caching should be done at the Executor level where the type context is stable.

	public static ValueInstance CreateNone(Context ctx)
	{
		var noneType = ctx.GetType(Base.None);
		return new ValueInstance(noneType);
	}

	public static ValueInstance CreateBoolean(Context ctx, bool value)
	{
		var boolType = ctx.GetType(Base.Boolean);
		var effectiveType = GetEffectiveType(boolType);
		if (!effectiveType.IsBoolean)
			throw new InvalidTypeValue(boolType, value);
		return new ValueInstance(boolType, value);
	}

	public static ValueInstance CreateNumber(Context ctx, double value)
	{
		var numberType = ctx.GetType(Base.Number);
		var effectiveType = GetEffectiveType(numberType);
		if (!IsNumberType(effectiveType))
			throw new InvalidTypeValue(numberType, value);
		return new ValueInstance(numberType, value);
	}

	public static ValueInstance CreateInteger(Context ctx, int value)
	{
		var numberType = ctx.GetType(Base.Number);
		var effectiveType = GetEffectiveType(numberType);
		if (IsNumberType(effectiveType))
			return CreateNumber(ctx, value);
		if (TryGetIntegerValue(effectiveType, value, out var integerValue))
			return new ValueInstance(numberType, integerValue);
		throw new InvalidTypeValue(numberType, value);
	}

	public static ValueInstance CreateText(Context ctx, string value)
	{
		var textType = ctx.GetType(Base.Text);
		var effectiveType = GetEffectiveType(textType);
		if (!IsTextType(effectiveType))
			throw new InvalidTypeValue(textType, value);
		return new ValueInstance(textType, value);
	}

	public static ValueInstance CreateObject(Type returnType, object? value)
	{
		if (value is Expression)
			throw new InvalidTypeValue(returnType, value);
		var effectiveType = GetEffectiveType(returnType);
		if (effectiveType.Name == Base.None || effectiveType.IsBoolean || IsNumberType(effectiveType) ||
			IsTextType(effectiveType))
			throw new InvalidTypeValue(returnType, value);
		var normalized = CheckIfValueMatchesReturnType(effectiveType, value);
		if (normalized is null)
			throw new InvalidTypeValue(returnType, value);
		if (TryGetIntegerValue(effectiveType, normalized, out var integerValue))
			return new ValueInstance(returnType, integerValue);
		if (normalized is List<ValueInstance> list)
			return new ValueInstance(returnType, list);
		if (normalized is Dictionary<string, ValueInstance> dict)
			return new ValueInstance(returnType, dict);
		if (normalized is Dictionary<ValueInstance, ValueInstance> instanceDict)
			return new ValueInstance(returnType, instanceDict);
		if (normalized is string str)
			return new ValueInstance(returnType, str);
		throw new InvalidTypeValue(returnType, value);
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
				return textValue;
			if (IsCharacterType(effectiveType) || effectiveType.IsEnum)
				return intValue;
			// For List and Dictionary, return the strongly typed value
			if (listValue is List<ValueInstance> list)
				return list;
			if (dictionaryValue is Dictionary<string, ValueInstance> dict)
				return dict;
			if (instanceDictionaryValue is Dictionary<ValueInstance, ValueInstance> instanceDict)
				return instanceDict;
			return null;
		}
	}

	private static bool IsCharacterType(Type type) => type.Name is Base.Character or Base.HashCode ||
		type.Members.Count == 1 && type.IsSameOrCanBeUsedAs(type.GetType(Base.Character));

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

	public static ValueInstance Create(Type returnType, object? value)
	{
		if (value is ValueInstance valueInstance)
			return Create(returnType, valueInstance);
		var effectiveType = GetEffectiveType(returnType);
		if (effectiveType.Name == Base.None)
			return new ValueInstance(returnType);
		if (effectiveType.IsBoolean)
			return new ValueInstance(returnType, (bool)value!);
		if (IsNumberType(effectiveType))
			return value is int intValue
				? new ValueInstance(returnType, (double)intValue)
				: new ValueInstance(returnType, EqualsExtensions.NumberToDouble(value));
		if (IsTextType(effectiveType))
			return new ValueInstance(returnType, (string)value!);
		return CreateObject(returnType, value);
	}

	public static ValueInstance Create(Type returnType, ValueInstance value)
	{
		var effectiveType = GetEffectiveType(returnType);
		if (effectiveType.Name == Base.None)
			return new ValueInstance(returnType);
		if (effectiveType.IsBoolean)
			return new ValueInstance(returnType, value.AsBool());
		if (IsNumberType(effectiveType))
			return new ValueInstance(returnType, value.AsNumber());
		if (IsTextType(effectiveType))
			return new ValueInstance(returnType, (string)value.Value!);
		return CreateObject(returnType, value.Value);
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
		else if (value is IDictionary<string, object?> valueDictionary)
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

	public sealed class UnableToAssignMemberToType(KeyValuePair<string, object?> member,
		IDictionary<string, object?> values,
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

	public ValueInstance? FindInnerValue(string name)
	{
		if (dictionaryValue != null && dictionaryValue.TryGetValue(name, out var value))
			return value;
		return null;
	}

	public Index GetIteratorLength()
	{
		var effectiveType = GetEffectiveType(ReturnType);
		if (IsNumberType(effectiveType))
			return (int)AsNumber();
		if (listValue != null)
			return listValue.Count;
		if (textValue != null)
			return textValue.Length;
		throw new IteratorNotSupported(this);
	}

	public class IteratorNotSupported(ValueInstance instance)
		: ExecutionFailed(instance.ReturnType, instance.ToString());

	public object? GetIteratorValue(int index)
	{
		if (ReturnType.Name is Base.Number or Base.Range)
			return index;
		if (textValue != null)
			return textValue[index];
		if (listValue != null)
			return listValue[index];
		throw new IteratorNotSupported(this);
	}
}




























