using System.Collections;
using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict.Expressions;

/// <summary>
/// Optimized for size, always just contains 2 values, a pointer to the type, string, list,
/// dictionary, or type instance and if it is a primitive (most common, most lines just return
/// None or True) the None, Boolean, or Number data. This is also
/// </summary>
public readonly struct ValueInstance : IEquatable<ValueInstance>
{
	public ValueInstance(Type noneReturnType) => value = noneReturnType;
	/// <summary>
	/// If number is IsText, value points to a string (only non Mutable, for Mutable IsType is used)
	/// If number is IsList, value points to ValueListInstance (ReturnType and Items)
	/// If number is IsDictionary, value points to ValueDictionaryInstance (ReturnType and Items)
	/// If number is IsType, then this points to a TypeValueInstance containing the ReturnType.
	/// In all other cases this is a primitive (None, Boolean, Number) and value is the ReturnType.
	/// </summary>
	private readonly object? value;
	/// <summary>
	/// Stores the value only if it is a None, Boolean, or Number. Otherwise use value below.
	/// </summary>
	internal readonly double number;
	/// <summary>
	/// These are all unsupported double values, which we don't allow or support.
	/// </summary>
	private const double IsText = -7.90897526e307;
	private const double IsList = -7.81590825e307;
	private const double IsDictionary = -7.719027815e307;
	private const double IsType = -7.657178621e307;

	public ValueInstance(Type booleanReturnType, bool isTrue)
	{
		value = booleanReturnType;
		number = isTrue
			? 1
			: 0;
	}

	public ValueInstance(Type numberReturnType, double setNumber)
	{
		if (setNumber is IsText or IsList or IsDictionary or IsType)
			throw new InvalidTypeValue(numberReturnType, setNumber);
		value = numberReturnType;
		number = setNumber;
	}

	public sealed class InvalidTypeValue(Type returnType, object? value) : ParsingFailed(returnType,
		0, value switch
		{
			null => "null",
			Expression => "Expression " + value + " needs to be evaluated!",
			_ => value + ""
		} + " (" + value?.GetType() + ") for " + returnType.Name);

	public ValueInstance(string text)
	{
		value = text;
		number = IsText;
	}

	public ValueInstance(Type returnType, List<ValueInstance> list)
	{
		value = new ValueListInstance(returnType, list);
		number = IsList;
	}

	public ValueInstance(Type returnType, Dictionary<ValueInstance, ValueInstance> dictionary)
	{
		value = new ValueDictionaryInstance(returnType, dictionary);
		number = IsDictionary;
	}

	public ValueInstance(ValueTypeInstance instance)
	{
		value = instance;
		number = IsType;
	}

	/*should be handled at Creation time, primitives can be mutable, but anything else need to
	become a ValueTypeInstance to support mutable (as string, list, and dictionary have no
	ReturnType)

		public static bool IsNumberType(Type type) =>
			type.Name == Base.Number || type.Members.Count == 1 && type.IsSameOrCanBeUsedAs(type.GetType(Base.Number));

		public static bool IsTextType(Type type) => type.Name is Base.Text or Base.Name;
	/*still needed?
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
	*/
	/*no idea what all this is about
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

	//not really needed ..
		//public override bool Equals(object? obj) => obj is ValueInstance other && Equals(other);

		public override int GetHashCode()
		{
			var effectiveType = GetEffectiveType(ReturnType);
			if (IsNumberType(effectiveType))
				return HashCode.Combine(ReturnType, AsNumber());
			if (IsCharacterType(effectiveType) || effectiveType.IsEnum)
				return HashCode.Combine(ReturnType, intValue);
			return HashCode.Combine(ReturnType, Value);
		}

	maybe needed?


			return false;
			var effectiveType = GetEffectiveType(ReturnType);
			var otherEffectiveType = GetEffectiveType(other.ReturnType);
			if (IsNumberType(effectiveType) && IsNumberType(otherEffectiveType))
				return AsNumber() == other.AsNumber();
			if ((IsCharacterType(effectiveType) || effectiveType.IsEnum) &&
				(IsCharacterType(otherEffectiveType) || otherEffectiveType.IsEnum))
				return intValue == other.intValue;
		*/
	public override string ToString() => GetTypeName() + ": " + ToExpressionCodeString();

	private string GetTypeName() =>
		number == IsText
			? Base.Text
			: number == IsList
				? ((ValueListInstance)value!).ReturnType.Name
				: number == IsDictionary
					? ((ValueDictionaryInstance)value!).ReturnType.Name
					: number == IsType
						? ((ValueTypeInstance)value!).ReturnType.Name
						: ((Type)value!).Name;

	public string ToExpressionCodeString()
	{
		if (number == IsText)
			return "\"" + EscapeText((string)value!) + "\"";
		if (number == IsList)
			return ((ValueListInstance)value!).Items.Select(v => v.ToExpressionCodeString()).ToWordList();
		if (number == IsDictionary)
			return ((ValueDictionaryInstance)value!).Items.Select(kv =>
					"(" + kv.Key.ToExpressionCodeString() + ", " + kv.Value.ToExpressionCodeString() + ")").
				ToWordList();
		if (number == IsType)
			return ((ValueTypeInstance)value!).ToString()!;
		var primitiveType = GetEffectiveType((Type)value!);
		if (primitiveType.Name == Base.Number)
			return number.ToString(System.Globalization.CultureInfo.InvariantCulture);
		if (primitiveType.Name == Base.Boolean)
			return number == 0
				? "false"
				: "true";
		return primitiveType.Name == Base.None
			? Base.None
			: throw new NotSupportedException(primitiveType.ToString());
	}

	private static string EscapeText(string s) => s.Replace("\\", @"\\").Replace("\"", "\\\"");

	public static Type GetEffectiveType(Type returnType) =>
		returnType.IsMutable
			? ((GenericTypeImplementation)returnType).ImplementationTypes[0]
			: returnType;

	/// <summary>
	/// Equals is a bit more complex as we need to handle all different kinds of ValueInstances we
	/// could have, either primary, text, list, dictionary, or type. Or any of those mixed.
	/// </summary>
	public bool Equals(ValueInstance other)
	{
		if (number == IsType)
		{
			var instance = (ValueTypeInstance)value!;
			if (other.number == IsType)
				return instance.Equals((ValueTypeInstance)other.value!);
			if (other.number != IsList && other.number != IsDictionary && other.number != IsText &&
				instance.Members.TryGetValue("number", out var numberMember) &&
				numberMember.number == other.number)
				return true;
		}
		else if (other.number == IsType && number != IsList && number != IsDictionary &&
			number != IsText &&
			((ValueTypeInstance)other.value!).Members.TryGetValue("number",
				out var otherNumberMember) && otherNumberMember.number == number)
			return true;
		if (number != other.number)
			return false;
		if (number == IsText)
			return (string)value! == (string)other.value!;
		if (number == IsList)
			return EqualsExtensions.AreEqual(value, other.value);
		return number == IsDictionary
			? EqualsExtensions.AreEqual(value, other.value)
			: ((Type)other.value!).IsSameOrCanBeUsedAs((Type)value!);
	}

	public override int GetHashCode() => HashCode.Combine(number, value);
}