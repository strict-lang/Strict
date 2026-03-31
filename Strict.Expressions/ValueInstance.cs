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
	/// If number is TextId, value points to a string (only non-Mutable, for Mutable TypeId is used)
	/// If number is ListId, value points to ValueListInstance (ReturnType and Items)
	/// If number is DictionaryId, value points to ValueDictionaryInstance (ReturnType and Items)
	/// If number is TypeId, then this points to a TypeValueInstance containing the ReturnType.
	/// In all other cases this is a primitive (None, Boolean, Number), and value is the ReturnType.
	/// </summary>
	private readonly object value;
	/// <summary>
	/// Stores the value only if it is a None, Boolean, or Number. Otherwise use value below.
	/// </summary>
	internal readonly double number;
	/// <summary>
	/// These are all unsupported double values, which we don't allow or support.
	/// </summary>
	private const double TextId = -7.90897526e307;
	private const double ListId = -7.81590825e307;
	private const double DictionaryId = -7.719027815e307;
	private const double TypeId = -7.657178621e307;

	public ValueInstance(Type booleanReturnType, bool isTrue)
	{
		value = booleanReturnType;
		number = isTrue
			? 1
			: 0;
	}

	public ValueInstance(Type numberReturnType, double setNumber)
	{
		if (setNumber is TextId or ListId or DictionaryId or TypeId)
			throw new InvalidTypeValue(numberReturnType, setNumber);
		value = numberReturnType;
		number = setNumber;
	}

	public sealed class InvalidTypeValue(Type returnType, object value) : ParsingFailed(returnType,
		//ncrunch: no coverage start
		0, value switch
		{
			null => "null",
			Expression => "Expression " + value + " needs to be evaluated!",
			//ncrunch: no coverage end
			_ => value + ""
		} + " (" + value?.GetType() + ") for " + returnType.Name);

	public ValueInstance(string text)
	{
		value = text;
		number = TextId;
	}

	public ValueInstance(Type returnType, Dictionary<ValueInstance, ValueInstance> dictionary)
	{
		value = new ValueDictionaryInstance(returnType, dictionary);
		number = DictionaryId;
	}

	public ValueInstance(Type returnType, ValueInstance[] values)
	{
		if (returnType.IsList)
		{
			value = new ValueListInstance(returnType, values);
			number = ListId;
			return;
		}
		if (!returnType.IsMutable && (returnType.IsNumber || returnType.IsText ||
			returnType.IsCharacter || returnType.IsDictionary ||
			returnType.IsEnum || returnType.IsBoolean || returnType.IsNone))
			throw new ValueTypeInstanceShouldOnlyBeCreatedForComplexTypes(returnType);
		value = new ValueTypeInstance(returnType, values);
		number = TypeId;
	}

	public class ValueTypeInstanceShouldOnlyBeCreatedForComplexTypes(Language.Type returnType)
		: Exception(returnType.ToString()) { }

	/// <summary>
	/// Used by ApplyMethodReturnTypeMutable and TryEvaluate to flip if this is a mutable or not.
	/// </summary>
	public ValueInstance(ValueInstance existingInstance, Type newType)
	{
		switch (existingInstance.number)
		{
		case TextId:
			var textTypeMembers = newType.Members;
			var textValues = new ValueInstance[textTypeMembers.Count];
			var textValue = new ValueInstance((string)existingInstance.value);
			if (newType.IsList)
			{
				var elementsMemberIndex = 0;
				for (var memberIndex = 0; memberIndex < textTypeMembers.Count; memberIndex++)
					if (textTypeMembers[memberIndex].Name == Type.ElementsLowercase)
					{
						elementsMemberIndex = memberIndex;
						break;
					}
				textValues[elementsMemberIndex] = textValue;
			}
			else
				textValues[0] = textValue;
			value = new ValueTypeInstance(newType, textValues);
			number = TypeId;
			break;
		case ListId:
			value = new ValueListInstance(newType, ((ValueListInstance)existingInstance.value).Items);
			number = ListId;
			break;
		case DictionaryId:
			value = new ValueDictionaryInstance(newType, ((ValueDictionaryInstance)existingInstance.value).Items);
			number = DictionaryId;
			break;
		case TypeId:
			var existingTypeInstance = (ValueTypeInstance)existingInstance.value;
			if (!newType.IsMutable && existingTypeInstance.ReturnType.IsMutable && newType.IsText)
			{
				value = existingTypeInstance.Values[0].value;
				number = TextId;
			}
			else
			{
				value = new ValueTypeInstance(newType, existingTypeInstance.Values);
				number = TypeId;
			}
			break;
		default:
			value = newType;
			number = existingInstance.number;
			break;
		}
	}

	public bool IsType(Type type) =>
		type is OneOfType oneOf
			? oneOf.Types.Any(IsType)
			: number switch
			{
				TextId => type.IsText,
				ListId => type == ((ValueListInstance)value).ReturnType,
				DictionaryId => type == ((ValueDictionaryInstance)value).ReturnType,
				TypeId => type == ((ValueTypeInstance)value).ReturnType,
				_ => IsPrimitiveType(type)
			};

	public bool IsPrimitiveType(Type noneBoolOrNumberType) => value == noneBoolOrNumberType;
	public bool HasValue => value != null;
	public bool IsText => number is TextId;
	public string Text => (string)value;
	public double Number => number;
	public bool Boolean => number != 0;
	public bool IsList => number is ListId;
	public ValueListInstance List => (ValueListInstance)value;
	public bool IsDictionary => number is DictionaryId;

	public bool IsNumberLike(Type numberType) =>
		IsPrimitiveType(numberType) ||
		!IsText && !IsList && !IsDictionary && GetType().IsSameOrCanBeUsedAs(numberType);

	public double GetArithmeticNumber()
	{
		if (number != TypeId)
			return number;
		var typeInstance = (ValueTypeInstance)value;
		return typeInstance.Values.Length > 0
			? typeInstance.Values[0].number
			: 0;
	}

	public bool IsSameOrCanBeUsedAs(Type otherType) =>
		number switch
		{
			TextId => otherType.IsText || otherType.IsList &&
				otherType is GenericTypeImplementation { ImplementationTypes: [{ IsCharacter: true }] },
			ListId => ((ValueListInstance)value).ReturnType.IsSameOrCanBeUsedAs(otherType),
			DictionaryId => ((ValueDictionaryInstance)value).ReturnType.IsSameOrCanBeUsedAs(otherType),
			TypeId => ((ValueTypeInstance)value).ReturnType.IsSameOrCanBeUsedAs(otherType),
			_ => ((Type)value).IsSameOrCanBeUsedAs(otherType)
		};

	public bool IsValueTypeInstanceType =>
		number == TypeId && value is ValueTypeInstance { ReturnType.Name: nameof(Type) };

	public ValueTypeInstance? TryGetValueTypeInstance() =>
		number == TypeId
			? (ValueTypeInstance)value
			: null;

	/// <summary>
	/// Special code to make the ValueInstance mutable if the method return type requires it (rare)
	/// </summary>
	public ValueInstance ApplyMethodReturnTypeMutable(Type methodReturnType)
	{
		var isInstanceMutable = IsMutable;
		if (isInstanceMutable == methodReturnType.IsMutable)
			return this;
		if (!isInstanceMutable)
			return IsSameOrCanBeUsedAs(methodReturnType.GetFirstImplementation())
				? new ValueInstance(this, methodReturnType)
				: this;
		return GetType().GetFirstImplementation().IsSameOrCanBeUsedAs(methodReturnType)
			? new ValueInstance(this, methodReturnType)
			: this;
	}

	/// <summary>
	/// Gets the underlying type, except if it is a Text, that should always be checked before.
	/// </summary>
	public new Type GetType() =>
		number switch
		{
			ListId => ((ValueListInstance)value).ReturnType,
			DictionaryId => ((ValueDictionaryInstance)value).ReturnType,
			TypeId => ((ValueTypeInstance)value).ReturnType,
			_ => (Type)value
		};

	public int GetIteratorLength()
	{
		if (number == ListId)
			return ((ValueListInstance)value).Items.Count;
		if (number == TextId)
			return ((string)value).Length;
		if (number == DictionaryId)
			throw new IteratorNotSupported(this);
		if (number == TypeId)
		{
			var typeInstance = (ValueTypeInstance)value;
			if (typeInstance.ReturnType.IsList)
			{
				for (var i = 0; i < typeInstance.Values.Length; i++)
					if (typeInstance.Values[i].IsText)
						return typeInstance.Values[i].Text.Length;
			} //ncrunch: no coverage
			if (typeInstance.TryGetValue("keysAndValues", out var elementsMember) && elementsMember.IsList)
				return elementsMember.List.Items.Count;
			throw new IteratorNotSupported(this);
		}
		return (int)number;
	}

	public Type GetIteratorType() => ((ValueListInstance)value).ReturnType.GetFirstImplementation();

	public ValueInstance GetIteratorValue(Type charTypeIfNeeded, int index)
	{
		var normalizedIndex = NormalizeIndexForIterator(index);
		return number switch
		{
			TextId => normalizedIndex >= 0 && normalizedIndex < ((string)value).Length
				? new ValueInstance(charTypeIfNeeded, ((string)value)[normalizedIndex])
				: new ValueInstance(charTypeIfNeeded, '\0'),
			ListId => ((ValueListInstance)value).Items[normalizedIndex],
			TypeId when ((ValueTypeInstance)value).ReturnType.IsList &&
				FindTextInValues((ValueTypeInstance)value, out var wrappedText) =>
				normalizedIndex >= 0 && normalizedIndex < wrappedText!.Length
					? new ValueInstance(charTypeIfNeeded, wrappedText[normalizedIndex])
					: new ValueInstance(charTypeIfNeeded, '\0'),
			TypeId when ((ValueTypeInstance)value).TryGetValue("elements", out var elementsMember) &&
				elementsMember.IsList => elementsMember.List.Items[normalizedIndex],
			_ => throw new IteratorNotSupported(this)
		};
	}

	private int NormalizeIndexForIterator(int index) =>
		index >= 0
			? index
			: GetIteratorLength() + index;

	private static bool FindTextInValues(ValueTypeInstance typeInstance, out string? text)
	{
		for (var i = 0; i < typeInstance.Values.Length; i++)
			if (typeInstance.Values[i].IsText)
			{
				text = typeInstance.Values[i].Text;
				return true;
			} //ncrunch: no coverage start
		text = null;
		return false;
	} //ncrunch: no coverage end

	public class IteratorNotSupported(ValueInstance instance) : Exception(instance.ToString());

	public Dictionary<ValueInstance, ValueInstance> GetDictionaryItems() =>
		((ValueDictionaryInstance)value).Items;

	public bool IsMutable =>
		number switch
		{
			TextId => false,
			ListId => ((ValueListInstance)value).ReturnType.IsMutable,
			DictionaryId => ((ValueDictionaryInstance)value).ReturnType.IsMutable,
			TypeId => ((ValueTypeInstance)value).ReturnType.IsMutable,
			_ => ((Type)value).IsMutable
		};
	public bool IsError => number == TypeId && ((ValueTypeInstance)value).ReturnType.IsError;
	public override string ToString() => GetTypeName() + ": " + ToExpressionCodeString(true);

	private string GetTypeName() =>
		number switch
		{
			TextId => Type.Text,
			ListId => ((ValueListInstance)value).ReturnType.Name,
			DictionaryId => ((ValueDictionaryInstance)value).ReturnType.Name,
			TypeId => ((ValueTypeInstance)value).ReturnType.Name,
			_ => ((Type)value).Name
		};

	public string ToExpressionCodeString(bool escapeText = false) =>
		number switch
		{
			TextId => escapeText
				? "\"" + EscapeText((string)value) + "\""
				: (string)value,
			ListId => BuildListString(((ValueListInstance)value).Items, escapeText),
			DictionaryId => BuildDictionaryString(((ValueDictionaryInstance)value).Items, escapeText),
			TypeId => ((ValueTypeInstance)value).ToAutomaticText(),
			_ => GetPrimitiveCodeString((Type)value)
		};

	private static string EscapeText(string text) =>
		text.Replace("\\", "\\\\", StringComparison.Ordinal).
			Replace("\n", "\\n", StringComparison.Ordinal).
			Replace("\r", "\\r", StringComparison.Ordinal).
			Replace("\t", "\\t", StringComparison.Ordinal).
			Replace("\"", "\\\"", StringComparison.Ordinal);

	private static string BuildListString(IReadOnlyList<ValueInstance> items, bool escapeText)
	{
		if (items.Count == 0)
			return "";
		if (items.Count == 1)
			return items[0].ToExpressionCodeString(escapeText);
		var parts = new string[items.Count];
		for (var i = 0; i < items.Count; i++)
			parts[i] = items[i].ToExpressionCodeString(escapeText);
		return parts.ToBrackets();
	}

	private static string BuildDictionaryString(Dictionary<ValueInstance, ValueInstance> items,
		bool escapeText)
	{
		if (items.Count == 0)
			return "";
		var parts = new string[items.Count];
		var i = 0;
		foreach (var kv in items)
			parts[i++] = "(" + kv.Key.ToExpressionCodeString(escapeText) + ", " +
				kv.Value.ToExpressionCodeString(escapeText) + ")";
		return parts.ToBrackets();
	}

	private string GetPrimitiveCodeString(Type primitiveType)
	{
		if (primitiveType.IsBoolean)
			return number == 0
				? "false"
				: "true";
		if (primitiveType.IsNone)
			return Type.None;
		if (primitiveType.IsNumber)
			return GetCachedNumberString();
		if (primitiveType.IsCharacter)
			return GetCachedCharString();
		return primitiveType.IsMutable
			// ReSharper disable once TailRecursiveCall
			? GetPrimitiveCodeString(primitiveType.GetFirstImplementation())
			: throw new InvalidTypeValue(primitiveType, primitiveType.ToString());
	}

	public string GetCachedNumberString()
	{
		if (double.IsInteger(number))
		{
			var intValue = (int)number;
			if ((uint)intValue < (uint)CachedIntegerStrings.Length)
				return CachedIntegerStrings[intValue];
			if (intValue == -1)
				return "-1";
		}
		var absoluteValue = Math.Abs(number);
		return absoluteValue is >= 10_000_000 or > 0 and <= 1e-9
			? number.ToString("0.################e0", System.Globalization.CultureInfo.InvariantCulture)
			: number.ToString(System.Globalization.CultureInfo.InvariantCulture);
	}

	private static readonly string[] CachedIntegerStrings = CreateIntegerStringCache();

	private static string[] CreateIntegerStringCache()
	{
		var cache = new string[101];
		for (var i = 0; i < cache.Length; i++)
			cache[i] = i.ToString(System.Globalization.CultureInfo.InvariantCulture);
		return cache;
	}

	private string GetCachedCharString()
	{
		var c = (char)number;
		return c < 128
			? CachedAsciiCharStrings[c]
			: c.ToString();
	}

	private static readonly string[] CachedAsciiCharStrings = CreateAsciiCharCache();

	private static string[] CreateAsciiCharCache()
	{
		var cache = new string[128];
		for (var i = 0; i < cache.Length; i++)
			cache[i] = ((char)i).ToString();
		return cache;
	}

	/// <summary>
	/// Equals is a bit more complex as we need to handle all different kinds of ValueInstances we
	/// could have, either primary, text, list, dictionary, or type. Or any of those mixed.
	/// </summary>
	public bool Equals(ValueInstance other)
	{
		if (number == other.number && value == other.value)
			return true;
		ComplexEqualsCalls++;
		if (number == TypeId)
		{
			var instance = (ValueTypeInstance)value;
			if (other.number == TypeId)
				return instance.Equals((ValueTypeInstance)other.value);
			if (other.number != ListId && other.number != DictionaryId && other.number != TextId &&
				instance.TryGetValue("number", out var numberMember) &&
				numberMember.number == other.number)
				return true;
		}
		else if (other.number == TypeId && number != ListId && number != DictionaryId &&
			number != TextId &&
			((ValueTypeInstance)other.value).TryGetValue("number",
				out var otherNumberMember) && otherNumberMember.number == number)
			return true;
		if (number != other.number)
			return false;
		if (number == TextId)
			return (string)value == (string)other.value;
		if (number == ListId)
			return ((ValueListInstance)value).Equals((ValueListInstance)other.value);
		return number == DictionaryId
			? ((ValueDictionaryInstance)value).Equals((ValueDictionaryInstance)other.value)
			: ((Type)other.value).IsSameOrCanBeUsedAs((Type)value);
	}

	public static int ComplexEqualsCalls;
	public override int GetHashCode() => HashCode.Combine(number, value);
}