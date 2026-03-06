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

	public ValueInstance(Type returnType, IReadOnlyList<ValueInstance> list)
	{
		value = new ValueListInstance(returnType, list);
		number = ListId;
	}

	public ValueInstance(Type returnType, Dictionary<ValueInstance, ValueInstance> dictionary)
	{
		value = new ValueDictionaryInstance(returnType, dictionary);
		number = DictionaryId;
	}

	public ValueInstance(Type returnType, Dictionary<string, ValueInstance> members)
	{
		if (!returnType.IsMutable && (returnType.IsNumber || returnType.IsText ||
			returnType.IsCharacter || returnType.IsList || returnType.IsDictionary ||
			returnType.IsEnum || returnType.IsBoolean || returnType.IsNone))
			throw new ValueTypeInstanceShouldOnlyBeCreatedForComplexTypes(returnType); //TODO: need test
		value = new ValueTypeInstance(returnType, members);
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
			value = new ValueTypeInstance(newType, new Dictionary<string, ValueInstance>
			{
				{ Type.Text, new ValueInstance((string)existingInstance.value) }
			});
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
				value = existingTypeInstance.Members[Type.Text].value;
				number = TextId;
			}
			else
			{
				value = new ValueTypeInstance(newType, existingTypeInstance.Members);
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
		number switch
		{
			TextId => type.IsText,
			ListId => type == ((ValueListInstance)value).ReturnType,
			DictionaryId => type == ((ValueDictionaryInstance)value).ReturnType,
			TypeId => type == ((ValueTypeInstance)value).ReturnType,
			_ => IsPrimitiveType(type)
		};

	public bool IsPrimitiveType(Type noneBoolOrNumberType) => value == noneBoolOrNumberType;
	public bool IsText => number is TextId;
	public string Text => (string)value;
	public double Number => number;
	public bool Boolean => number != 0;
	public bool IsList => number is ListId;
	public ValueListInstance List => (ValueListInstance)value;
	public bool IsDictionary => number is DictionaryId;

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
		return GetTypeExceptText().GetFirstImplementation().IsSameOrCanBeUsedAs(methodReturnType)
			? new ValueInstance(this, methodReturnType)
			: this;
	}

	public Type GetTypeExceptText() =>
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
		//TODO: this is ugly, check if needed or can be simplified
		if (number == TypeId)
		{
			var typeInstance = (ValueTypeInstance)value;
			if (typeInstance.ReturnType.IsList && typeInstance.Members.TryGetValue(Type.Text, out var textMember))
				return textMember.Text.Length;
			if (typeInstance.Members.TryGetValue("keysAndValues", out var elementsMember) && elementsMember.IsList)
				return elementsMember.List.Items.Count;
			throw new IteratorNotSupported(this);
		}
		return (int)number;
	}

	public Type GetIteratorType() => ((ValueListInstance)value).ReturnType.GetFirstImplementation();

	public ValueInstance GetIteratorValue(Type charTypeIfNeeded, int index) =>
		number switch
		{
			TextId => new ValueInstance(charTypeIfNeeded, ((string)value)[index]),
			ListId => ((ValueListInstance)value).Items[index],
			//TODO: this is ugly, check if needed or can be simplified
			TypeId when ((ValueTypeInstance)value).ReturnType.IsList &&
				((ValueTypeInstance)value).Members.TryGetValue(Type.Text, out var textMember) =>
				new ValueInstance(charTypeIfNeeded, textMember.Text[index]),
			TypeId when ((ValueTypeInstance)value).Members.TryGetValue("elements", out var elementsMember) &&
				elementsMember.IsList => elementsMember.List.Items[index],
			_ => throw new IteratorNotSupported(this)
		};

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
			TypeId => ((ValueTypeInstance)value).ToString(),
			_ => GetPrimitiveCodeString((Type)value)
		};

	private static string EscapeText(string s) => s.Replace("\\", @"\\").Replace("\"", "\\\"");

	private static string BuildListString(IReadOnlyList<ValueInstance> items, bool escapeText)
	{
		if (items.Count == 0)
			return "";
		if (items.Count == 1)
			return items[0].ToExpressionCodeString(escapeText);
		var parts = new string[items.Count];
		for (var i = 0; i < items.Count; i++)
			parts[i] = items[i].ToExpressionCodeString(escapeText);
		return string.Join(", ", parts);
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
		return string.Join(", ", parts);
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
			return number.ToString(System.Globalization.CultureInfo.InvariantCulture);
		if (primitiveType.IsCharacter)
			return ((char)number).ToString();
		return primitiveType.IsMutable
			// ReSharper disable once TailRecursiveCall
			? GetPrimitiveCodeString(primitiveType.GetFirstImplementation())
			: throw new NotSupportedException(primitiveType.ToString());
	}

	/// <summary>
	/// Equals is a bit more complex as we need to handle all different kinds of ValueInstances we
	/// could have, either primary, text, list, dictionary, or type. Or any of those mixed.
	/// </summary>
	public bool Equals(ValueInstance other)
	{
		if (number == other.number && value == other.value)
			return true;
		if (number == TypeId)
		{
			var instance = (ValueTypeInstance)value;
			if (other.number == TypeId)
				return instance.Equals((ValueTypeInstance)other.value);
			if (other.number != ListId && other.number != DictionaryId && other.number != TextId &&
				instance.Members.TryGetValue("number", out var numberMember) &&
				numberMember.number == other.number)
				return true;
		}
		else if (other.number == TypeId && number != ListId && number != DictionaryId &&
			number != TextId &&
			((ValueTypeInstance)other.value).Members.TryGetValue("number",
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

	public override int GetHashCode() => HashCode.Combine(number, value);
}