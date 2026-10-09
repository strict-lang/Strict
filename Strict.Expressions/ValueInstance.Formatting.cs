using System.Globalization;
using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict.Expressions;

public readonly partial struct ValueInstance : IEquatable<ValueInstance>
{
	private static string BuildInvalidTypeValueMessage(Type returnType, object? value)
	{
		if (value is string text)
			return $"Cannot use runtime text '{text}' as {returnType}. " +
				$"This usually means code tried to read member data from missing or wrong instance.";
		return $"Cannot use runtime {DescribeStoredValueKind(value)} as {returnType}. " +
			$"Stored value={DescribeStoredValue(value)} ({value?.GetType()})";
	}

	private static string DescribeStoredValueKind(object? value) =>
		value switch
		{
			null => "null",
			Expression => "unevaluated expression",
			double valueDouble => valueDouble switch
			{
				TextId => "text marker",
				ListId => "list marker",
				DictionaryId => "dictionary marker",
				TypeId => "type marker",
				FlatNumericId => "flat numeric marker",
				_ => "number " + value
			},
			_ => value.GetType().Name
		};

	private static string DescribeStoredValue(object? value) =>
		value switch
		{
			null => "null",
			Expression expression => expression.ToString(),
			_ => value.ToString() ?? value.GetType().Name
		};

	public override string ToString() => GetTypeName() + ": " + ToExpressionCodeString(true);

	private string GetTypeName() =>
		/*TODO: IsPackedRgba
			? RgbaType.ReturnType.Name
			: */number switch
		{
			TextId => Type.Text,
			ListId => ((ValueArrayInstance)value).ReturnType.Name,
			DictionaryId => ((ValueDictionaryInstance)value).ReturnType.Name,
			TypeId => ((ValueTypeInstance)value).ReturnType.Name,
			FlatNumericId => ((ValueArrayInstance)value).ReturnType.Name,
			_ => ((Type)value).Name
		};

	public string ToExpressionCodeString(bool escapeText = false)
	{
		var generatedText = /*TODO: IsPackedRgba
			? ToPackedRgbaText()
			: */number switch
		{
			TextId => escapeText
				? "\"" + EscapeText((string)value) + "\""
				: (string)value,
			ListId => BuildListString((ValueArrayInstance)value, escapeText),
			DictionaryId => BuildDictionaryString(((ValueDictionaryInstance)value).Items, escapeText),
			TypeId => ((ValueTypeInstance)value).ToAutomaticText(),
			FlatNumericId => ((ValueArrayInstance)value).MaterializeAsType().ToAutomaticText(),
			_ => GetPrimitiveCodeString((Type)value)
		};
#if DEBUG
		if (PerformanceLog.IsEnabled)
			PerformanceLog.Write("ValueInstance.ToExpressionCodeString",
				"input=" + DescribeValue(this) + ", escapeText=" + escapeText + ", generated=" +
				generatedText + ", callers=" + PerformanceLog.GetCallers(1));
#endif
		return generatedText;
	}

#if DEBUG
	private void LogCreated(string constructorName)
	{
		if (PerformanceLog.IsEnabled)
			PerformanceLog.Write("ValueInstance." + constructorName, "stored=" + DescribeValue(this));
	}

	private static string DescribeValues(IReadOnlyList<ValueInstance> values)
	{
		if (values.Count == 0)
			return "[]";
		var parts = new string[values.Count];
		for (var index = 0; index < values.Count; index++)
			parts[index] = DescribeValue(values[index]);
		return "[" + string.Join(", ", parts) + "]";
	}

	//TODO: only allow this stuff in debug mode
	private static string DescribeValue(ValueInstance instance) =>
		/*TODO: instance.IsPackedRgba
			? "PackedRgba(type=" + instance.RgbaType.ReturnType.Name + ")"
			: */instance.number switch
		{
			TextId => "Text(" + instance.Text + ")",
			ListId => "List(type=" + instance.List.ReturnType.Name + ", count=" + instance.List.Count +
				")",
			DictionaryId => "Dictionary(type=" +
				((ValueDictionaryInstance)instance.value).ReturnType.Name + ", count=" +
				((ValueDictionaryInstance)instance.value).Items.Count + ")",
			TypeId => "TypeInstance(type=" + ((ValueTypeInstance)instance.value).ReturnType.Name +
				", members=" + ((ValueTypeInstance)instance.value).Values.Length + ")",
			FlatNumericId => "FlatNumeric(type=" + ((ValueArrayInstance)instance.value).ReturnType.Name +
				", width=" + ((ValueArrayInstance)instance.value).FlatWidth + ")",
			_ => ((Type)instance.value).IsBoolean
				? "Boolean(" + (instance.number != 0) + ")"
				: ((Type)instance.value).IsNumber
					? "Number(" + instance.number + ")"
					: ((Type)instance.value).Name
		};
#endif

	private static string EscapeText(string text) =>
		text.Replace("\\", @"\\", StringComparison.Ordinal).
			Replace("\n", "\\n", StringComparison.Ordinal).Replace("\r", "\\r", StringComparison.Ordinal).
			Replace("\t", "\\t", StringComparison.Ordinal).
			Replace("\"", "\\\"", StringComparison.Ordinal);

	private static string BuildListString(ValueArrayInstance list, bool escapeText)
	{
		if (list.Count == 0)
			return "";
		if (list.Count == 1)
			return list[0].ToExpressionCodeString(escapeText);
		const int MaxItems = 10;
		var itemsToAdd = Math.Min(list.Count, MaxItems);
		var parts = new string[itemsToAdd];
		for (var itemIndex = 0; itemIndex < itemsToAdd; itemIndex++)
			parts[itemIndex] = list[itemIndex].ToExpressionCodeString(escapeText);
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
			: throw new InvalidTypeValue(primitiveType, number);
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
			? number.ToString("0.################e0", CultureInfo.InvariantCulture)
			: number.ToString(CultureInfo.InvariantCulture);
	}

	private static readonly string[] CachedIntegerStrings = CreateIntegerStringCache();

	private static string[] CreateIntegerStringCache()
	{
		var cache = new string[101];
		for (var i = 0; i < cache.Length; i++)
			cache[i] = i.ToString(CultureInfo.InvariantCulture);
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
}
