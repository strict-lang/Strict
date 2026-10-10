using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict.Expressions;

/// <summary>
/// A value accepted only via its to or the target from method is converted where it is stored or
/// passed, typed lists convert their elements. Still printed as the original value.
/// </summary>
public sealed class Conversion : MethodCall
{
	private Conversion(Method method, Expression value, Type targetType, For? elements = null) :
		base(method, method.Name == Method.From
			? null
			: value, method.Name == Method.From
			? [value]
			: [], targetType, value.LineNumber)
	{
		Value = value;
		Elements = elements;
	}

	public Expression Value { get; }
	/// <summary>
	/// A list variable converts each element, iterating Value with the element conversion as body.
	/// </summary>
	public For? Elements { get; }

	public static Expression ConvertIfNeeded(Body body, Expression value, Type targetType)
	{
		if (targetType.IsMutable)
			targetType = targetType.GetFirstImplementation();
		if (targetType.IsGeneric || value.ReturnType.IsError)
			return value;
		if (targetType is GenericTypeImplementation { Generic.IsList: true } listType)
			if (value is List list)
				return ConvertAll(body, list.Values, _ => listType.ImplementationTypes[0]) is { } elements
					? list.WithValues(listType, elements)
					: list;
			else if (TryConvertElements(body, value, listType) is { } converted)
				return converted;
		if (value.ReturnType.IsSameOrCanBeUsedAs(targetType))
			return value;
		var method = value.ReturnType.FindConversionMethod(targetType);
		return method == null
			? value
			: new Conversion(method, value, targetType);
	}

	private static Conversion? TryConvertElements(Body body, Expression value,
		GenericTypeImplementation listType)
	{
		var sourceType = value.ReturnType.IsMutable
			? value.ReturnType.GetFirstImplementation()
			: value.ReturnType;
		if (sourceType is not GenericTypeImplementation { Generic.IsList: true } sourceList)
			return null;
		var element = new Variable(Type.ValueLowercase, true,
			new Instance(sourceList.ImplementationTypes[0], value.LineNumber), body, true);
		return ConvertIfNeeded(body, new VariableCall(element, value.LineNumber),
			listType.ImplementationTypes[0]) is Conversion converted
			? new Conversion(listType.AvailableMethods[Method.From][0], value, listType,
				new For([], value, converted, value.LineNumber))
			: null;
	}

	/// <summary>
	/// Null if no value needed a conversion, the target type of a value without one is null.
	/// </summary>
	internal static List<Expression>? ConvertAll(Body body, IReadOnlyList<Expression> values,
		Func<int, Type?> targetTypeAt)
	{
		List<Expression>? converted = null;
		for (var index = 0; index < values.Count; index++)
			if (targetTypeAt(index) is { } targetType &&
				ConvertIfNeeded(body, values[index], targetType) is var value &&
				!ReferenceEquals(value, values[index]))
				(converted ??= values.ToList())[index] = value;
		return converted;
	}

	public override string ToString() => Value.ToString();
}
