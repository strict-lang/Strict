using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict.Expressions;

/// <summary>
/// A value accepted only via its to or the target from method is converted where it is stored or
/// passed, typed lists convert their elements. Still printed as the original value.
/// </summary>
public sealed class Conversion : MethodCall
{
	private Conversion(Method method, Expression value, Type targetType) : base(method,
		method.Name == Method.From
			? null
			: value, method.Name == Method.From
			? [value]
			: [], targetType, value.LineNumber) =>
		Value = value;

	public Expression Value { get; }

	public static Expression ConvertIfNeeded(Expression value, Type targetType)
	{
		if (targetType.IsMutable)
			targetType = targetType.GetFirstImplementation();
		if (targetType.IsGeneric || value.ReturnType.IsError)
			return value;
		if (value is List list &&
			targetType is GenericTypeImplementation { Generic.IsList: true } listType)
			return ConvertAll(list.Values, _ => listType.ImplementationTypes[0]) is { } elements
				? list.WithValues(listType, elements)
				: list;
		if (value.ReturnType.IsSameOrCanBeUsedAs(targetType))
			return value;
		var method = value.ReturnType.FindConversionMethod(targetType);
		return method == null
			? value
			: new Conversion(method, value, targetType);
	}

	/// <summary>
	/// Null if no value needed a conversion, the target type of a value without one is null.
	/// </summary>
	internal static List<Expression>? ConvertAll(IReadOnlyList<Expression> values,
		Func<int, Type?> targetTypeAt)
	{
		List<Expression>? converted = null;
		for (var index = 0; index < values.Count; index++)
			if (targetTypeAt(index) is { } targetType &&
				ConvertIfNeeded(values[index], targetType) is var value &&
				!ReferenceEquals(value, values[index]))
				(converted ??= values.ToList())[index] = value;
		return converted;
	}

	public override string ToString() => Value.ToString();
}
