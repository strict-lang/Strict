using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict.Expressions;

public sealed class To(Expression left, Method operatorMethod, Type conversionType)
	: MethodCall(operatorMethod, left, [], conversionType)
{
	public Type ConversionType { get; } = conversionType;
	public override bool IsConstant => Instance!.IsConstant;

	public override string ToString() =>
		$"{AddNestedBracketsIfNeeded(Instance!)} {Method.Name} {ConversionType.ToCodeString()}";

	public static Expression Parse(Body body, ReadOnlySpan<char> text, Expression left)
	{
		var conversionType = body.ReturnType.TryGetType(text.ToString());
		if (conversionType == null)
			throw new ConversionTypeNotFound(body, text.ToString());
		var method = left.ReturnType.GetMethod(BinaryOperator.To, []);
		// Special case for lists, which automatically use the implementation type To method
		if (conversionType.IsList && left.ReturnType.IsList &&
			(method.IsTrait || method.ReturnType != conversionType))
		{
			method = FindConversionMethod(left.ReturnType.GetFirstImplementation(),
				conversionType.GetFirstImplementation());
			return method == null
				? throw new ConversionTypeNotFound(body, text.ToString())
				: new To(left, method, conversionType);
		}
		if (method.ReturnType.Name != conversionType.Name)
			method = FindConversionMethod(left.ReturnType, conversionType) ?? method;
		if (method.ReturnType.Name != conversionType.Name &&
			!left.ReturnType.IsUpcastable(conversionType) &&
			!left.ReturnType.IsSameOrCanBeUsedAs(conversionType))
			throw new ConversionTypeIsIncompatible(body,
				$"Conversion for {left.ReturnType.ToCodeString()} to {conversionType.ToCodeString()} does " +
				$"not exist, underlying list elements don't support it and no member is compatible (method " +
				$"returns {method.ReturnType.ToCodeString()}, token '{text.ToString()}')", conversionType);
		return new To(left, method, conversionType);
	}

	private static Method? FindConversionMethod(Type type, Type conversionType) =>
		type.AvailableMethods.TryGetValue(BinaryOperator.To, out var methods)
			? methods.FirstOrDefault(m => m.ReturnType.Name == conversionType.Name ||
				m.ReturnType.IsSameOrCanBeUsedAs(conversionType))
			: null;

	/// <summary>
	/// Creates a To expression converting left to Text, using the type's own to Text method or
	/// the inherited Any.to Text fallback.
	/// </summary>
	public static To ConvertToText(Expression left, Type textType)
	{
		var method = left.ReturnType.GetMethod(BinaryOperator.To, []);
		if (method.ReturnType.Name != textType.Name)
			method = FindConversionMethod(left.ReturnType, textType) ?? method;
		return new To(left, method, textType);
	}

	public sealed class ConversionTypeNotFound(Body body, string typeName)
		: ParsingFailed(body, typeName);

	public sealed class ConversionTypeIsIncompatible(Body body, string message, Type type)
		: ParsingFailed(body, message, type);

	//ncrunch: no coverage start
	public override bool Equals(Expression? other) =>
		ReferenceEquals(this, other) || other is To to &&
		Method.IsSameMethodNameReturnTypeAndParameters(to.Method) && Equals(Instance, to.Instance) &&
		ConversionType == to.ConversionType;

	public override int GetHashCode() => Method.GetHashCode() ^ ConversionType.GetHashCode();
}