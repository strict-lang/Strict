using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict.Expressions;

public sealed class To(Expression left, Method operatorMethod, Type conversionType)
	: MethodCall(operatorMethod, left, [], conversionType)
{
	public Type ConversionType { get; } = conversionType;
	public override bool IsConstant => Instance!.IsConstant;
	public override string ToString() => $"{Instance} {Method.Name} {ConversionType.Name}";

	public static Expression Parse(Body body, ReadOnlySpan<char> text, Expression left)
	{
		var conversionType = body.ReturnType.FindType(text.ToString());
		if (conversionType == null)
			throw new ConversionTypeNotFound(body, text.ToString());
		var method = left.ReturnType.GetMethod(BinaryOperator.To, []);
		if (method.ReturnType.Name != conversionType.Name &&
			!left.ReturnType.IsUpcastable(conversionType) &&
			!left.ReturnType.IsSameOrCanBeUsedAs(conversionType))
			throw new ConversionTypeIsIncompatible(body,
				$"Conversion for {
					left.ReturnType.Name
				} to {
					conversionType.Name
				} does not exist and no member is compatible", conversionType);
		return new To(left, method, conversionType);
	}

	public sealed class ConversionTypeNotFound(Body body, string typeName)
		: ParsingFailed(body, typeName);

	public sealed class ConversionTypeIsIncompatible(Body body, string message, Type type)
		: ParsingFailed(body, message, type);
}