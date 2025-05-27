namespace Strict.Language.Expressions;

public sealed class To(Expression left, Method operatorMethod, Type conversionType)
	: MethodCall(operatorMethod, left, conversionType)
{
	public Type ConversionType { get; } = conversionType;
	public override string ToString() => $"{Instance} {Method.Name} {ConversionType.Name}";

	public static Expression Parse(Body body, ReadOnlySpan<char> text, Expression left)
	{
		var conversionType = body.ReturnType.FindType(text.ToString());
		if (conversionType == null)
			throw new ConversionTypeNotFound(body);
		var method = left.ReturnType.GetMethod(BinaryOperator.To, [], body.Method.Parser);
		if (method.ReturnType.Name != conversionType.Name && !left.ReturnType.IsUpcastable(conversionType))
			throw new ConversionTypeIsIncompatible(body,
				$"Conversion for {left.ReturnType.Name} and {conversionType.Name} does not exist",
				conversionType);
		return new To(left, method,
			conversionType);
	}

	public sealed class ConversionTypeNotFound(Body body) : ParsingFailed(body);

	public sealed class ConversionTypeIsIncompatible(Body body, string message, Type type)
		: ParsingFailed(body, message, type);
}