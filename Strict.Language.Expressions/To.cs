using System;

namespace Strict.Language.Expressions;

public sealed class To : MethodCall
{
	public To(Expression left, Method operatorMethod, Type conversionType) : base(operatorMethod,
		left) =>
		ConversionType = conversionType;

	public Type ConversionType { get; }
	public override string ToString() => $"{Instance} {Method.Name} {ConversionType.Name}";

	public static Expression Parse(Body body, ReadOnlySpan<char> text, Expression left)
	{
		var conversionType = body.ReturnType.FindType(text.ToString());
		if (conversionType == null)
			throw new ConversionTypeNotFound(body);
		var method = left.ReturnType.GetMethod(BinaryOperator.To, Array.Empty<Expression>());
		if (method.ReturnType != conversionType)
			throw new NotImplemented(body);
		return new To(left, left.ReturnType.GetMethod(BinaryOperator.To, Array.Empty<Expression>()), conversionType);
	}

	public sealed class ConversionTypeNotFound : ParsingFailed
	{
		public ConversionTypeNotFound(Body body) : base(body) { }
	}

	public sealed class NotImplemented : ParsingFailed
	{
		public NotImplemented(Body body) : base(body) { }
	}
}