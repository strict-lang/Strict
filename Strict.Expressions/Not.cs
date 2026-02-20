using Strict.Language;

namespace Strict.Expressions;

/// <summary>
/// Not is the only unary expression that needs an extra space, e.g., 5 is not 6. When using
/// -number, it will fail to find this as an identifier and check if one of the supported unary
/// one-letter expressions was used and will do the minus operation directly there.
/// </summary>
public sealed class Not(Method method, Expression right)
	: MethodCall(method, right, [], null, right.LineNumber)
{
	public static Not Parse(Body body, ReadOnlySpan<char> input, Range methodRange)
	{
		var right = body.Method.ParseExpression(body, input[methodRange]);
		return new Not(right.ReturnType.GetMethod(UnaryOperator.Not, []), right);
	}

	public override string ToString() =>
		Instance is Binary binary
			? binary.Method.Name == BinaryOperator.Is
				? binary.Instance + " is not " + binary.Arguments[0]
				: binary.Method.Name == BinaryOperator.In
					? binary.Arguments[0] + " is not in " + binary.Instance
					: UnaryOperator.Not + " " + Instance!
			: UnaryOperator.Not + " " + Instance!;

	public override bool IsConstant => Instance!.IsConstant;
}