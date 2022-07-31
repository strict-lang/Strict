using System;

namespace Strict.Language.Expressions;

/// <summary>
/// Not is the only unary expression that needs an extra space, e.g. 5 is not 6. When using -number
/// it will fail to find this as an identifier and check if one of the supported unary one letter
/// expression was used and will do the minus operation directly there.
/// </summary>
public sealed class Not : NoArgumentMethodCall
{
	public Not(Expression right) : base(
		right.ReturnType.FindMethod(UnaryOperator.Not) ??
		throw new Binary.NoMatchingOperatorFound(right.ReturnType, UnaryOperator.Not), right) { }

	public override string ToString() => UnaryOperator.Not + " " + Instance;

	public static Expression Parse(Method.Line line, ShuntingYard postfixTokens)
	{
		var right = line.Method.ParseExpression(line, postfixTokens.Output.Pop());
		var operatorText = line.Text.GetSpanFromRange(postfixTokens.Output.Pop());
		return operatorText.Equals(UnaryOperator.Not, StringComparison.Ordinal)
			? new Not(right)
			: throw new MethodExpressionParser.InvalidOperatorHere(line, operatorText.ToString());
	}
}