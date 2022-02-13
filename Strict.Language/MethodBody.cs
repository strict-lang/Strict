using System.Collections.Generic;

namespace Strict.Language;

/// <summary>
/// Every method body is just an expression, which might contain multiple expressions, which are
/// all executed and then the final result is returned (all previous expressions must succeed).
/// Method parameters are in this context and can be used by any of the expressions nested here.
/// </summary>
public class MethodBody : BlockExpression
{
	public MethodBody(Method method, IReadOnlyList<Expression> expressions) : base(
		method.ReturnType)
	{
		Method = method;
		Expressions = expressions;
	}

	public Method Method { get; }
	public IReadOnlyList<Expression> Expressions { get; }
}