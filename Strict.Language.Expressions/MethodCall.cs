using System;
using System.Collections.Generic;

namespace Strict.Language.Expressions;

/// <summary>
/// Any type of method we can call, this includes normal local method calls, recursions, calls to
/// any of our implement base types (instance is null in all of those cases), calls to other types
/// (either From(type) or instance method calls, there are no static methods) or any operator
/// <see cref="Binary"/> or <see cref="Not"/> unary call (which are all normal methods as well).
/// Like MemberCall has the same syntax when parent instance is used: Type.Method
/// </summary>
public class MethodCall : NonGenericExpression
{
	public MethodCall(Method method, Expression? instance, IReadOnlyList<Expression> arguments) :
		base(method.ReturnType)
	{
		Instance = instance;
		Method = method;
		Arguments = arguments;
	}

	public Method Method { get; }
	public Expression? Instance { get; }
	public IReadOnlyList<Expression> Arguments { get; }

	public MethodCall(Method method, Expression? instance = null) : this(method, instance,
		Array.Empty<Expression>()) { }

	public override string ToString() =>
		Instance != null
			? Method.Name == Method.From
				? $"{Instance}{Arguments.ToBrackets()}"
				: $"{Instance}.{Method.Name}{Arguments.ToBrackets()}"
			: Arguments.Count > 0
				? $"{Method.Name}{Arguments.ToBrackets()}"
				: Method.Name;
}