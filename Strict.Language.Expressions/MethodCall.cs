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
public class MethodCall : ConcreteExpression
{
	public MethodCall(Method method, Expression? instance, IReadOnlyList<Expression> arguments) :
		base(method.ReturnType)
	{
		if (method.Name == Method.From && instance != null)
			throw new NotSupportedException("Makes no sense, we don't have an instance yet"); //ncrunch: no coverage
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
			? CheckNestedMethodCallAndPrint(Instance)
			: $"{GetProperMethodName()}{PrintArguments(Arguments)}";

	private string CheckNestedMethodCallAndPrint(Expression instance)
	{
		var expressionText = instance.ToString();
		return IsNestedMethodCall(expressionText)
			? $"{expressionText.Replace(' ', '(') + ")"}.{Method.Name}{Arguments.ToBrackets()}"
			: $"{instance}.{Method.Name}{PrintArguments(Arguments)}";
	}

	private static bool IsNestedMethodCall(ReadOnlySpan<char> input) =>
		input.Contains(' ') && !input.Contains('(');

	private static string PrintArguments(IReadOnlyList<Expression> arguments)
	{
		if (arguments.Count != 1)
			return arguments.ToBrackets();
		var argumentText = arguments[0].ToString();
		return !argumentText.Contains(' ')
			? " " + argumentText
			: !argumentText.Contains('?') && !argumentText.Contains('(')
				? "(" + argumentText.Replace(' ', '(') + "))"
				: arguments.ToBrackets();
	}

	private string GetProperMethodName() =>
		Method.Name == Method.From
			? Method.ReturnType.Name
			: Method.Name;
}