using System;
using System.Collections.Generic;
using System.Linq;

namespace Strict.Language.Expressions;

// ReSharper disable once HollowTypeName
public class MethodCall : Expression
{
	public MethodCall(Expression? instance, Method method, params Expression[] arguments) : base(
		method.ReturnType)
	{
		Instance = instance;
		Method = method;
		Arguments = arguments;
	}

	public Expression? Instance { get; }
	public Method Method { get; }
	public IReadOnlyList<Expression> Arguments { get; }

	public override string ToString() =>
		(Instance != null
			? Instance + "."
			: "") + Method.Name + Arguments.ToBrackets();

	public static Expression? TryParse(Method context, string input) =>
		input.EndsWith(')') && input.Contains('(')
			? TryParseMethod(context,
				input.Split(new[] { '(', ')' }, StringSplitOptions.RemoveEmptyEntries))
			: TryParseMethod(context, input);

	private static Expression? TryParseMethod(Method context, params string[] parts)
	{
		var methodName = parts[0];
		var arguments = parts.Length > 1
			? GetArguments(context, parts[1], methodName)
			: Array.Empty<Expression>();
		if (!methodName.Contains('.'))
			return FindMethodCall(null, context.Type, methodName, arguments);
		var memberParts = methodName.Split('.', 2);
		var firstMember = MemberCall.TryParse(context, memberParts[0]);
		if (firstMember == null)
			throw new MemberCall.MemberNotFound(memberParts[0], context.Type);
		return FindMethodCall(firstMember, firstMember.ReturnType, memberParts[1], arguments);
	}

	private static Expression[] GetArguments(Method method, string argumentsText, string methodName)
	{
		var parts = argumentsText.Split(", ");
		var arguments = new Expression[parts.Length];
		for (var index = 0; index < parts.Length; index++)
			arguments[index] = method.TryParse(parts[index]) ??
				throw new MethodExpressionParser.UnknownExpression(method,
					parts[index] + " for " + methodName + " argument " + index);
		return arguments;
	}

	// ReSharper disable once TooManyArguments
	private static MethodCall? FindMethodCall(Expression? instance, Type context, string methodName,
		Expression[] arguments)
	{
		if (!methodName.IsWord())
			return null;
		var method = context.Methods.FirstOrDefault(m => m.Name == methodName);
		return method == null
			? throw new MethodNotFound(methodName, context)
			: method.Parameters.Count != arguments.Length
				? throw new ArgumentsDoNotMatchMethodParameters(arguments, method)
				: new MethodCall(instance, method, arguments);
	}

	public sealed class MethodNotFound : Exception
	{
		public MethodNotFound(string methodName, Type type) : base(methodName + " in " + type) { }
	}

	public sealed class ArgumentsDoNotMatchMethodParameters : Exception
	{
		public ArgumentsDoNotMatchMethodParameters(Expression[] arguments, Method method) : base(
			(arguments.Length == 0
				? "No arguments does "
				: "Arguments: " + arguments.ToBrackets() + " do ") + "not match \"" + method.Type + "." +
			method.Name + "\" method parameters: " + method.Parameters.ToBrackets()) { }
	}
}