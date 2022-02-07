using System;
using System.Collections.Generic;
using System.Linq;

namespace Strict.Language.Expressions;

// ReSharper disable once HollowTypeName
public class MethodCall : Expression
{
	public MethodCall(Expression instance, Method method, params Expression[] arguments) : base(
		method.ReturnType)
	{
		Instance = instance;
		Method = method;
		Arguments = arguments;
	}

	public Expression Instance { get; }
	public Method Method { get; }
	public IReadOnlyList<Expression> Arguments { get; }

	public override string ToString() =>
		Instance + "." + Method.Name + Arguments.ToBracketsString();

	public static Expression? TryParse(Method context, string input) =>
		input.EndsWith(')') && input.Contains('(')
			? TryParseMethod(context,
				input.Split(new[] { '(', ')' }, StringSplitOptions.RemoveEmptyEntries))
			: TryParseMethod(context, input);

	private static Expression? TryParseMethod(Method context, params string[] parts)
	{
		var methodName = parts[0];
		if (!methodName.Contains('.'))
			return TryParseMemberCallMethod(context, parts, methodName);
		var memberParts = methodName.Split('.', 2);
		methodName = memberParts[1];
		var firstMember = MemberCall.TryParse(context, memberParts[0]);
		if (firstMember == null)
			throw new MemberCall.MemberNotFound(memberParts[0], context.Type);
		return GetMethodCall(firstMember, firstMember.ReturnType, methodName,
			GetArguments(context, parts, methodName));
	}

	private static Expression? TryParseMemberCallMethod(Method context, IReadOnlyList<string> parts,
		string methodName)
	{
		var member = MemberCall.TryParse(context, methodName);
		return member == null
			? null
			: GetMethodCall(member, context.Type, methodName, GetArguments(context, parts, methodName));
	}

	private static Expression? GetMethodCall(Expression instance, Type context, string methodName,
		Expression[] arguments)
	{
		var method = context.Methods.FirstOrDefault(m => m.Name == methodName);
		return method == null
			? null
			: new MethodCall(instance, method, arguments);
	}

	private static Expression[] GetArguments(Method method, IReadOnlyList<string> parts,
		string methodName)
	{
		var arguments = new Expression[parts.Count - 1];
		for (var i = 0; i < parts.Count - 1; i++)
			arguments[i] = method.TryParse(parts[i + 1]) ??
				throw new MethodExpressionParser.UnknownExpression(method,
					parts[i + 1] + " for " + methodName);
		return arguments;
	}
}