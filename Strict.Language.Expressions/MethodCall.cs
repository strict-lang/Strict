using System;
using System.Collections.Generic;
using System.Linq;

namespace Strict.Language.Expressions
{
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
				return GetMethodCall(new MemberCall(context.GetMember(methodName)), context.Type,
					methodName, GetArguments(context, parts, methodName));
			var memberParts = methodName.Split('.', 2);
			methodName = memberParts[1];
			var firstMember = context.GetMember(memberParts[0]);
			return GetMethodCall(new MemberCall(firstMember), firstMember.Type, methodName,
				GetArguments(context, parts, methodName));
		}

		private static Expression? GetMethodCall(Expression instance, Type context, string methodName,
			Expression[] arguments)
		{
			var method = context.Methods.FirstOrDefault(m => m.Name == methodName);
			return method == null
				? null
				: new MethodCall(instance, method, arguments);
		}

		private static Expression[] GetArguments(Method context, string[] parts, string methodName)
		{
			var arguments = new Expression[parts.Length - 1];
			for (var i = 0; i < parts.Length - 1; i++)
				arguments[i] = MethodExpressionParser.TryParse(context, parts[i + 1]) ??
					throw new MethodExpressionParser.UnknownExpression(context,
						parts[i] + " for " + methodName);
			return arguments;
		}
	}
}