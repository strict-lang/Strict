using System;
using System.Collections.Generic;
using System.Linq;

namespace Strict.Language.Expressions
{
	// ReSharper disable once HollowTypeName
	public class MethodCall : Expression
	{
		public MethodCall(Expression instance, Method method, params Expression[] arguments)
			: base(method.ReturnType)
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
				? TryParseMethod(context, input.Split('(', ')'))
				: TryParseMethod(context, input);

		private static Expression? TryParseMethod(Method context, params string[] parts)
		{
			var methodName = parts[0];
			if (methodName.Contains('.'))
				throw new NotSupportedException(methodName);
			var member = new MemberCall(context.FindMember(methodName));
			var method = context.Type.Methods.FirstOrDefault(m => m.Name == methodName);
			if (method == null)
				throw new MethodNotFound(context, member, methodName);
			var arguments = new Expression[parts.Length - 1];
			for (int i = 1; i < parts.Length; i++)
				arguments[i - 1] = MethodExpressionParser.TryParse(context, parts[i]) ??
					throw new InvalidExpression(parts[i], methodName);
			return new MethodCall(member, method, (Expression[])arguments);
		}

		private class MethodNotFound : Exception
		{
			public MethodNotFound(Method context, MemberCall member, string methodName) : base(
				methodName + " of " + member + " not found in " + context) { }
		}
	}
}