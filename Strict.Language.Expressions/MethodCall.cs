﻿using System.Collections.Generic;
using Strict.Tokens;

namespace Strict.Language.Expressions
{
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
			Instance + Token.Dot.Name + Method.Name + Arguments.ToBracketsString();
	}
}