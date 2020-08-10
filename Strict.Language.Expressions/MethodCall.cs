using System.Collections.Generic;

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
			//TODO: (Instance as Identifier)?.Type == Method.Parent ? "" :
			Instance + "." + Method.Name + Arguments.ToBracketsString();
	}
}