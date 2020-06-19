using System.Collections.Generic;

namespace Strict.Language
{
	public class Method
	{
		public Method(string name, IReadOnlyList<Parameter> parameters, Type returnType)
		{
			Name = name;
			Parameters = parameters;
			ReturnType = returnType;
		}

		public string Name { get; }
		public IReadOnlyList<Parameter> Parameters { get; }
		public Type ReturnType { get; }
	}
}