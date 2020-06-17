using System.Collections.Generic;

namespace Strict.Language
{
	public class Type
	{
		public Type(string name, Implement? implement, IReadOnlyList<Member> has, IReadOnlyList<Method> methods)
		{
			Name = name;
			Implement = implement;
			Has = has;
			Methods = methods;
		}

		public string Name { get; }
		public Implement? Implement { get; }
		public IReadOnlyList<Member> Has { get; }
		public IReadOnlyList<Method> Methods { get; }
	}
}