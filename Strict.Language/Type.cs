using System.Collections.Generic;

namespace Strict.Language
{
	
	//TODO: merge with old Strict Type class
	public class Type
	{
		public Type(string name, Implement? implement = null) : this(name, implement,
			new Member[0], new Method[0]) { }

		public Type(string name, Implement? implement, IReadOnlyList<Member> method,
			IReadOnlyList<Method> methods)
		{
			Name = name;
			Implement = implement;
			Method = method;
			Methods = methods;
		}

		public string Name { get; }
		public Implement? Implement { get; }
		public IReadOnlyList<Member> Method { get; }
		public IReadOnlyList<Method> Methods { get; }
		public static readonly Type Void = new Type("void");

		public override string ToString() =>
			Name + (Implement != null
				? " " + Implement
				: "");
	}
}