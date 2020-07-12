using System;
using System.Runtime.CompilerServices;

namespace Strict.Language
{
	/// <summary>
	/// Keeps all known types for use, if in <see cref="Package"/> contains all known types
	/// and traits the context is inside a type, all members are available as
	/// well, in a method more scope information is available. The high level context knows it all.
	/// </summary>
	public abstract class Context
	{
		protected Context(Context parent, string name)
		{
			if (parent != null && (string.IsNullOrWhiteSpace(name) ||
				!(this is Method) && !name.IsWord()))
				throw new NameMustBeAWordWithoutAnySpecialCharactersOrNumbers(name);
			Parent = parent!;
			Name = name;
		}

		public class NameMustBeAWordWithoutAnySpecialCharactersOrNumbers : Exception
		{
			public NameMustBeAWordWithoutAnySpecialCharactersOrNumbers(string name) : base(name) { }
		}

		public Context Parent { get; }
		public string Name { get; }

		public Type GetType(string name)
		{
			// Generics still need to be supported (see Log.strict for Output<text>)
			if (name.StartsWith("Iterator<"))
				name = name.Split('<', '>')[1];
			if (name.Contains("<"))
				name = name.Split('<')[0];
			// Arrays are also not supported yet, simply return base type, however only if we do not find a name ending with s already and do proper array fun
			if (name.EndsWith('s'))
				name = name[..^1];
			if (name == Name)
				return (Type)this;
			var type = FindFullType(name) ?? FindType(name, this);
			// packages.strict.dev should be asked if there is any downloadable public package containing the missing type
			if (type == null)
				throw new TypeNotFound(name, ToString());
			return type;
		}

		private Type? FindFullType(string name) =>
			name.Contains(".")
				? name == ToString()
					? this as Type
					: GetPackage().FindFullType(name)
				: null;

		public override string ToString() =>
			(string.IsNullOrEmpty(Parent.Name)
				? ""
				: Parent + ".") + Name;
		
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public Package GetPackage() =>
			this is Package package
				? package
				: Parent.GetPackage();

		public class TypeNotFound : Exception
		{
			public TypeNotFound(string typeName, string contextFullName) : base(typeName +
				" not found in " + contextFullName) { }
		}

		public abstract Type? FindType(string name, Context? searchingFrom = null);
	}
}