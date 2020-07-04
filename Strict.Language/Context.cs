using System;
using System.Collections.Generic;
using System.Text.RegularExpressions;

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
			// ReSharper disable once ConditionIsAlwaysTrueOrFalse
			if (parent != null)
				parent.children.Add(this);
			Name = name;
		}

		public class NameMustBeAWordWithoutAnySpecialCharactersOrNumbers : Exception
		{
			public NameMustBeAWordWithoutAnySpecialCharactersOrNumbers(string name) : base(name) { }
		}

		public Context Parent { get; }
		public IReadOnlyList<Context> Children => children;
		private readonly List<Context> children = new List<Context>();
		public string Name { get; }

		public Type GetType(string name)
		{
			// Generics still need to be supported (see Log.strict for Output<text>)
			if (name.StartsWith("Iterator<"))
				name = name.Split('<', '>')[1];
			if (name.Contains("<"))
				name = name.Split('<')[0];
			// Arrays are also not supported yet, simply return base type
			if (name.EndsWith('s'))
				name = name.Substring(0, name.Length - 1);
			if (name == Name || name == ToString())
				return (Type)this;
			var type = FindType(name);
			if (type == null)
				throw new TypeNotFound(name, ToString());
			return type;
		}

		public override string ToString() =>
			(string.IsNullOrEmpty(Parent.Name)
				? ""
				: Parent + ".") + Name;

		public class TypeNotFound : Exception
		{
			public TypeNotFound(string typeName, string contextFullName) : base(typeName +
				" not found in " + contextFullName) { }
		}

		public abstract Type? FindType(string name, Package? searchingFromPackage = null, Type? searchingFromType = null);
	}
}