using System;
using System.Collections.Generic;

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
			Parent = parent;
			// ReSharper disable once ConditionIsAlwaysTrueOrFalse
			if (parent != null)
				parent.children.Add(this);
			Name = name;
		}

		public Context Parent { get; }
		public IReadOnlyList<Context> Children => children;
		private readonly List<Context> children = new List<Context>();
		public string Name { get; }

		public Type GetType(string name)
		{
			if (name == Name || name == FullName)
				return (Type)this;
			var type = FindType(name);
			if (type == null)
				throw new TypeNotFound(name, FullName);
			return type;
		}

		public string FullName =>
			(string.IsNullOrEmpty(Parent.Name)
				? ""
				: Parent.FullName + ".") + Name;

		public class TypeNotFound : Exception
		{
			public TypeNotFound(string typeName, string contextFullName) : base(typeName +
				" not found in " + contextFullName) { }
		}

		public abstract Type? FindType(string name, Type? searchingFromType = null);
	}
}