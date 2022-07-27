using System;

namespace Strict.Language.Expressions;

/// <summary>
/// Identifiers are variables added to the current context giving any expression access to it and
/// its value in addition to type <see cref="Member"/> and method <see cref="Parameter"/>.
/// </summary>
public class Identifier : NamedType
{
	public Identifier(string name, Type type) : base(name, type)
	{
		if (char.IsUpper(name[0]))
			throw new InvalidNameForIdentifier(name);
	}

	public class InvalidNameForIdentifier : Exception
	{
		public InvalidNameForIdentifier(string name) : base(name) { }
	}

	public override string ToString() => Name;
}