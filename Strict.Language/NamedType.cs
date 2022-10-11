﻿using System;

namespace Strict.Language;

public abstract class NamedType
{
	protected NamedType(Context definedIn, ReadOnlySpan<char> nameAndType, Type? typeFromValue = null)
	{
		IsMutable = nameAndType.Contains("Mutable(", StringComparison.Ordinal);
		if (typeFromValue == null)
		{
			var parts = nameAndType.Split();
			parts.MoveNext();
			Name = parts.Current.ToString();
			if (!Name.IsWord())
				throw new Context.NameMustBeAWordWithoutAnySpecialCharactersOrNumbers(Name);
			Type = definedIn.GetType(parts.MoveNext()
				? GetTypeName(parts.Current.ToString())
				: Name.MakeFirstLetterUppercase());
		}
		else
		{
			Name = nameAndType.ToString();
			Type = typeFromValue;
			if (Name.Contains(' '))
				throw new AssignmentWithInitializerTypeShouldNotHaveNameWithType(Name);
			if (!Name.IsWord())
				throw new Context.NameMustBeAWordWithoutAnySpecialCharactersOrNumbers(Name);
		}
	}

	public bool IsMutable { get; }

	private string GetTypeName(string typeName) =>
		IsMutable
			? typeName[(typeName.IndexOf('(') + 1)..typeName.IndexOf(')')]
			: typeName;

	public sealed class AssignmentWithInitializerTypeShouldNotHaveNameWithType : Exception
	{
		public AssignmentWithInitializerTypeShouldNotHaveNameWithType(string name) : base(name) { }
	}

	public string Name { get; }
	public Type Type { get; protected set; }
	public override bool Equals(object? obj) => obj is NamedType other && Name == other.Name;
	public override int GetHashCode() => Name.GetHashCode();
	public override string ToString() => Name + " " + Type;
}