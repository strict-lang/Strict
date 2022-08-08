using System;

namespace Strict.Language;

// ReSharper disable once HollowTypeName
public abstract class NamedType
{
	protected NamedType(Context definedIn, ReadOnlySpan<char> nameAndType, Type? typeFromValue = null)
	{
		if (typeFromValue == null)
		{
			var parts = nameAndType.Split();
			parts.MoveNext();
			Name = parts.Current.ToString();
			if (!Name.IsWord())
				throw new Context.NameMustBeAWordWithoutAnySpecialCharactersOrNumbers(Name);
			Type = definedIn.GetType(parts.MoveNext()
				? parts.Current.ToString()
				: Name.MakeFirstLetterUppercase());
		}
		else
		{
			Name = nameAndType.ToString();
			if (!Name.IsWord())
				throw new Context.NameMustBeAWordWithoutAnySpecialCharactersOrNumbers(Name);
			Type = typeFromValue;
		}
	}

	public string Name { get; }
	public Type Type { get; }
	public override bool Equals(object? obj) => obj is NamedType other && Name == other.Name;
	public override int GetHashCode() => Name.GetHashCode();
	public override string ToString() => Name + " " + Type;
}