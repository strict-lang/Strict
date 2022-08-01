namespace Strict.Language;

// ReSharper disable once HollowTypeName
public abstract class NamedType
{
	protected NamedType(Context definedIn, string nameAndType, Type? typeFromValue = null)
	{
		if (typeFromValue == null)
		{
			var parts = nameAndType.Split(' ');
			Name = parts[0];
			if (!Name.IsWord())
				throw new Context.NameMustBeAWordWithoutAnySpecialCharactersOrNumbers(Name);
			Type = definedIn.GetType(parts.Length == 1
				? Name.MakeFirstLetterUppercase()
				: parts[1]);
		}
		else
		{
			Name = nameAndType;
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