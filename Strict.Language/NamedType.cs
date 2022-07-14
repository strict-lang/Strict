namespace Strict.Language;

// ReSharper disable once HollowTypeName
public abstract class NamedType
{
	protected NamedType(Context definedIn, string nameAndType)
	{
		var parts = nameAndType.Split(' ');
		Name = parts[0];
		if (!Name.IsWord())
			throw new Context.NameMustBeAWordWithoutAnySpecialCharactersOrNumbers(Name);
		Type = definedIn.GetType(parts.Length == 1
			? Name.MakeFirstLetterUppercase()
			: parts[1]);
	}

	protected NamedType(string name, Type type)
	{
		Name = name;
		if (!Name.IsWord())
			throw new Context.NameMustBeAWordWithoutAnySpecialCharactersOrNumbers(Name);
		Type = type;
	}

	public string Name { get; }
	public Type Type { get; }
	public override bool Equals(object? obj) => obj is NamedType other && Name == other.Name;
	public override int GetHashCode() => Name.GetHashCode();
	public override string ToString() => Name + " " + Type;
}