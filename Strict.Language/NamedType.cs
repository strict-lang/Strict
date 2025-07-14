namespace Strict.Language;

public abstract class NamedType
{
	protected NamedType(Context definedIn, ReadOnlySpan<char> nameAndType,
		Type? typeFromValue = null)
	{
		if (typeFromValue == null)
		{
			var parts = nameAndType.Split();
			parts.MoveNext();
			Name = parts.Current.ToString();
			if (!Name.IsWord())
				throw new Context.NameMustBeAWordWithoutAnySpecialCharactersOrNumbers(Name);
			if (Name.IsKeyword())
				throw new CannotUseKeywordsAsName(Name);
			Type = definedIn.GetType(parts.MoveNext()
				? nameAndType[(Name.Length + 1)..].ToString()
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
		if (!Name.Length.IsWithinLimit())
			throw new NameLengthIsNotWithinTheAllowedLimit(Name);
	}

	public sealed class CannotUseKeywordsAsName(string name) : Exception(name +
		" is a keyword and cannot be used as a identifier name. Keywords List: " +
		Keyword.GetAllKeywords.ToWordList());

	/// <summary>
	/// Most things should NOT be mutable, this is mostly for optimizations like going through for loops
	/// </summary>
	public bool IsMutable { get; protected init; }
	/// <summary>
	/// While members of types are usually not mutable and cannot be reassigned, it doesn't make them
	/// pure constant values. When a type is created, non-constant members can be injected or created in
	/// the from-constructor. Constant members are just there (like static/const in other languages).
	/// </summary>
	public bool IsConstant { get; protected init; }

	//TODO: remove if unused, or add test
	public sealed class ListPrefixIsNotAllowedUseImplementationTypeNameInPlural(string typeName)
		: Exception($"List should not be used as prefix for {
			typeName
		} instead use {
			typeName.GetTextInsideBrackets()
		}s");

	public sealed class AssignmentWithInitializerTypeShouldNotHaveNameWithType(string name)
		: Exception(name);

	public sealed class NameLengthIsNotWithinTheAllowedLimit(string name) : Exception($"Name {
		name
	} length is {
		name.Length
	} but allowed limit is between {
		Limit.NameMinLimit
	} and {
		Limit.NameMaxLimit
	}");

	public string Name { get; }
	public Type Type { get; protected set; }
	public override bool Equals(object? obj) => obj is NamedType other && Name == other.Name;
	public override int GetHashCode() => Name.GetHashCode();
	public override string ToString() => Name + " " + Type;
}