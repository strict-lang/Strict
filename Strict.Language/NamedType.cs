using System;

namespace Strict.Language;

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
				? GetTypeName(nameAndType[(Name.Length + 1)..].ToString())
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

	public bool IsMutable { get; protected init; }

	private static string GetTypeName(string typeName)
	{
		if (typeName.StartsWith("List(", StringComparison.Ordinal))
			throw new ListPrefixIsNotAllowedUseImplementationTypeNameInPlural(typeName);
		return typeName;
	}

	public sealed class ListPrefixIsNotAllowedUseImplementationTypeNameInPlural : Exception
	{
		public ListPrefixIsNotAllowedUseImplementationTypeNameInPlural(string typeName) : base($"List should not be used as prefix for {typeName} instead use {typeName.GetTextInsideBrackets()}s") { }
	}

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