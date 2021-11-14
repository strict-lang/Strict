namespace Strict.Language.Expressions;

/// <summary>
/// Identifiers are added to the current context giving any expression access to it and its value
/// in addition to type <see cref="Member"/> and method <see cref="Parameter"/>.
/// </summary>
public class Identifier : NamedType
{
	public Identifier(string name, Type type) : base(name, type) { }

	//still needed? public static bool IsIdentifier(string input) => input.IsWord() && char.IsLower(input[0]);
	public override bool Equals(object? obj) => obj is NamedType other && Name == other.Name;

	public override int GetHashCode() => Name.GetHashCode();
	public override string ToString() => Name;
}