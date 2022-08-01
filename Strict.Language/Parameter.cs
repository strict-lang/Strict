namespace Strict.Language;

public sealed class Parameter : NamedType
{
	public Parameter(Method definedIn, string nameAndType) : base(definedIn, nameAndType) { }
}