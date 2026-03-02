namespace Strict.Language;

/// <summary>
/// It is faster to check for common types via this value than comparing Types or Type.Names.
/// </summary>
public enum TypeKind : ushort
{
	None,
	Boolean,
	Number,
	Text,
	Character,
	List,
	Dictionary,
	Error,
	Enum,
	Iterator,
	Mutable,
	Any,
	Unknown
}