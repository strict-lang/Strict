namespace Strict.Language;

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