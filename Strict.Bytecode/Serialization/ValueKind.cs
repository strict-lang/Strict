namespace Strict.Bytecode.Serialization;

internal enum ValueKind : byte
{
	None,
	/// <summary>
	/// 0–255 stored as 1 byte; Number type is implied
	/// </summary>
	SmallNumber,
	/// <summary>
	/// Any 32-bit signed integer value, no floating point.
	/// </summary>
	IntegerNumber,
	/// <summary>
	/// Any other number is stored as a 64-bit double floating point number (default in Strict)
	/// </summary>
	Number,
	Text,
	Boolean,
	List,
	Character,
	Name,
	Dictionary
}