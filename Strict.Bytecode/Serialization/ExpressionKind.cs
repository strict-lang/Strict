namespace Strict.Bytecode.Serialization;

internal enum ExpressionKind : byte
{
	/// <summary>
	/// Store any small number as just 1 extra byte (only values 0–255 would work)
	/// </summary>
	SmallNumberValue,
	/// <summary>
	/// 4-byte integer number, second most common like int in c-type languages.
	/// </summary>
	IntegerNumberValue,
	/// <summary>
	/// 8-byte double floating point number for everything else
	/// </summary>
	NumberValue,
	/// <summary>
	/// Stored as a NameTable index
	/// </summary>
	TextValue,
	/// <summary>
	/// NameTable index + 1-byte bool
	/// </summary>
	BooleanValue,
	/// <summary>
	/// NameTable index (name) + NameTable index (type)
	/// </summary>
	VariableRef,
	/// <summary>
	/// NameTable index (name) + NameTable index (type) + optional instance
	/// </summary>
	MemberRef,
	/// <summary>
	/// NameTable index (op) + left + right
	/// </summary>
	BinaryExpr,
	/// <summary>
	/// NameTable indices + optional instance + args
	/// </summary>
	MethodCallExpr,
	/// <summary>
	/// NameTable index (list type) + item count + item expressions
	/// </summary>
	ListExpr,
	/// <summary>
	/// List expression + index expression + optional second index expression
	/// </summary>
	ListCallExpr
}