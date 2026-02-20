using Type = Strict.Language.Type;

namespace Strict.Runtime.Statements;

public sealed class Conversion(
	Register storedValueRegister,
	Register registerToStoreConversion,
	Type conversionType,
	Instruction instruction) : RegisterStatement(instruction, storedValueRegister)
{
	public Type ConversionType { get; } = conversionType;
	public Register RegisterToStoreConversion { get; } = registerToStoreConversion;
}
