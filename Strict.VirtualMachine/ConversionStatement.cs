using Type = Strict.Language.Type;

namespace Strict.VirtualMachine;

public sealed class ConversionStatement(Register storedValueRegister,
	Register registerToStoreConversion, Type conversionType, Instruction instruction) :
	RegisterStatement(storedValueRegister, instruction)
{
	public Type ConversionType { get; } = conversionType;
	public Register RegisterToStoreConversion { get; } = registerToStoreConversion;
}