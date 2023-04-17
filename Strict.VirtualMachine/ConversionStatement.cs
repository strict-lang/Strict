using Type = Strict.Language.Type;

namespace Strict.VirtualMachine;

public sealed class ConversionStatement : RegisterStatement
{
	public ConversionStatement(Register storedValueRegister, Register registerToStoreConversion,
		Type conversionType, Instruction instruction) : base(storedValueRegister, instruction)
	{
		RegisterToStoreConversion = registerToStoreConversion;
		ConversionType = conversionType;
	}

	public Type ConversionType { get; }
	public Register RegisterToStoreConversion { get; }
}