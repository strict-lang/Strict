using StrictType = Strict.Language.Type;

namespace Strict.Bytecode.Instructions;

/// <summary>
/// Creates a new value-type instance (struct) of <see cref="ReturnType"/> by reading one value per
/// field from <see cref="FieldRegisters"/> in declaration order. This replaces an Invoke of the
/// From-constructor to avoid method-dispatch overhead in hot loops after inlining.
/// Holds the actual Type reference to avoid a name-lookup at execution time.
/// </summary>
public sealed class ConstructValueTypeInstruction(Register outRegister, StrictType returnType,
	Register[] fieldRegisters)
	: RegisterInstruction(InstructionType.ConstructValueType, outRegister)
{
	public StrictType ReturnType { get; } = returnType;
	public Register[] FieldRegisters { get; } = fieldRegisters;

	public override string ToString() =>
		$"{InstructionType} {Register} = {ReturnType.Name}({string.Join(", ", FieldRegisters)})";
}