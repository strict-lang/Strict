using Strict.Bytecode.Serialization;
using StrictType = Strict.Language.Type;

namespace Strict.Bytecode.Instructions;

/// <summary>
/// Creates a new value-type instance (struct) of <see cref="ReturnType"/> by reading one value per
/// field from <see cref="FieldRegisters"/> in declaration order. This replaces an Invoke of the
/// From-constructor to avoid method-dispatch overhead in hot loops after inlining.
/// Holds the actual Type reference to avoid a name-lookup at execution time.
/// </summary>
public sealed class ConstructValueTypeInstruction(Register outRegister,
	StrictType returnType,
	Register[] fieldRegisters) : RegisterInstruction(InstructionType.ConstructValueType, outRegister)
{
	public ConstructValueTypeInstruction(BinaryReader reader, NameTable table, BinaryExecutable binary)
		: this((Register)reader.ReadByte(),
			BinaryExecutable.EnsureResolvedType(binary.basePackage, table.names[reader.Read7BitEncodedInt()]),
			new Register[reader.Read7BitEncodedInt()])
	{
		for (var index = 0; index < FieldRegisters.Length; index++)
			FieldRegisters[index] = (Register)reader.ReadByte();
	}

	protected override void WritePayload(BinaryWriter writer, NameTable table)
	{
		base.WritePayload(writer, table);
		writer.Write7BitEncodedInt(table[ReturnType.FullName]);
		writer.Write7BitEncodedInt(FieldRegisters.Length);
		foreach (var register in FieldRegisters)
			writer.Write((byte)register);
	}

	public StrictType ReturnType { get; } = returnType;
	public Register[] FieldRegisters { get; } = fieldRegisters;

	public override string ToString() =>
		$"{InstructionType} {Register} = {ReturnType.Name}({string.Join(", ", FieldRegisters)})";
}