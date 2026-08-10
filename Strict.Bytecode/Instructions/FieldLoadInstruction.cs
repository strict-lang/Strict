using Strict.Bytecode.Serialization;

namespace Strict.Bytecode.Instructions;

/// <summary>
/// Loads a named field value from a ValueTypeInstance stored in <see cref="ObjectRegister"/>
/// into the output register. Used for struct field access (e.g. Language Type.Name) and
/// constructor-to-field-mutations optimizers.
/// </summary>
public sealed class FieldLoadInstruction : RegisterInstruction
{
	public FieldLoadInstruction(Register outRegister, Register objectRegister, string fieldName) :
		base(InstructionType.FieldLoad, outRegister)
	{
		ObjectRegister = objectRegister;
		FieldName = fieldName;
	}

	public FieldLoadInstruction(BinaryReader reader, NameTable table) : base(
		InstructionType.FieldLoad, (Register)reader.ReadByte())
	{
		ObjectRegister = (Register)reader.ReadByte();
		FieldName = table.names[reader.Read7BitEncodedInt()];
	}

	public Register ObjectRegister { get; }
	public string FieldName { get; }

	protected override void WritePayload(BinaryWriter writer, NameTable table)
	{
		base.WritePayload(writer, table);
		writer.Write((byte)ObjectRegister);
		writer.Write7BitEncodedInt(table[FieldName]);
	}

	public override string ToString() =>
		$"{InstructionType} {Register} <- {ObjectRegister}.{FieldName}";
}
