using Strict.Bytecode.Serialization;

namespace Strict.Bytecode.Instructions;

/// <summary>
/// Emits a line of text to standard output, optionally appending a value from a register.
/// Replaces the runtime-only Invoke(logger.Log) pattern so the assembly backend can emit printf.
/// </summary>
public sealed class PrintInstruction(string textPrefix, Register? valueRegister = null,
	bool valueIsText = false)
	: Instruction(InstructionType.Print)
{
	public PrintInstruction(BinaryReader reader, NameTable table)
		: this(table.names[reader.Read7BitEncodedInt()])
	{
		if (!reader.ReadBoolean())
			return;
		ValueRegister = (Register)reader.ReadByte();
		ValueIsText = reader.ReadBoolean();
	}

	public string TextPrefix { get; } = textPrefix;
	public Register? ValueRegister { get; private set; } = valueRegister;
	public bool ValueIsText { get; private set; } = valueIsText;

	public override string ToString() =>
		ValueRegister.HasValue
			? $"Print \"{
				TextPrefix
			}\" + {
				(ValueIsText
					? "text"
					: "number")
			} {
				ValueRegister.Value
			}"
			: $"Print \"{TextPrefix}\"";

	public override void Write(BinaryWriter writer, NameTable table)
	{
		base.Write(writer, table);
		writer.Write7BitEncodedInt(table[TextPrefix]);
		writer.Write(ValueRegister.HasValue);
		if (!ValueRegister.HasValue)
			return;
		writer.Write((byte)ValueRegister.Value);
		writer.Write(ValueIsText);
	}
}