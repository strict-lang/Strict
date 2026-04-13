using Strict.Bytecode.Serialization;

namespace Strict.Bytecode.Instructions;

public sealed class JumpToId(int id, InstructionType instructionType)
	: Instruction(instructionType)
{
	public JumpToId(BinaryReader reader, InstructionType instructionType)
		: this(reader.Read7BitEncodedInt(), instructionType) { }

	public int Id { get; } = id;

	protected override void WritePayload(BinaryWriter writer, NameTable table) =>
		writer.Write7BitEncodedInt(Id);
}