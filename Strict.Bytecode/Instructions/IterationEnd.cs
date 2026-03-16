using Strict.Bytecode.Serialization;

namespace Strict.Bytecode.Instructions;

public sealed class LoopEndInstruction(int steps) : Instruction(InstructionType.LoopEnd)
{
	public LoopEndInstruction(BinaryReader reader) : this(reader.Read7BitEncodedInt()) { }
	public int Steps { get; } = steps;
	public LoopBeginInstruction? Begin { get; set; }

	public override void Write(BinaryWriter writer, NameTable table)
	{
		base.Write(writer, table);
		writer.Write7BitEncodedInt(Steps);
	}
}