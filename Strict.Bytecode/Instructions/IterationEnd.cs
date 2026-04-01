using Strict.Bytecode.Serialization;

namespace Strict.Bytecode.Instructions;

public sealed class LoopEndInstruction(int steps) : Instruction(InstructionType.LoopEnd)
{
	public LoopEndInstruction(BinaryReader reader) : this(reader.Read7BitEncodedInt()) { }
	public int Steps { get; } = steps;
	public LoopBeginInstruction? Begin { get; set; }
	/// <summary>
	/// Cached index of the Begin instruction in the current instruction list, -1 means not cached.
	/// Reset to -1 whenever Begin changes.
	/// </summary>
	public int BeginIndex { get; set; } = -1;

	public override void Write(BinaryWriter writer, NameTable table)
	{
		base.Write(writer, table);
		writer.Write7BitEncodedInt(Steps);
	}
}