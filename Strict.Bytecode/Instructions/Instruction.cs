using Strict.Bytecode.Serialization;

namespace Strict.Bytecode.Instructions;

public abstract class Instruction(InstructionType instructionType)
{
	public InstructionType InstructionType { get; } = instructionType;
	public int SourceLine { get; set; }

	/// <summary>
	/// Used for tests to check the instructions generated with simple multiline instructions list.
	/// </summary>
	public override string ToString() => $"{InstructionType}";

	public void Write(BinaryWriter writer, NameTable table)
	{
		var prevSourceLine = 0;
		WriteCompressed(writer, table, ref prevSourceLine);
	}

	internal void WriteCompressed(BinaryWriter writer, NameTable table, ref int prevSourceLine)
	{
		if (SourceLine != 0 && SourceLine != prevSourceLine)
		{
			writer.Write((byte)((byte)InstructionType | (byte)InstructionType.IncludesSourceLine));
			writer.Write7BitEncodedInt(SourceLine);
			prevSourceLine = SourceLine;
		}
		else
			writer.Write((byte)InstructionType);
		WritePayload(writer, table);
	}

	protected virtual void WritePayload(BinaryWriter writer, NameTable table) { }
}