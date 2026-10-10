using Strict.Bytecode.Serialization;

namespace Strict.Bytecode.Instructions;

/// <summary>
/// A list variable gets its own copy before its first element write while another variable,
/// member or list may still use that list (values never change). Owned lists change in place.
/// </summary>
public sealed class CopyListInstruction(string identifier) : Instruction(InstructionType.CopyList)
{
	public CopyListInstruction(BinaryReader reader, NameTable table) : this(
		table.names[reader.Read7BitEncodedInt()]) { }

	public string Identifier { get; } = identifier;
	public override string ToString() => $"{InstructionType} {Identifier}";

	protected override void WritePayload(BinaryWriter writer, NameTable table) =>
		writer.Write7BitEncodedInt(table[Identifier]);
}
