using Strict.Bytecode.Serialization;
using Strict.Expressions;
using Strict.Language;

namespace Strict.Bytecode.Instructions;

public sealed class StoreVariableInstruction(ValueInstance constant, string identifier,
	bool isMember = false) : InstanceInstruction(InstructionType.StoreConstantToVariable, constant)
{
	public StoreVariableInstruction(BinaryReader reader, NameTable table, StrictBinary binary)
		: this(binary.ReadValueInstance(reader, table), table.Names[reader.Read7BitEncodedInt()],
				reader.ReadBoolean()) { }

	public string Identifier { get; } = identifier;
	public bool IsMember { get; } = isMember;
	public override string ToString() => $"{base.ToString()} {Identifier}";

	public override void Write(BinaryWriter writer, NameTable table)
	{
		base.Write(writer, table);
		writer.Write7BitEncodedInt(table[Identifier]);
		writer.Write(IsMember);
	}
}