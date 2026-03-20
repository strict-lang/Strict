using Strict.Bytecode.Serialization;
using Strict.Expressions;

namespace Strict.Bytecode.Instructions;

public sealed class SetInstruction(ValueInstance valueInstance, Register register)
	: InstanceInstruction(InstructionType.Set, valueInstance)
{
	public SetInstruction(BinaryReader reader, NameTable table, BinaryExecutable binary)
		: this(binary.ReadValueInstance(reader, table), (Register)reader.ReadByte()) { }

	public Register Register { get; } = register;
	public override string ToString() => $"{base.ToString()} {Register}";

	public override void Write(BinaryWriter writer, NameTable table)
	{
		base.Write(writer, table);
		writer.Write((byte)Register);
	}
}