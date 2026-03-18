using Strict.Bytecode.Serialization;
using Strict.Expressions;

namespace Strict.Bytecode.Instructions;

/// <summary>
/// Loads a constant value to one of the <see cref="Register" />s, which could be an actual
/// number, a boolean or a pointer to some memory (usually an offset).
/// </summary>
public sealed class LoadConstantInstruction(Register register, ValueInstance constant)
	: RegisterInstruction(InstructionType.LoadConstantToRegister, register)
{
	public LoadConstantInstruction(BinaryReader reader, NameTable table, BinaryExecutable binary)
		: this((Register)reader.ReadByte(), binary.ReadValueInstance(reader, table)) { }

	public ValueInstance Constant { get; } = constant;
	public override string ToString() => $"{base.ToString()} {Register} {Constant}";

	public override void Write(BinaryWriter writer, NameTable table)
	{
		base.Write(writer, table);
		BinaryExecutable.WriteValueInstance(writer, Constant, table);
	}
}