using Strict.Bytecode.Serialization;
using Strict.Expressions;

namespace Strict.Bytecode.Instructions;

//TODO: remove
public sealed class Invoke(Register register, MethodCall method, Registry persistedRegistry)
	: RegisterInstruction(InstructionType.Invoke, register)
{
	public Invoke(BinaryReader reader, NameTable table, BinaryExecutable binary)
		: this((Register)reader.ReadByte(), binary.ReadMethodCall(reader, table), new Registry(reader)) { }

	public MethodCall Method { get; } = method;
	public Registry PersistedRegistry { get; } = persistedRegistry;

	public override void Write(BinaryWriter writer, NameTable table)
	{
		base.Write(writer, table);
		BinaryExecutable.WriteMethodCallData(writer, Method, PersistedRegistry, table);
	}
}