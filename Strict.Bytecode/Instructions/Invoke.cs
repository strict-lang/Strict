using Strict.Bytecode.Serialization;
using Strict.Expressions;
using Strict.Language;

namespace Strict.Bytecode.Instructions;

public sealed class Invoke(Register register, MethodCall method, Registry persistedRegistry)
	: RegisterInstruction(InstructionType.Invoke, register)
{
	public Invoke(BinaryReader reader, NameTable table, StrictBinary binary)
		: this((Register)reader.ReadByte(), binary.ReadMethodCall(reader, table), new Registry(reader)) { }

	public MethodCall Method { get; } = method;
	public Registry PersistedRegistry { get; } = persistedRegistry;

	public override void Write(BinaryWriter writer, NameTable table)
	{
		base.Write(writer, table);
		BytecodeSerializer.WriteMethodCallData(writer, Method, PersistedRegistry, table);
	}
}