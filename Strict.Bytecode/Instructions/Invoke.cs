using Strict.Bytecode.Serialization;
using Strict.Expressions;

namespace Strict.Bytecode.Instructions;

public sealed class Invoke(Register register, MethodCall method, Registry persistedRegistry)
	: RegisterInstruction(InstructionType.Invoke, register)
{
	public Invoke(BinaryReader reader, NameTable table, BinaryExecutable binary) : this(
		(Register)reader.ReadByte(), ReadMethod(reader, table, binary), ReadRegistry(reader)) { }

	private static MethodCall ReadMethod(BinaryReader reader, NameTable table,
		BinaryExecutable binary) =>
		reader.ReadBoolean()
			? binary.ReadMethodCall(reader, table)
			: throw new InvalidOperationException("Invoke instruction is missing method call data");

	private static Registry ReadRegistry(BinaryReader reader) =>
		reader.ReadBoolean()
			? new Registry(reader)
			: new Registry();

	public MethodCall Method { get; } = method;
	public Registry PersistedRegistry { get; } = persistedRegistry;
	/// <summary>
	/// Lazily cached precompiled instructions for this invoke, set on first execution to avoid
	/// repeated dictionary lookups in the hot path.
	/// </summary>
	internal IReadOnlyList<Instruction>? CachedInstructions { get; set; }

	public override void Write(BinaryWriter writer, NameTable table)
	{
		base.Write(writer, table);
		BinaryExecutable.WriteMethodCallData(writer, Method, PersistedRegistry, table);
	}
}