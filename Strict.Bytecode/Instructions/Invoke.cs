using Strict.Bytecode.Serialization;

namespace Strict.Bytecode.Instructions;

/// <summary>
/// Invokes a method using only register-based arguments and pure metadata (no expression trees).
/// Arguments and instance are compiled into registers by the BinaryGenerator before this
/// instruction, so the VM reads values from registers instead of evaluating expressions.
/// </summary>
public sealed class Invoke : RegisterInstruction
{
	public Invoke(Register register, InvokeMethodInfo methodInfo) : base(
		InstructionType.Invoke, register) =>
		MethodInfo = methodInfo;

	public Invoke(BinaryReader reader, NameTable table, BinaryExecutable binary) : base(
		InstructionType.Invoke, (Register)reader.ReadByte()) =>
		MethodInfo = new InvokeMethodInfo(reader, table);

	public InvokeMethodInfo MethodInfo { get; }
	/// <summary>
	/// Lazily cached precompiled instructions for this invoke, set on first execution to avoid
	/// repeated dictionary lookups in the hot path.
	/// </summary>
	internal List<Instruction>? CachedInstructions { get; set; }

	public override void Write(BinaryWriter writer, NameTable table)
	{
		base.Write(writer, table);
		MethodInfo.Write(writer, table);
	}
}