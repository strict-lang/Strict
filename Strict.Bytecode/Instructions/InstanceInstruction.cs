using Strict.Bytecode.Serialization;
using Strict.Expressions;

namespace Strict.Bytecode.Instructions;

public abstract class InstanceInstruction(InstructionType instructionType,
	ValueInstance valueInstance) : Instruction(instructionType)
{
	public ValueInstance ValueInstance { get; } = valueInstance;
	public override string ToString() => $"{InstructionType} {ValueInstance.ToExpressionCodeString()}";

	protected override void WritePayload(BinaryWriter writer, NameTable table) =>
		BinaryExecutable.WriteValueInstance(writer, ValueInstance, table);
}