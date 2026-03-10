using Strict.Expressions;

namespace Strict.Bytecode.Instructions;

public abstract class InstanceInstruction(InstructionType instructionType,
	ValueInstance valueInstance) : Instruction(instructionType)
{
	public ValueInstance ValueInstance { get; } = valueInstance;
	public override string ToString() => $"{InstructionType} {ValueInstance.ToExpressionCodeString()}";
}