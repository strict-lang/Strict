using Strict.Bytecode;
using Strict.Bytecode.Instructions;

namespace Strict.Optimizers;

/// <summary>
/// Base class for all instruction-level optimizers that transform a list of bytecode instructions
/// into an equivalent but more efficient list. Each optimizer focuses on a single optimization.
/// </summary>
public abstract class InstructionOptimizer
{
	public virtual void Optimize(BinaryExecutable binary)
	{
		foreach (var type in binary.MethodsPerType.Values)
		foreach (var methodGroup in type.MethodGroups.Values)
		foreach (var method in methodGroup)
			method.instructions = Optimize(method.instructions);
	}

	public abstract List<Instruction> Optimize(List<Instruction> instructions);

	protected static Register? GetWrittenRegister(Instruction instruction) =>
		instruction switch
		{
			BinaryInstruction { Registers.Length: >= 3 } binary => binary.Registers[^1],
			LoadVariableToRegister or LoadConstantInstruction or SetInstruction or Invoke
				or FieldLoadInstruction or ConstructValueTypeInstruction or ListCallInstruction =>
				((RegisterInstruction)instruction).Register,
			_ => null
		};

	protected static bool IsControlFlow(Instruction instruction) =>
		instruction.InstructionType is >= InstructionType.LoopBegin and
			<= InstructionType.JumpToIdIfTrue;
}