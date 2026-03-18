using Strict.Bytecode;
using Strict.Bytecode.Instructions;

namespace Strict.Optimizers;

/// <summary>
/// Base class for all instruction-level optimizers that transform a list of bytecode instructions
/// into an equivalent but more efficient list. Each optimizer focuses on a single optimization.
/// </summary>
public abstract class InstructionOptimizer
{
	public void Optimize(BinaryExecutable binary)
	{
		foreach (var type in binary.MethodsPerType.Values)
		foreach (var methodGroup in type.MethodGroups.Values)
		foreach (var method in methodGroup)
			method.instructions = Optimize(method.instructions);
	}

	public abstract List<Instruction> Optimize(List<Instruction> instructions);
}