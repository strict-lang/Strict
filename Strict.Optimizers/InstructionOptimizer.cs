using Strict.Runtime.Instructions;

namespace Strict.Optimizers;

/// <summary>
/// Base class for all statement-level optimizers that transform a list of bytecode statements
/// into an equivalent but more efficient list. Each optimizer focuses on a single optimization.
/// </summary>
public abstract class InstructionOptimizer
{
	public abstract List<Instruction> Optimize(List<Instruction> instructions);
}