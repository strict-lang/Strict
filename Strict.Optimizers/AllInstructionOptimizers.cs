using Strict.Runtime.Instructions;

namespace Strict.Optimizers;

/// <summary>
/// Chains all instruction-level optimizers into a single optimization pipeline. The order is:
/// 1. TestCodeRemover — strip passed test assertions first (the core Optimizer mission)
/// 2. ConstantFoldingOptimizer — fold remaining constant arithmetic
/// 3. StrengthReducer — simplify identity/zero operations (x*1→x, x+0→x, x*0→0)
/// 4. DeadStoreEliminator — remove stores to variables that are never loaded
/// 5. RedundantLoadEliminator — remove duplicate loads of the same variable
/// 6. JumpThreadingOptimizer — remove empty conditional jump blocks
/// 7. UnreachableCodeEliminator — remove instructions after unconditional return/jump
/// </summary>
public class AllInstructionOptimizers : InstructionOptimizer
{
	private readonly InstructionOptimizer[] optimizers =
	[
		new TestCodeRemover(),
		new ConstantFoldingOptimizer(),
		new StrengthReducer(),
		new DeadStoreEliminator(),
		new RedundantLoadEliminator(),
		new JumpThreadingOptimizer(),
		new UnreachableCodeEliminator()
	];

	public override List<Instruction> Optimize(List<Instruction> instructions)
	{
		foreach (var optimizer in optimizers)
			instructions = optimizer.Optimize(instructions);
		return instructions;
	}
}