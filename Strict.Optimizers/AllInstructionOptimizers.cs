using Strict.Bytecode;
using Strict.Bytecode.Instructions;

namespace Strict.Optimizers;

/// <summary>
/// Chains all instruction-level optimizers into a single optimization pipeline.
/// </summary>
public class AllInstructionOptimizers : InstructionOptimizer
{
	private readonly InstructionOptimizer[] optimizers =
	[
		new TestCodeRemover(),
		new ConstantFoldingOptimizer(),
		new MethodInliningOptimizer(),
		new StrengthReducer(),
		new DeadStoreEliminator(),
		new RedundantLoadEliminator(),
		new JumpThreadingOptimizer(),
		new UnreachableCodeEliminator()
	];
	public int NumberOfOptimizers => optimizers.Length;

	public override void Optimize(BinaryExecutable binary)
	{
		foreach (var optimizer in optimizers)
			optimizer.Optimize(binary);
	}

	public override List<Instruction> Optimize(List<Instruction> instructions)
	{
		foreach (var optimizer in optimizers)
			instructions = optimizer.Optimize(instructions);
		return instructions;
	}
}