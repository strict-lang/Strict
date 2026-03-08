using Strict.Runtime.Statements;

namespace Strict.Optimizers;

/// <summary>
/// Chains all statement-level optimizers into a single optimization pipeline. Runs
/// ConstantFoldingOptimizer, then DeadStoreEliminator, then RedundantLoadEliminator
/// in order, each feeding its output to the next. This order ensures that constants
/// are folded first, then dead stores from folded code are removed, and finally
/// redundant loads are eliminated.
/// </summary>
public sealed class InstructionOptimizer : StatementOptimizer
{
	private readonly StatementOptimizer[] optimizers =
	[
		new ConstantFoldingOptimizer(),
		new DeadStoreEliminator(),
		new RedundantLoadEliminator()
	];

	public override List<Statement> Optimize(List<Statement> statements)
	{
		foreach (var optimizer in optimizers)
			statements = optimizer.Optimize(statements);
		return statements;
	}
}
