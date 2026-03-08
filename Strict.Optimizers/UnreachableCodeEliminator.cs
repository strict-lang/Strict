using Strict.Runtime;
using Strict.Runtime.Statements;

namespace Strict.Optimizers;

/// <summary>
/// Removes statements that follow an unconditional Return or Jump instruction and can never
/// execute. Scans linearly; once an unconditional Return is found, everything after it is dead
/// unless a JumpEnd or loop boundary appears that could serve as a branch target.
/// </summary>
public sealed class UnreachableCodeEliminator : StatementOptimizer
{
	public override List<Statement> Optimize(List<Statement> statements)
	{
		var depth = 0;
		for (var i = 0; i < statements.Count - 1; i++)
		{
			if (statements[i] is JumpToId { Instruction: Instruction.JumpToIdIfFalse or Instruction.JumpToIdIfTrue })
				depth++;
			else if (statements[i] is JumpToId { Instruction: Instruction.JumpEnd })
				depth = Math.Max(0, depth - 1);
			else if (depth == 0 && statements[i] is Return or Jump { Instruction: Instruction.Jump })
			{
				statements.RemoveRange(i + 1, statements.Count - i - 1);
				break;
			}
		}
		return statements;
	}
}