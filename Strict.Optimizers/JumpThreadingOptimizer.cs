using Strict.Runtime;
using Strict.Runtime.Statements;

namespace Strict.Optimizers;

/// <summary>
/// Removes empty conditional blocks — JumpToIdIfFalse/JumpToIdIfTrue immediately followed by
/// the matching JumpEnd with the same ID. This pattern appears when test code or other dead
/// branches have been stripped but the jump pair remains. Also removes the preceding comparison
/// Binary statement if it is no longer needed.
/// </summary>
public sealed class JumpThreadingOptimizer : StatementOptimizer
{
	public override List<Statement> Optimize(List<Statement> statements)
	{
		bool changed;
		do
		{
			changed = false;
			for (var i = 0; i < statements.Count - 1; i++)
			{
				if (statements[i] is not JumpToId conditional ||
					conditional.Instruction is not (Instruction.JumpToIdIfFalse
						or Instruction.JumpToIdIfTrue))
					continue;
				if (statements[i + 1] is not JumpToId end ||
					end.Instruction != Instruction.JumpEnd || end.Id != conditional.Id)
					continue;
				statements.RemoveAt(i + 1);
				statements.RemoveAt(i);
				if (i > 0 && statements[i - 1] is Binary binary && binary.IsConditional())
					statements.RemoveAt(--i);
				changed = true;
				i = Math.Max(0, i - 1);
			}
		} while (changed);
		return statements;
	}
}