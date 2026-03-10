using Strict.Runtime;
using Strict.Runtime.Instructions;

namespace Strict.Optimizers;

/// <summary>
/// Removes empty conditional blocks — JumpToIdIfFalse/JumpToIdIfTrue immediately followed by
/// the matching JumpEnd with the same ID. This pattern appears when test code or other dead
/// branches have been stripped but the jump pair remains. Also removes the preceding comparison
/// BinaryInstruction if it is no longer needed.
/// </summary>
public sealed class JumpThreadingOptimizer : InstructionOptimizer
{
	public override List<Instruction> Optimize(List<Instruction> instructions)
	{
		bool changed;
		do
		{
			changed = false;
			for (var i = 0; i < instructions.Count - 1; i++)
			{
				if (instructions[i] is not JumpToId conditional ||
					conditional.InstructionType is not (InstructionType.JumpToIdIfFalse
						or InstructionType.JumpToIdIfTrue))
					continue;
				if (instructions[i + 1] is not JumpToId end ||
					end.InstructionType != InstructionType.JumpEnd || end.Id != conditional.Id)
					continue;
				instructions.RemoveAt(i + 1);
				instructions.RemoveAt(i);
				if (i > 0 && instructions[i - 1] is BinaryInstruction binary && binary.IsConditional())
					instructions.RemoveAt(--i);
				changed = true;
				i = Math.Max(0, i - 1);
			}
		} while (changed);
		return instructions;
	}
}