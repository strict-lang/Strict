using Strict.Runtime;
using Strict.Runtime.Instructions;

namespace Strict.Optimizers;

/// <summary>
/// Removes instructions that follow an unconditional Return or Jump instruction and can never
/// execute. Scans linearly; once an unconditional Return is found, everything after it is dead
/// unless a JumpEnd or loop boundary appears that could serve as a branch target.
/// </summary>
public sealed class UnreachableCodeEliminator : InstructionOptimizer
{
	public override List<Instruction> Optimize(List<Instruction> instructions)
	{
		var depth = 0;
		for (var i = 0; i < instructions.Count - 1; i++)
		{
			if (instructions[i] is JumpToId
				{
					InstructionType: InstructionType.JumpToIdIfFalse or InstructionType.JumpToIdIfTrue
				})
				depth++;
			else if (instructions[i] is JumpToId { InstructionType: InstructionType.JumpEnd })
				depth = Math.Max(0, depth - 1);
			else if (depth == 0 && instructions[i] is ReturnInstruction or Jump
				{
					InstructionType: InstructionType.Jump
				})
			{
				instructions.RemoveRange(i + 1, instructions.Count - i - 1);
				break;
			}
		}
		return instructions;
	}
}