using Strict.Expressions;
using Strict.Bytecode;
using Strict.Bytecode.Instructions;

namespace Strict.Optimizers;

/// <summary>
/// Removes passed test assertion patterns from bytecode. In Strict, every method starts with
/// self-contained tests that are executed once; passing expressions become true and are pruned
/// </summary>
public sealed class TestCodeRemover : InstructionOptimizer
{
	public override List<Instruction> Optimize(List<Instruction> instructions)
	{
		bool changed;
		do
		{
			changed = false;
			for (var i = 0; i <= instructions.Count - 5; i++)
			{
				if (!IsPassedTestPattern(instructions, i))
					continue;
				instructions.RemoveRange(i, 5);
				changed = true;
				i = Math.Max(0, i - 1);
			}
		} while (changed);
		return instructions;
	}

	private static bool IsPassedTestPattern(List<Instruction> instructions, int startIndex)
	{
		if (instructions[startIndex] is not LoadConstantInstruction leftLoad ||
			instructions[startIndex + 1] is not LoadConstantInstruction rightLoad)
			return false;
		if (instructions[startIndex + 2] is not BinaryInstruction binary || !binary.IsConditional())
			return false;
		if (binary.InstructionType != InstructionType.Equal)
			return false; //ncrunch: no coverage
		if (instructions[startIndex + 3] is not JumpToId conditional ||
			conditional.InstructionType != InstructionType.JumpToIdIfFalse)
			return false; //ncrunch: no coverage
		if (instructions[startIndex + 4] is not JumpToId end ||
			end.InstructionType != InstructionType.JumpEnd || end.Id != conditional.Id)
			return false;
		return AreValuesEqual(leftLoad.Constant, rightLoad.Constant);
	}

	private static bool AreValuesEqual(ValueInstance left, ValueInstance right)
	{
		if (left.IsText && right.IsText)
			return left.Text == right.Text;
		if (!left.IsText && !right.IsText)
			return left.Number == right.Number;
		return false; //ncrunch: no coverage
	}
}
