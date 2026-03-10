using Strict.Expressions;
using Strict.Bytecode;
using Strict.Bytecode.Instructions;

namespace Strict.Optimizers;

/// <summary>
/// Folds constant arithmetic at the instruction level. When two LoadConstantInstructions feed
/// directly into a Binary arithmetic operation (Add, Subtract, Multiply, Divide, Modulo),
/// all three instructions are replaced by a single LoadConstantInstruction with the computed
/// result. This is repeated until no more folding is possible (handles chained constants).
/// </summary>
public sealed class ConstantFoldingOptimizer : InstructionOptimizer
{
	public override List<Instruction> Optimize(List<Instruction> instructions)
	{
		bool changed;
		do
		{
			changed = false;
			for (var i = 2; i < instructions.Count; i++)
			{
				if (!TryFoldAt(instructions, i))
					continue;
				changed = true;
				// After folding, the replacement shifts down so recheck from the new position
				i = Math.Max(1, i - 2);
			}
		} while (changed);
		return instructions;
	}

	private static bool TryFoldAt(List<Instruction> instructions, int binaryIndex)
	{
		if (instructions[binaryIndex] is not BinaryInstruction binary || binary.IsConditional() ||
			!IsArithmetic(binary.InstructionType) || binary.Registers.Length < 3)
			return false;
		var leftRegister = binary.Registers[0];
		var rightRegister = binary.Registers[1];
		var resultRegister = binary.Registers[2];
		var leftIndex = FindLoadConstantIndex(instructions, binaryIndex, leftRegister);
		var rightIndex = FindLoadConstantIndex(instructions, binaryIndex, rightRegister);
		if (leftIndex < 0 || rightIndex < 0)
			return false;
		var leftLoad = (LoadConstantInstruction)instructions[leftIndex];
		var rightLoad = (LoadConstantInstruction)instructions[rightIndex];
		var result = ComputeResult(binary.InstructionType, leftLoad.ValueInstance,
			rightLoad.ValueInstance);
		if (result == null)
			return false; //ncrunch: no coverage
		instructions[binaryIndex] = new LoadConstantInstruction(resultRegister, result.Value);
		// Remove in descending index order to preserve positions
		var first = Math.Max(leftIndex, rightIndex);
		var second = Math.Min(leftIndex, rightIndex);
		instructions.RemoveAt(first);
		instructions.RemoveAt(second);
		return true;
	}

	private static int FindLoadConstantIndex(List<Instruction> instructions, int beforeIndex,
		Register register)
	{
		for (var i = beforeIndex - 1; i >= 0; i--)
			if (instructions[i] is LoadConstantInstruction load && load.Register == register)
				return i;
		return -1;
	}

	private static bool IsArithmetic(InstructionType instruction) =>
		instruction is > InstructionType.StoreSeparator and < InstructionType.ArithmeticSeparator;

	private static ValueInstance? ComputeResult(InstructionType operation, ValueInstance left,
		ValueInstance right) =>
		operation switch
		{
			InstructionType.Add when left.IsText || right.IsText => new ValueInstance((left.IsText
				? left.Text
				: left.ToExpressionCodeString()) + (right.IsText
				? right.Text
				: right.ToExpressionCodeString())),
			InstructionType.Add => new ValueInstance(left.GetTypeExceptText(), left.Number + right.Number),
			InstructionType.Subtract => new ValueInstance(left.GetTypeExceptText(),
				left.Number - right.Number),
			InstructionType.Multiply => new ValueInstance(left.GetTypeExceptText(),
				left.Number * right.Number),
			InstructionType.Divide => new ValueInstance(left.GetTypeExceptText(),
				left.Number / right.Number),
			InstructionType.Modulo => new ValueInstance(left.GetTypeExceptText(),
				left.Number % right.Number),
			_ => null //ncrunch: no coverage
		};
}