using Strict.Expressions;
using Strict.Runtime;
using Strict.Runtime.Instructions;

namespace Strict.Optimizers;

/// <summary>
/// Reduces arithmetic identity and zero operations to simpler equivalents:
/// - x * 1 or 1 * x → x (remove constant load and binary, keep the variable load)
/// - x + 0 or 0 + x → x
/// - x - 0 → x
/// - x / 1 → x
/// - x * 0 or 0 * x → 0 (replace entire sequence with load constant 0)
/// The non-identity operand's register is rewritten to the result register where needed.
/// </summary>
public sealed class StrengthReducer : InstructionOptimizer
{
	public override List<Instruction> Optimize(List<Instruction> instructions)
	{
		for (var i = 2; i < instructions.Count; i++)
		{
			if (instructions[i] is not BinaryInstruction binary || binary.IsConditional() ||
				!IsArithmetic(binary.InstructionType) || binary.Registers.Length < 3)
				continue;
			var leftIndex = FindInstructionIndex(instructions, i, binary.Registers[0]);
			var rightIndex = FindInstructionIndex(instructions, i, binary.Registers[1]);
			if (leftIndex < 0 || rightIndex < 0)
				continue; //ncrunch: no coverage
			var leftIsConst = instructions[leftIndex] is LoadConstantInstruction;
			var rightIsConst = instructions[rightIndex] is LoadConstantInstruction;
			if (!leftIsConst && !rightIsConst)
				continue;
			var leftValue = leftIsConst
				? ((LoadConstantInstruction)instructions[leftIndex]).ValueInstance
				: (ValueInstance?)null;
			var rightValue = rightIsConst
				? ((LoadConstantInstruction)instructions[rightIndex]).ValueInstance
				: (ValueInstance?)null;
			if (TryReduceMultiplyByZero(instructions, i, binary, leftIndex, rightIndex, leftValue,
				rightValue))
				i = Math.Max(1, i - 2);
			else if (TryReduceIdentity(instructions, i, binary, leftIndex, rightIndex, leftValue,
				rightValue))
				i = Math.Max(1, i - 2);
		}
		return instructions;
	}

	private static bool TryReduceMultiplyByZero(List<Instruction> Instructions, int binaryIndex,
		BinaryInstruction binary, int leftIndex, int rightIndex, ValueInstance? leftValue,
		ValueInstance? rightValue)
	{
		if (binary.InstructionType != InstructionType.Multiply)
			return false;
		var isLeftZero = leftValue is { } lv && !lv.IsText && lv.Number == 0;
		var isRightZero = rightValue is { } rv && !rv.IsText && rv.Number == 0;
		if (!isLeftZero && !isRightZero)
			return false;
		var resultRegister = binary.Registers[2];
		var zeroConst = isLeftZero
			? (LoadConstantInstruction)Instructions[leftIndex]
			: (LoadConstantInstruction)Instructions[rightIndex];
		Instructions[binaryIndex] = new LoadConstantInstruction(resultRegister, zeroConst.ValueInstance);
		RemoveIndicesDescending(Instructions, leftIndex, rightIndex);
		return true;
	}

	private static bool TryReduceIdentity(List<Instruction> Instructions, int binaryIndex,
		BinaryInstruction binary, int leftIndex, int rightIndex, ValueInstance? leftValue,
		ValueInstance? rightValue)
	{
		var identitySide = GetIdentitySide(binary.InstructionType, leftValue, rightValue);
		if (identitySide == IdentitySide.None)
			return false;
		var resultRegister = binary.Registers[2];
		var keepIndex = identitySide == IdentitySide.Left
			? rightIndex
			: leftIndex;
		var removeIndex = identitySide == IdentitySide.Left
			? leftIndex
			: rightIndex;
		RewriteRegister(Instructions, keepIndex, resultRegister);
		RemoveIndicesDescending(Instructions, binaryIndex, removeIndex);
		return true;
	}

	private static void RewriteRegister(List<Instruction> Instructions, int index, Register newRegister)
	{
		var Instruction = Instructions[index];
		Instructions[index] = Instruction switch
		{
			LoadVariableToRegister load => new LoadVariableToRegister(newRegister, load.Identifier),
			//ncrunch: no coverage start
			LoadConstantInstruction load => new LoadConstantInstruction(newRegister, load.ValueInstance),
			_ => Instruction
		}; //ncrunch: no coverage end
	}

	private static void RemoveIndicesDescending(List<Instruction> Instructions, int a, int b)
	{
		var first = Math.Max(a, b);
		var second = Math.Min(a, b);
		Instructions.RemoveAt(first);
		Instructions.RemoveAt(second);
	}

	private static bool IsArithmetic(InstructionType instruction) =>
		instruction is > InstructionType.StoreSeparator and < InstructionType.ArithmeticSeparator;

	private static IdentitySide GetIdentitySide(InstructionType instruction, ValueInstance? leftValue,
		ValueInstance? rightValue) =>
		IsIdentityValue(instruction, leftValue)
			? IdentitySide.Left
			: IsIdentityValue(instruction, rightValue)
				? IdentitySide.Right
				: IdentitySide.None;

	private static bool IsIdentityValue(InstructionType instruction, ValueInstance? value)
	{
		if (value is not { } v || v.IsText)
			return false;
		return instruction switch
		{
			InstructionType.Multiply when v.Number == 1 => true,
			InstructionType.Divide when v.Number == 1 => true,
			InstructionType.Add when v.Number == 0 => true,
			InstructionType.Subtract when v.Number == 0 => true,
			_ => false
		};
	}

	private static int FindInstructionIndex(List<Instruction> Instructions, int beforeIndex,
		Register register)
	{
		for (var i = beforeIndex - 1; i >= 0; i--)
			if (Instructions[i] is LoadConstantInstruction load && load.Register == register ||
				Instructions[i] is LoadVariableToRegister varLoad && varLoad.Register == register)
				return i;
		return -1; //ncrunch: no coverage
	}

	private enum IdentitySide
	{
		None,
		Left,
		Right
	}
}