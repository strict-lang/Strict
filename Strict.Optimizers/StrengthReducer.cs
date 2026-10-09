using Strict.Bytecode;
using Strict.Bytecode.Instructions;
using Strict.Expressions;
using Type = Strict.Language.Type;

namespace Strict.Optimizers;

/// <summary>
/// Reduces arithmetic identity and zero operations to simpler equivalents:
/// - x * 1 or 1 * x → x (remove constant load and binary, keep the variable load)
/// - x + 0 or 0 + x → x
/// - x - 0 → x
/// - x / 1 → x
/// - x * 0 or 0 * x → 0 (replace entire sequence with load constant 0)
/// The non-identity operand's register is rewritten to the result register where needed.
/// + 0, - 0 and * 0 need proven numbers: list + 0 appends, text + 0 concatenates.
/// </summary>
public sealed class StrengthReducer(params string[] numberVariables) : InstructionOptimizer
{
	public override void Optimize(BinaryExecutable binary)
	{
		foreach (var type in binary.MethodsPerType.Values)
		foreach (var methodGroup in type.MethodGroups.Values)
		foreach (var method in methodGroup)
			method.instructions = new StrengthReducer(type.Members.Concat(method.parameters).
				Where(member => member.JustTypeName == Type.Number).Select(member => member.Name).
				ToArray()).Optimize(method.instructions);
	}

	public override List<Instruction> Optimize(List<Instruction> instructions)
	{
		var numbers = numberVariables.Concat(instructions.OfType<StoreVariableInstruction>().
			Where(store => IsNumber(store.ValueInstance)).Select(store => store.Identifier)).ToHashSet();
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
				? ((LoadConstantInstruction)instructions[leftIndex]).Constant
				: (ValueInstance?)null;
			var rightValue = rightIsConst
				? ((LoadConstantInstruction)instructions[rightIndex]).Constant
				: (ValueInstance?)null;
			var areNumbers = IsNumber(instructions[leftIndex], numbers) &&
				IsNumber(instructions[rightIndex], numbers);
			if (areNumbers && TryReduceMultiplyByZero(instructions, i, binary, leftIndex, rightIndex,
				leftValue, rightValue))
				i = Math.Max(1, i - 2);
			else if (TryReduceIdentity(instructions, i, binary, leftIndex, rightIndex, leftValue,
				rightValue, areNumbers))
				i = Math.Max(1, i - 2);
		}
		return instructions;
	}

	private static bool IsNumber(ValueInstance value) =>
		value is { IsText: false, IsList: false, IsDictionary: false };

	private static bool IsNumber(Instruction instruction, HashSet<string> numbers) =>
		instruction is LoadConstantInstruction load && IsNumber(load.Constant) ||
		instruction is LoadVariableToRegister variable && numbers.Contains(variable.Identifier);

	private static bool TryReduceMultiplyByZero(List<Instruction> instructions, int binaryIndex,
		BinaryInstruction binary, int leftIndex, int rightIndex, ValueInstance? leftValue,
		ValueInstance? rightValue)
	{
		if (binary.InstructionType != InstructionType.Multiply)
			return false;
		var isLeftZero = leftValue is { Number: 0 };
		var isRightZero = rightValue is { Number: 0 };
		if (!isLeftZero && !isRightZero)
			return false;
		var resultRegister = binary.Registers[2];
		var zeroConst = isLeftZero
			? (LoadConstantInstruction)instructions[leftIndex]
			: (LoadConstantInstruction)instructions[rightIndex];
		instructions[binaryIndex] = new LoadConstantInstruction(resultRegister, zeroConst.Constant);
		RemoveIndicesDescending(instructions, leftIndex, rightIndex);
		return true;
	}

	private static bool TryReduceIdentity(List<Instruction> instructions, int binaryIndex,
		BinaryInstruction binary, int leftIndex, int rightIndex, ValueInstance? leftValue,
		ValueInstance? rightValue, bool areNumbers)
	{
		var identitySide = GetIdentitySide(binary.InstructionType, leftValue, rightValue, areNumbers);
		if (identitySide == IdentitySide.None)
			return false;
		var resultRegister = binary.Registers[2];
		var keepIndex = identitySide == IdentitySide.Left
			? rightIndex
			: leftIndex;
		var removeIndex = identitySide == IdentitySide.Left
			? leftIndex
			: rightIndex;
		RewriteRegister(instructions, keepIndex, resultRegister);
		RemoveIndicesDescending(instructions, binaryIndex, removeIndex);
		return true;
	}

	private static void RewriteRegister(List<Instruction> instructions, int index,
		Register newRegister)
	{
		if (instructions[index] is LoadVariableToRegister load1)
			instructions[index] = new LoadVariableToRegister(newRegister, load1.Identifier);
		else if (instructions[index] is LoadConstantInstruction load2)
			instructions[index] = new LoadConstantInstruction(newRegister, load2.Constant);
	}

	private static void RemoveIndicesDescending(List<Instruction> instructions, int a, int b)
	{
		instructions.RemoveAt(Math.Max(a, b));
		instructions.RemoveAt(Math.Min(a, b));
	}

	private static bool IsArithmetic(InstructionType instruction) =>
		instruction is > InstructionType.StoreSeparator and < InstructionType.ArithmeticSeparator;

	private static IdentitySide GetIdentitySide(InstructionType instruction, ValueInstance? leftValue,
		ValueInstance? rightValue, bool areNumbers) =>
		IsIdentityValue(instruction, leftValue, areNumbers)
			? IdentitySide.Left
			: IsIdentityValue(instruction, rightValue, areNumbers)
				? IdentitySide.Right
				: IdentitySide.None;

	private static bool IsIdentityValue(InstructionType instruction, ValueInstance? value,
		bool areNumbers) =>
		value is { } v && IsNumber(v) && instruction switch
		{
			InstructionType.Multiply or InstructionType.Divide => v.Number == 1,
			InstructionType.Add or InstructionType.Subtract => areNumbers && v.Number == 0,
			_ => false
		};

	private static int FindInstructionIndex(List<Instruction> instructions, int beforeIndex,
		Register register)
	{
		for (var index = beforeIndex - 1; index >= 0; index--)
		{
			var instruction = instructions[index];
			if (instruction is LoadConstantInstruction or LoadVariableToRegister &&
				((RegisterInstruction)instruction).Register == register)
				return index;
			if (GetWrittenRegister(instruction) == register || IsControlFlow(instruction))
				return -1;
		}
		return -1; //ncrunch: no coverage
	}

	private enum IdentitySide
	{
		None,
		Left,
		Right
	}
}