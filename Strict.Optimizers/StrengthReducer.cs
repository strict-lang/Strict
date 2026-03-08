using Strict.Expressions;
using Strict.Runtime;
using Strict.Runtime.Statements;
using Binary = Strict.Runtime.Statements.Binary;

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
public sealed class StrengthReducer : StatementOptimizer
{
	public override List<Statement> Optimize(List<Statement> statements)
	{
		for (var i = 2; i < statements.Count; i++)
		{
			if (statements[i] is not Binary binary || binary.IsConditional() ||
				!IsArithmetic(binary.Instruction) || binary.Registers.Length < 3)
				continue;
			var leftIndex = FindStatementIndex(statements, i, binary.Registers[0]);
			var rightIndex = FindStatementIndex(statements, i, binary.Registers[1]);
			if (leftIndex < 0 || rightIndex < 0)
				continue;
			var leftIsConst = statements[leftIndex] is LoadConstantStatement;
			var rightIsConst = statements[rightIndex] is LoadConstantStatement;
			if (!leftIsConst && !rightIsConst)
				continue;
			var leftValue = leftIsConst ? ((LoadConstantStatement)statements[leftIndex]).ValueInstance : (ValueInstance?)null;
			var rightValue = rightIsConst ? ((LoadConstantStatement)statements[rightIndex]).ValueInstance : (ValueInstance?)null;
			if (TryReduceMultiplyByZero(statements, i, binary, leftIndex, rightIndex, leftValue, rightValue))
				{ i = Math.Max(1, i - 2); continue; }
			if (TryReduceIdentity(statements, i, binary, leftIndex, rightIndex, leftValue, rightValue))
				{ i = Math.Max(1, i - 2); continue; }
		}
		return statements;
	}

	private static bool TryReduceMultiplyByZero(List<Statement> statements, int binaryIndex,
		Binary binary, int leftIndex, int rightIndex, ValueInstance? leftValue,
		ValueInstance? rightValue)
	{
		if (binary.Instruction != Instruction.Multiply)
			return false;
		var isLeftZero = leftValue is { } lv && !lv.IsText && lv.Number == 0;
		var isRightZero = rightValue is { } rv && !rv.IsText && rv.Number == 0;
		if (!isLeftZero && !isRightZero)
			return false;
		var resultRegister = binary.Registers[2];
		var zeroConst = isLeftZero ? (LoadConstantStatement)statements[leftIndex]
			: (LoadConstantStatement)statements[rightIndex];
		statements[binaryIndex] = new LoadConstantStatement(resultRegister, zeroConst.ValueInstance);
		RemoveIndicesDescending(statements, leftIndex, rightIndex);
		return true;
	}

	private static bool TryReduceIdentity(List<Statement> statements, int binaryIndex,
		Binary binary, int leftIndex, int rightIndex, ValueInstance? leftValue,
		ValueInstance? rightValue)
	{
		var identitySide = GetIdentitySide(binary.Instruction, leftValue, rightValue);
		if (identitySide == IdentitySide.None)
			return false;
		var resultRegister = binary.Registers[2];
		var keepIndex = identitySide == IdentitySide.Left ? rightIndex : leftIndex;
		var removeIndex = identitySide == IdentitySide.Left ? leftIndex : rightIndex;
		RewriteRegister(statements, keepIndex, resultRegister);
		RemoveIndicesDescending(statements, binaryIndex, removeIndex);
		return true;
	}

	private static void RewriteRegister(List<Statement> statements, int index,
		Register newRegister)
	{
		var statement = statements[index];
		statements[index] = statement switch
		{
			LoadVariableToRegister load => new LoadVariableToRegister(newRegister, load.Identifier),
			LoadConstantStatement load => new LoadConstantStatement(newRegister, load.ValueInstance),
			_ => statement
		};
	}

	private static void RemoveIndicesDescending(List<Statement> statements, int a, int b)
	{
		var first = Math.Max(a, b);
		var second = Math.Min(a, b);
		statements.RemoveAt(first);
		statements.RemoveAt(second);
	}

	private static bool IsArithmetic(Instruction instruction) =>
		instruction is > Instruction.StoreSeparator and < Instruction.ArithmeticSeparator;

	private static IdentitySide GetIdentitySide(Instruction instruction, ValueInstance? leftValue,
		ValueInstance? rightValue)
	{
		if (IsIdentityValue(instruction, leftValue))
			return IdentitySide.Left;
		if (IsIdentityValue(instruction, rightValue))
			return IdentitySide.Right;
		return IdentitySide.None;
	}

	private static bool IsIdentityValue(Instruction instruction, ValueInstance? value)
	{
		if (value is not { } v || v.IsText)
			return false;
		return instruction switch
		{
			Instruction.Multiply when v.Number == 1 => true,
			Instruction.Divide when v.Number == 1 => true,
			Instruction.Add when v.Number == 0 => true,
			Instruction.Subtract when v.Number == 0 => true,
			_ => false
		};
	}

	private static int FindStatementIndex(List<Statement> statements, int beforeIndex,
		Register register)
	{
		for (var i = beforeIndex - 1; i >= 0; i--)
			if (statements[i] is LoadConstantStatement load && load.Register == register ||
				statements[i] is LoadVariableToRegister varLoad && varLoad.Register == register)
				return i;
		return -1;
	}

	private enum IdentitySide { None, Left, Right }
}
