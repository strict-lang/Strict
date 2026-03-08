using Strict.Expressions;
using Strict.Runtime;
using Strict.Runtime.Statements;
using Binary = Strict.Runtime.Statements.Binary;

namespace Strict.Optimizers;

/// <summary>
/// Folds constant arithmetic at the instruction level. When two LoadConstantStatements feed
/// directly into a Binary arithmetic operation (Add, Subtract, Multiply, Divide, Modulo),
/// all three statements are replaced by a single LoadConstantStatement with the computed result.
/// This is repeated until no more folding is possible (handles chained constants).
/// </summary>
public sealed class ConstantFoldingOptimizer : StatementOptimizer
{
	public override List<Statement> Optimize(List<Statement> statements)
	{
		bool changed;
		do
		{
			changed = false;
			for (var i = 2; i < statements.Count; i++)
			{
				if (!TryFoldAt(statements, i))
					continue;
				changed = true;
				i--;
			}
		} while (changed);
		return statements;
	}

	private static bool TryFoldAt(List<Statement> statements, int binaryIndex)
	{
		if (statements[binaryIndex] is not Binary binary || binary.IsConditional() ||
			!IsArithmetic(binary.Instruction) || binary.Registers.Length < 3)
			return false;
		var leftRegister = binary.Registers[0];
		var rightRegister = binary.Registers[1];
		var resultRegister = binary.Registers[2];
		var leftLoad = FindLoadConstant(statements, binaryIndex, leftRegister);
		var rightLoad = FindLoadConstant(statements, binaryIndex, rightRegister);
		if (leftLoad == null || rightLoad == null)
			return false;
		var result = ComputeResult(binary.Instruction, leftLoad.ValueInstance,
			rightLoad.ValueInstance);
		if (result == null)
			return false;
		var foldedStatement = new LoadConstantStatement(resultRegister, result.Value);
		statements[binaryIndex] = foldedStatement;
		statements.Remove(leftLoad);
		statements.Remove(rightLoad);
		return true;
	}

	private static LoadConstantStatement? FindLoadConstant(List<Statement> statements,
		int beforeIndex, Register register)
	{
		for (var i = beforeIndex - 1; i >= 0; i--)
			if (statements[i] is LoadConstantStatement load && load.Register == register)
				return load;
		return null;
	}

	private static bool IsArithmetic(Instruction instruction) =>
		instruction is > Instruction.StoreSeparator and < Instruction.ArithmeticSeparator;

	private static ValueInstance? ComputeResult(Instruction operation, ValueInstance left,
		ValueInstance right) =>
		operation switch
		{
			Instruction.Add when left.IsText || right.IsText => new ValueInstance(
				(left.IsText ? left.Text : left.ToExpressionCodeString()) +
				(right.IsText ? right.Text : right.ToExpressionCodeString())),
			Instruction.Add => new ValueInstance(left.GetTypeExceptText(),
				left.Number + right.Number),
			Instruction.Subtract => new ValueInstance(left.GetTypeExceptText(),
				left.Number - right.Number),
			Instruction.Multiply => new ValueInstance(left.GetTypeExceptText(),
				left.Number * right.Number),
			Instruction.Divide => new ValueInstance(left.GetTypeExceptText(),
				left.Number / right.Number),
			Instruction.Modulo => new ValueInstance(left.GetTypeExceptText(),
				left.Number % right.Number),
			_ => null
		};
}
