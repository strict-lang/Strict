using Strict.Expressions;
using Strict.Runtime;
using Strict.Runtime.Statements;
using Binary = Strict.Runtime.Statements.Binary;

namespace Strict.Optimizers;

/// <summary>
/// Removes passed test assertion patterns from bytecode. In Strict, every method starts with
/// self-contained tests that are "executed once; passing expressions become true and are pruned"
/// (README). A passed test typically generates:
///   LoadConstant result, LoadConstant expected, Equal, JumpToIdIfFalse(id), JumpEnd(id)
/// When both constants match (the test passed), the entire block is provably dead and removed.
/// Only removes the pattern when: both operands are constants, they are equal, the conditional
/// block is empty (JumpToIdIfFalse immediately followed by its matching JumpEnd).
/// </summary>
public sealed class TestCodeRemover : StatementOptimizer
{
	public override List<Statement> Optimize(List<Statement> statements)
	{
		bool changed;
		do
		{
			changed = false;
			for (var i = 0; i <= statements.Count - 5; i++)
			{
				if (!IsPassedTestPattern(statements, i))
					continue;
				statements.RemoveRange(i, 5);
				changed = true;
				i = Math.Max(0, i - 1);
			}
		} while (changed);
		return statements;
	}

	private static bool IsPassedTestPattern(List<Statement> statements, int startIndex)
	{
		if (statements[startIndex] is not LoadConstantStatement leftLoad ||
			statements[startIndex + 1] is not LoadConstantStatement rightLoad)
			return false;
		if (statements[startIndex + 2] is not Binary binary || !binary.IsConditional())
			return false;
		if (binary.Instruction != Instruction.Equal)
			return false; //ncrunch: no coverage
		if (statements[startIndex + 3] is not JumpToId conditional ||
			conditional.Instruction != Instruction.JumpToIdIfFalse)
			return false; //ncrunch: no coverage
		if (statements[startIndex + 4] is not JumpToId end ||
			end.Instruction != Instruction.JumpEnd || end.Id != conditional.Id)
			return false;
		return AreValuesEqual(leftLoad.ValueInstance, rightLoad.ValueInstance);
	}

	private static bool AreValuesEqual(ValueInstance left, ValueInstance right)
	{
		if (left.IsText && right.IsText)
			return left.Text == right.Text; //ncrunch: no coverage
		if (!left.IsText && !right.IsText)
			return left.Number == right.Number;
		return false; //ncrunch: no coverage
	}
}
