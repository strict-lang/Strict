using Strict.Expressions;
using Strict.Runtime.Statements;
using Binary = Strict.Runtime.Statements.Binary;
using Return = Strict.Runtime.Statements.Return;

namespace Strict.Optimizers.Tests;

public sealed class JumpThreadingOptimizerTests
{
	private static readonly Type NumberType = TestPackage.Instance.GetType(Type.Number);
	private static readonly Type BooleanType = TestPackage.Instance.GetType(Type.Boolean);
	private static ValueInstance Number(double value) => new(NumberType, value);

	[Test]
	public void RemoveEmptyConditionalBlock()
	{
		var statements = new List<Statement>
		{
			new LoadConstantStatement(Register.R0, Number(5)),
			new LoadConstantStatement(Register.R1, Number(5)),
			new Binary(Instruction.Equal, Register.R0, Register.R1),
			new JumpToId(Instruction.JumpToIdIfFalse, 0),
			new JumpToId(Instruction.JumpEnd, 0),
			new Return(Register.R0)
		};
		var optimized = new JumpThreadingOptimizer().Optimize(statements);
		Assert.That(optimized.Count(s => s is JumpToId), Is.EqualTo(0));
	}

	[Test]
	public void KeepNonEmptyConditionalBlock()
	{
		var statements = new List<Statement>
		{
			new LoadConstantStatement(Register.R0, Number(5)),
			new LoadConstantStatement(Register.R1, Number(5)),
			new Binary(Instruction.Equal, Register.R0, Register.R1),
			new JumpToId(Instruction.JumpToIdIfFalse, 0),
			new LoadConstantStatement(Register.R2, Number(10)),
			new Return(Register.R2),
			new JumpToId(Instruction.JumpEnd, 0),
			new LoadConstantStatement(Register.R3, Number(20)),
			new Return(Register.R3)
		};
		var optimized = new JumpThreadingOptimizer().Optimize(statements);
		Assert.That(optimized, Has.Count.EqualTo(9));
	}

	[Test]
	public void RemoveMultipleEmptyConditionalBlocks()
	{
		var statements = new List<Statement>
		{
			new Binary(Instruction.Equal, Register.R0, Register.R1),
			new JumpToId(Instruction.JumpToIdIfFalse, 0),
			new JumpToId(Instruction.JumpEnd, 0),
			new Binary(Instruction.Equal, Register.R2, Register.R3),
			new JumpToId(Instruction.JumpToIdIfFalse, 1),
			new JumpToId(Instruction.JumpEnd, 1),
			new Return(Register.R0)
		};
		var optimized = new JumpThreadingOptimizer().Optimize(statements);
		Assert.That(optimized.Count(s => s is JumpToId), Is.EqualTo(0));
	}

	[Test]
	public void HandleEmptyStatementList()
	{
		var statements = new List<Statement>();
		var optimized = new JumpThreadingOptimizer().Optimize(statements);
		Assert.That(optimized, Is.Empty);
	}
}
