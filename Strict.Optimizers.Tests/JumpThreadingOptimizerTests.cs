using Strict.Runtime.Statements;
using Binary = Strict.Runtime.Statements.Binary;
using Return = Strict.Runtime.Statements.Return;

namespace Strict.Optimizers.Tests;

public sealed class JumpThreadingOptimizerTests : TestOptimizers
{
	[Test]
	public void RemoveEmptyConditionalBlock() =>
		Assert.That(Optimize([
			new LoadConstantStatement(Register.R0, Num(5)),
			new LoadConstantStatement(Register.R1, Num(5)),
			new Binary(Instruction.Equal, Register.R0, Register.R1),
			new JumpToId(Instruction.JumpToIdIfFalse, 0),
			new JumpToId(Instruction.JumpEnd, 0),
			new Return(Register.R0)
		], 3).Count(s => s is JumpToId), Is.EqualTo(0));

	private List<Statement> Optimize(List<Statement> statements, int expectedCount) =>
		Optimize(new JumpThreadingOptimizer(), statements, expectedCount);

	[Test]
	public void KeepNonEmptyConditionalBlock() =>
		Optimize([
			new LoadConstantStatement(Register.R0, Num(5)),
			new LoadConstantStatement(Register.R1, Num(5)),
			new Binary(Instruction.Equal, Register.R0, Register.R1),
			new JumpToId(Instruction.JumpToIdIfFalse, 0),
			new LoadConstantStatement(Register.R2, Num(10)),
			new Return(Register.R2),
			new JumpToId(Instruction.JumpEnd, 0),
			new LoadConstantStatement(Register.R3, Num(20)),
			new Return(Register.R3)
		], 9);

	[Test]
	public void RemoveMultipleEmptyConditionalBlocks() =>
		Assert.That(Optimize([
			new Binary(Instruction.Equal, Register.R0, Register.R1),
			new JumpToId(Instruction.JumpToIdIfFalse, 0),
			new JumpToId(Instruction.JumpEnd, 0),
			new Binary(Instruction.Equal, Register.R2, Register.R3),
			new JumpToId(Instruction.JumpToIdIfFalse, 1),
			new JumpToId(Instruction.JumpEnd, 1),
			new Return(Register.R0)
		], 1).Count(s => s is JumpToId), Is.EqualTo(0));

	[Test]
	public void HandleEmptyStatementList() => Optimize([], 0);
}