using Strict.Bytecode.Instructions;

namespace Strict.Optimizers.Tests;

public sealed class JumpThreadingOptimizerTests : TestOptimizers
{
	[Test]
	public void RemoveEmptyConditionalBlock() =>
		Assert.That(Optimize([
			new LoadConstantInstruction(Register.R0, Num(5)),
			new LoadConstantInstruction(Register.R1, Num(5)),
			new BinaryInstruction(InstructionType.Equal, Register.R0, Register.R1),
			new JumpToId(0, InstructionType.JumpToIdIfFalse),
			new JumpToId(0, InstructionType.JumpEnd),
			new ReturnInstruction(Register.R0)
		], 3).Count(s => s is JumpToId), Is.EqualTo(0));

	private List<Instruction> Optimize(List<Instruction> instructions, int expectedCount) =>
		Optimize(new JumpThreadingOptimizer(), instructions, expectedCount);

	[Test]
	public void KeepNonEmptyConditionalBlock() =>
		Optimize([
			new LoadConstantInstruction(Register.R0, Num(5)),
			new LoadConstantInstruction(Register.R1, Num(5)),
			new BinaryInstruction(InstructionType.Equal, Register.R0, Register.R1),
			new JumpToId(0, InstructionType.JumpToIdIfFalse),
			new LoadConstantInstruction(Register.R2, Num(10)),
			new ReturnInstruction(Register.R2),
			new JumpToId(0, InstructionType.JumpEnd),
			new LoadConstantInstruction(Register.R3, Num(20)),
			new ReturnInstruction(Register.R3)
		], 9);

	[Test]
	public void RemoveMultipleEmptyConditionalBlocks() =>
		Assert.That(Optimize([
			new BinaryInstruction(InstructionType.Equal, Register.R0, Register.R1),
			new JumpToId(0, InstructionType.JumpToIdIfFalse),
			new JumpToId(0, InstructionType.JumpEnd),
			new BinaryInstruction(InstructionType.Equal, Register.R2, Register.R3),
			new JumpToId(1, InstructionType.JumpToIdIfFalse),
			new JumpToId(1, InstructionType.JumpEnd),
			new ReturnInstruction(Register.R0)
		], 1).Count(s => s is JumpToId), Is.EqualTo(0));

	[Test]
	public void HandleEmptyInstructionList() => Optimize([], 0);
}