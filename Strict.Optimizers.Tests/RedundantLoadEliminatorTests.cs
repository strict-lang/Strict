using Strict.Bytecode.Instructions;

namespace Strict.Optimizers.Tests;

public sealed class RedundantLoadEliminatorTests : TestOptimizers
{
	[Test]
	public void EliminateDuplicateLoadOfSameVariable()
	{
		var optimized = Optimize([
			new LoadVariableToRegister(Register.R0, "x"),
			new LoadVariableToRegister(Register.R1, "x"),
			new BinaryInstruction(InstructionType.Add, Register.R0, Register.R1, Register.R2),
			new ReturnInstruction(Register.R2)
		], 3);
		Assert.That(optimized[0], Is.InstanceOf<LoadVariableToRegister>());
		Assert.That(((BinaryInstruction)optimized[1]).Registers,
			Is.EqualTo(new[] { Register.R0, Register.R0, Register.R2 }));
	}

	private List<Instruction> Optimize(List<Instruction> instructions, int expectedCount) =>
		Optimize(new RedundantLoadEliminator(), instructions, expectedCount);

	[Test]
	public void DoNotEliminateLoadAfterStoreToSameVariable() =>
		Optimize([
			new LoadVariableToRegister(Register.R0, "x"),
			new StoreFromRegisterInstruction(Register.R1, "x"),
			new LoadVariableToRegister(Register.R2, "x"),
			new ReturnInstruction(Register.R2)
		], 4);

	[Test]
	public void KeepLoadsOfDifferentVariables() =>
		Optimize([
			new LoadVariableToRegister(Register.R0, "x"),
			new LoadVariableToRegister(Register.R1, "y"),
			new BinaryInstruction(InstructionType.Add, Register.R0, Register.R1, Register.R2),
			new ReturnInstruction(Register.R2)
		], 4);

	[Test]
	public void DoNotEliminateLoadAfterLoopBegin() =>
		Optimize([
			new LoadVariableToRegister(Register.R0, "x"),
			new LoopBeginInstruction(Register.R0),
			new LoadVariableToRegister(Register.R1, "x"),
			new LoopEndInstruction(3),
			new ReturnInstruction(Register.R0)
		], 5);

	[Test]
	public void DoNotEliminateWhenNoRedundancy() =>
		Optimize([
			new StoreVariableInstruction(Num(5), "x"),
			new LoadVariableToRegister(Register.R0, "x"),
			new ReturnInstruction(Register.R0)
		], 3);
}