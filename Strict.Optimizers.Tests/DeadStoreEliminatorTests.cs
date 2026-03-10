using Strict.Runtime.Instructions;

namespace Strict.Optimizers.Tests;

public sealed class DeadStoreEliminatorTests : TestOptimizers
{
	[Test]
	public void RemoveUnusedVariable() =>
		Assert.That(((StoreVariableInstruction)Optimize([
			new StoreVariableInstruction(Num(5), "unused"),
			new StoreVariableInstruction(Num(10), "used"),
			new LoadVariableToRegister(Register.R0, "used"),
			new ReturnInstruction(Register.R0)
		], 3)[0]).Identifier, Is.EqualTo("used"));

	private List<Instruction> Optimize(List<Instruction> instructions, int expectedCount) =>
		Optimize(new DeadStoreEliminator(), instructions, expectedCount);

	[Test]
	public void KeepVariableThatIsLoaded() =>
		Optimize([
			new StoreVariableInstruction(Num(5), "x"),
			new LoadVariableToRegister(Register.R0, "x"),
			new ReturnInstruction(Register.R0)
		], 3);

	[Test]
	public void KeepMemberVariables() =>
		Optimize([
			new StoreVariableInstruction(Num(5), "member", isMember: true),
			new LoadConstantInstruction(Register.R0, Num(10)),
			new ReturnInstruction(Register.R0)
		], 3);

	[Test]
	public void RemoveMultipleDeadStores() =>
		Assert.That(((StoreVariableInstruction)Optimize([
			new StoreVariableInstruction(Num(1), "dead1"),
			new StoreVariableInstruction(Num(2), "dead2"),
			new StoreVariableInstruction(Num(3), "alive"),
			new LoadVariableToRegister(Register.R0, "alive"),
			new ReturnInstruction(Register.R0)
		], 3)[0]).Identifier, Is.EqualTo("alive"));

	[Test]
	public void KeepStoreWhenVariableUsedInStoreFromRegister() =>
		Optimize([
			new StoreVariableInstruction(Num(0), "count"),
			new LoadVariableToRegister(Register.R0, "count"),
			new LoadConstantInstruction(Register.R1, Num(1)),
			new BinaryInstruction(InstructionType.Add, Register.R0, Register.R1, Register.R2),
			new StoreFromRegisterInstruction(Register.R2, "count"),
			new LoadVariableToRegister(Register.R3, "count"),
			new ReturnInstruction(Register.R3)
		], 7);

	[Test]
	public void DoNotRemoveEmptyList() => Optimize([], 0);
}