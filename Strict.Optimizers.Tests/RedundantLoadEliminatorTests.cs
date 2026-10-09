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

	[Test]
	public void RemapInvokeArgumentOfEliminatedLoad()
	{
		var optimized = Optimize([
			new LoadVariableToRegister(Register.R0, "x"),
			new LoadVariableToRegister(Register.R1, "x"),
			new Invoke(Register.R2, new InvokeMethodInfo("Holder", "from", ["x"], "Holder", [Register.R1],
				null)),
			new ReturnInstruction(Register.R2)
		], 3);
		Assert.That(((Invoke)optimized[1]).MethodInfo.ArgumentRegisters, Is.EqualTo(new[] { Register.R0 }));
	}

	[Test]
	public void RemapFieldLoadOfEliminatedLoad()
	{
		var optimized = Optimize([
			new LoadVariableToRegister(Register.R0, "state"),
			new LoadVariableToRegister(Register.R1, "state"),
			new FieldLoadInstruction(Register.R2, Register.R1, "memory"),
			new ReturnInstruction(Register.R2)
		], 3);
		Assert.That(((FieldLoadInstruction)optimized[1]).ObjectRegister, Is.EqualTo(Register.R0));
	}

	[Test]
	public void RemapWriteToListOfEliminatedLoad() =>
		Assert.That(((WriteToListInstruction)Optimize([
			new LoadVariableToRegister(Register.R0, "x"),
			new LoadVariableToRegister(Register.R1, "x"),
			new WriteToListInstruction(Register.R1, "items"),
			new ReturnInstruction(Register.R0)
		], 3)[1]).Register, Is.EqualTo(Register.R0));

	[Test]
	public void RemapOnlyReadsAfterEliminatedLoadWhenRegisterIsReused()
	{
		var optimized = Optimize([
			new LoadVariableToRegister(Register.R0, "x"),
			new LoadConstantInstruction(Register.R1, Num(2)),
			new BinaryInstruction(InstructionType.Add, Register.R1, Register.R0, Register.R2),
			new LoadVariableToRegister(Register.R1, "x"),
			new BinaryInstruction(InstructionType.Add, Register.R2, Register.R1, Register.R3),
			new ReturnInstruction(Register.R3)
		], 5);
		Assert.That(((BinaryInstruction)optimized[2]).Registers,
			Is.EqualTo(new[] { Register.R1, Register.R0, Register.R2 }));
		Assert.That(((BinaryInstruction)optimized[3]).Registers,
			Is.EqualTo(new[] { Register.R2, Register.R0, Register.R3 }));
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