using Strict.Runtime.Instructions;

namespace Strict.Optimizers.Tests;

public sealed class UnreachableCodeEliminatorTests : TestOptimizers
{
	[Test]
	public void RemoveInstructionsAfterUnconditionalReturn() =>
		Assert.That(Optimize([
			new LoadConstantInstruction(Register.R0, Num(5)),
			new ReturnInstruction(Register.R0),
			new LoadConstantInstruction(Register.R1, Num(10)),
			new ReturnInstruction(Register.R1)
		], 2)[1], Is.InstanceOf<ReturnInstruction>());

	private List<Instruction> Optimize(List<Instruction> instructions, int expectedCount) =>
		Optimize(new UnreachableCodeEliminator(), instructions, expectedCount);

	[Test]
	public void KeepAllInstructionsWhenNoDeadCodeExists() =>
		Optimize([
			new LoadConstantInstruction(Register.R0, Num(5)),
			new LoadConstantInstruction(Register.R1, Num(3)),
			new BinaryInstruction(InstructionType.Add, Register.R0, Register.R1, Register.R2),
			new ReturnInstruction(Register.R2)
		], 4);

	[Test]
	public void DoNotRemoveCodeAfterConditionalJump() =>
		Optimize([
			new LoadConstantInstruction(Register.R0, Num(1)),
			new LoadConstantInstruction(Register.R1, Num(1)),
			new BinaryInstruction(InstructionType.Equal, Register.R0, Register.R1),
			new JumpToId(InstructionType.JumpToIdIfFalse, 0),
			new LoadConstantInstruction(Register.R2, Num(5)),
			new ReturnInstruction(Register.R2),
			new JumpToId(InstructionType.JumpEnd, 0),
			new LoadConstantInstruction(Register.R3, Num(10)),
			new ReturnInstruction(Register.R3)
		], 9);

	[Test]
	public void RemoveInstructionsAfterReturnInsideConditionalBlock() =>
		Optimize([
			new LoadConstantInstruction(Register.R0, Num(5)),
			new ReturnInstruction(Register.R0),
			new JumpToId(InstructionType.JumpEnd, 0),
			new LoadConstantInstruction(Register.R1, Num(10)),
			new ReturnInstruction(Register.R1)
		], 2);

	[Test]
	public void HandleEmptyInstructionList() => Optimize([], 0);

	[Test]
	public void PreserveCodeInsideLoop() =>
		Optimize([
			new StoreVariableInstruction(Num(0), "sum"),
			new LoadVariableToRegister(Register.R0, "numbers"),
			new LoopBeginInstruction(Register.R0),
			new LoadVariableToRegister(Register.R1, "sum"),
			new LoadConstantInstruction(Register.R2, Num(1)),
			new BinaryInstruction(InstructionType.Add, Register.R1, Register.R2, Register.R3),
			new StoreFromRegisterInstruction(Register.R3, "sum"),
			new LoopEndInstruction(6),
			new LoadVariableToRegister(Register.R4, "sum"),
			new ReturnInstruction(Register.R4)
		], 10);
}