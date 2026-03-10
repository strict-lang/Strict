using Strict.Runtime.Instructions;

namespace Strict.Optimizers.Tests;

/// <summary>
/// Remove passed test code and provably dead instructions, Tests are executed once; passing
/// expressions become true and are pruned. In Strict, test assertions are comparison + conditional
/// jump blocks at the start of a method. After tests pass, these blocks are dead code and removed.
/// </summary>
public sealed class TestCodeRemoverTests : TestOptimizers
{
	private List<Instruction> Optimize(List<Instruction> instructions, int expectedCount) =>
		Optimize(new TestCodeRemover(), instructions, expectedCount);

	[Test]
	public void RemovePassedTestAssertionPattern()
	{
		var optimized = Optimize([
			new LoadConstantInstruction(Register.R0, Num(5)),
			new LoadConstantInstruction(Register.R1, Num(5)),
			new BinaryInstruction(InstructionType.Equal, Register.R0, Register.R1),
			new JumpToId(InstructionType.JumpToIdIfFalse, 0),
			new JumpToId(InstructionType.JumpEnd, 0),
			new LoadVariableToRegister(Register.R2, "x"),
			new ReturnInstruction(Register.R2)
		], 2);
		Assert.That(optimized[0], Is.InstanceOf<LoadVariableToRegister>());
		Assert.That(optimized[1], Is.InstanceOf<ReturnInstruction>());
	}

	[Test]
	public void RemoveMultiplePassedTestAssertions() =>
		Optimize([
			new LoadConstantInstruction(Register.R0, Num(5)),
			new LoadConstantInstruction(Register.R1, Num(5)),
			new BinaryInstruction(InstructionType.Equal, Register.R0, Register.R1),
			new JumpToId(InstructionType.JumpToIdIfFalse, 0),
			new JumpToId(InstructionType.JumpEnd, 0),
			new LoadConstantInstruction(Register.R2, Num(10)),
			new LoadConstantInstruction(Register.R3, Num(10)),
			new BinaryInstruction(InstructionType.Equal, Register.R2, Register.R3),
			new JumpToId(InstructionType.JumpToIdIfFalse, 1),
			new JumpToId(InstructionType.JumpEnd, 1),
			new LoadVariableToRegister(Register.R4, "result"),
			new ReturnInstruction(Register.R4)
		], 2);

	[Test]
	public void DoNotRemoveTestWithMismatchedConstants() =>
		Optimize([
			new LoadConstantInstruction(Register.R0, Num(5)),
			new LoadConstantInstruction(Register.R1, Num(3)),
			new BinaryInstruction(InstructionType.Equal, Register.R0, Register.R1),
			new JumpToId(InstructionType.JumpToIdIfFalse, 0),
			new JumpToId(InstructionType.JumpEnd, 0),
			new LoadVariableToRegister(Register.R2, "x"),
			new ReturnInstruction(Register.R2)
		], 7);

	[Test]
	public void DoNotRemoveConditionalBlockWithBody() =>
		Optimize([
			new LoadConstantInstruction(Register.R0, Num(5)),
			new LoadConstantInstruction(Register.R1, Num(5)),
			new BinaryInstruction(InstructionType.Equal, Register.R0, Register.R1),
			new JumpToId(InstructionType.JumpToIdIfFalse, 0),
			new LoadConstantInstruction(Register.R2, Num(99)),
			new ReturnInstruction(Register.R2),
			new JumpToId(InstructionType.JumpEnd, 0),
			new ReturnInstruction(Register.R0)
		], 8);

	[Test]
	public void DoNotRemoveWhenVariablesInvolvedInComparison() =>
		Optimize([
			new LoadVariableToRegister(Register.R0, "x"),
			new LoadConstantInstruction(Register.R1, Num(5)),
			new BinaryInstruction(InstructionType.Equal, Register.R0, Register.R1),
			new JumpToId(InstructionType.JumpToIdIfFalse, 0),
			new JumpToId(InstructionType.JumpEnd, 0),
			new ReturnInstruction(Register.R0)
		], 6);

	[Test]
	public void HandleEmptyInstructionList() => Optimize([], 0);
}