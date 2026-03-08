using Strict.Runtime.Statements;
using Binary = Strict.Runtime.Statements.Binary;
using Return = Strict.Runtime.Statements.Return;

namespace Strict.Optimizers.Tests;

/// <summary>
/// Remove passed test code and provably dead instructions, Tests are executed once; passing
/// expressions become true and are pruned. In Strict, test assertions are comparison + conditional
/// jump blocks at the start of a method. After tests pass, these blocks are dead code and removed.
/// </summary>
public sealed class TestCodeRemoverTests : TestOptimizers
{
	private List<Statement> Optimize(List<Statement> statements, int expectedCount) =>
		Optimize(new TestCodeRemover(), statements, expectedCount);

	[Test]
	public void RemovePassedTestAssertionPattern()
	{
		var optimized = Optimize([
			new LoadConstantStatement(Register.R0, Num(5)),
			new LoadConstantStatement(Register.R1, Num(5)),
			new Binary(Instruction.Equal, Register.R0, Register.R1),
			new JumpToId(Instruction.JumpToIdIfFalse, 0),
			new JumpToId(Instruction.JumpEnd, 0),
			new LoadVariableToRegister(Register.R2, "x"),
			new Return(Register.R2)
		], 2);
		Assert.That(optimized[0], Is.InstanceOf<LoadVariableToRegister>());
		Assert.That(optimized[1], Is.InstanceOf<Return>());
	}

	[Test]
	public void RemoveMultiplePassedTestAssertions() =>
		Optimize([
			new LoadConstantStatement(Register.R0, Num(5)),
			new LoadConstantStatement(Register.R1, Num(5)),
			new Binary(Instruction.Equal, Register.R0, Register.R1),
			new JumpToId(Instruction.JumpToIdIfFalse, 0),
			new JumpToId(Instruction.JumpEnd, 0),
			new LoadConstantStatement(Register.R2, Num(10)),
			new LoadConstantStatement(Register.R3, Num(10)),
			new Binary(Instruction.Equal, Register.R2, Register.R3),
			new JumpToId(Instruction.JumpToIdIfFalse, 1),
			new JumpToId(Instruction.JumpEnd, 1),
			new LoadVariableToRegister(Register.R4, "result"),
			new Return(Register.R4)
		], 2);

	[Test]
	public void DoNotRemoveTestWithMismatchedConstants() =>
		Optimize([
			new LoadConstantStatement(Register.R0, Num(5)),
			new LoadConstantStatement(Register.R1, Num(3)),
			new Binary(Instruction.Equal, Register.R0, Register.R1),
			new JumpToId(Instruction.JumpToIdIfFalse, 0),
			new JumpToId(Instruction.JumpEnd, 0),
			new LoadVariableToRegister(Register.R2, "x"),
			new Return(Register.R2)
		], 7);

	[Test]
	public void DoNotRemoveConditionalBlockWithBody() =>
		Optimize([
			new LoadConstantStatement(Register.R0, Num(5)),
			new LoadConstantStatement(Register.R1, Num(5)),
			new Binary(Instruction.Equal, Register.R0, Register.R1),
			new JumpToId(Instruction.JumpToIdIfFalse, 0),
			new LoadConstantStatement(Register.R2, Num(99)),
			new Return(Register.R2),
			new JumpToId(Instruction.JumpEnd, 0),
			new Return(Register.R0)
		], 8);

	[Test]
	public void DoNotRemoveWhenVariablesInvolvedInComparison() =>
		Optimize([
			new LoadVariableToRegister(Register.R0, "x"),
			new LoadConstantStatement(Register.R1, Num(5)),
			new Binary(Instruction.Equal, Register.R0, Register.R1),
			new JumpToId(Instruction.JumpToIdIfFalse, 0),
			new JumpToId(Instruction.JumpEnd, 0),
			new Return(Register.R0)
		], 6);

	[Test]
	public void HandleEmptyStatementList() => Optimize([], 0);
}