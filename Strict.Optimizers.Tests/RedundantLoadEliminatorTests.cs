using Strict.Runtime.Statements;
using Binary = Strict.Runtime.Statements.Binary;
using Return = Strict.Runtime.Statements.Return;

namespace Strict.Optimizers.Tests;

public sealed class RedundantLoadEliminatorTests : TestOptimizers
{
	[Test]
	public void EliminateDuplicateLoadOfSameVariable()
	{
		var optimized = Optimize([
			new LoadVariableToRegister(Register.R0, "x"),
			new LoadVariableToRegister(Register.R1, "x"),
			new Binary(Instruction.Add, Register.R0, Register.R1, Register.R2),
			new Return(Register.R2)
		], 3);
		Assert.That(optimized[0], Is.InstanceOf<LoadVariableToRegister>());
		Assert.That(((Binary)optimized[1]).Registers,
			Is.EqualTo(new[] { Register.R0, Register.R0, Register.R2 }));
	}

	private List<Statement> Optimize(List<Statement> statements, int expectedCount) =>
		Optimize(new RedundantLoadEliminator(), statements, expectedCount);

	[Test]
	public void DoNotEliminateLoadAfterStoreToSameVariable() =>
		Optimize([
			new LoadVariableToRegister(Register.R0, "x"),
			new StoreFromRegisterStatement(Register.R1, "x"),
			new LoadVariableToRegister(Register.R2, "x"),
			new Return(Register.R2)
		], 4);

	[Test]
	public void KeepLoadsOfDifferentVariables() =>
		Optimize([
			new LoadVariableToRegister(Register.R0, "x"),
			new LoadVariableToRegister(Register.R1, "y"),
			new Binary(Instruction.Add, Register.R0, Register.R1, Register.R2),
			new Return(Register.R2)
		], 4);

	[Test]
	public void DoNotEliminateLoadAfterLoopBegin() =>
		Optimize([
			new LoadVariableToRegister(Register.R0, "x"),
			new LoopBeginStatement(Register.R0),
			new LoadVariableToRegister(Register.R1, "x"),
			new LoopEndStatement(3),
			new Return(Register.R0)
		], 5);

	[Test]
	public void DoNotEliminateWhenNoRedundancy() =>
		Optimize([
			new StoreVariableStatement(Num(5), "x"),
			new LoadVariableToRegister(Register.R0, "x"),
			new Return(Register.R0)
		], 3);
}