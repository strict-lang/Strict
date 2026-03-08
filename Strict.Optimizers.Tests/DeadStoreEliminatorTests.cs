using Strict.Runtime.Statements;
using Binary = Strict.Runtime.Statements.Binary;
using Return = Strict.Runtime.Statements.Return;

namespace Strict.Optimizers.Tests;

public sealed class DeadStoreEliminatorTests : TestOptimizers
{
	[Test]
	public void RemoveUnusedVariable() =>
		Assert.That(((StoreVariableStatement)Optimize([
			new StoreVariableStatement(Num(5), "unused"),
			new StoreVariableStatement(Num(10), "used"),
			new LoadVariableToRegister(Register.R0, "used"),
			new Return(Register.R0)
		], 3)[0]).Identifier, Is.EqualTo("used"));

	private List<Statement> Optimize(List<Statement> statements, int expectedCount) =>
		Optimize(new DeadStoreEliminator(), statements, expectedCount);

	[Test]
	public void KeepVariableThatIsLoaded() =>
		Optimize([
			new StoreVariableStatement(Num(5), "x"),
			new LoadVariableToRegister(Register.R0, "x"),
			new Return(Register.R0)
		], 3);

	[Test]
	public void KeepMemberVariables() =>
		Optimize([
			new StoreVariableStatement(Num(5), "member", isMember: true),
			new LoadConstantStatement(Register.R0, Num(10)),
			new Return(Register.R0)
		], 3);

	[Test]
	public void RemoveMultipleDeadStores() =>
		Assert.That(((StoreVariableStatement)Optimize([
			new StoreVariableStatement(Num(1), "dead1"),
			new StoreVariableStatement(Num(2), "dead2"),
			new StoreVariableStatement(Num(3), "alive"),
			new LoadVariableToRegister(Register.R0, "alive"),
			new Return(Register.R0)
		], 3)[0]).Identifier, Is.EqualTo("alive"));

	[Test]
	public void KeepStoreWhenVariableUsedInStoreFromRegister() =>
		Optimize([
			new StoreVariableStatement(Num(0), "count"),
			new LoadVariableToRegister(Register.R0, "count"),
			new LoadConstantStatement(Register.R1, Num(1)),
			new Binary(Instruction.Add, Register.R0, Register.R1, Register.R2),
			new StoreFromRegisterStatement(Register.R2, "count"),
			new LoadVariableToRegister(Register.R3, "count"),
			new Return(Register.R3)
		], 7);

	[Test]
	public void DoNotRemoveEmptyList() => Optimize([], 0);
}