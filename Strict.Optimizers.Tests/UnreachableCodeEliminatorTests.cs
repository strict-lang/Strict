using Strict.Runtime.Statements;
using Binary = Strict.Runtime.Statements.Binary;
using Return = Strict.Runtime.Statements.Return;

namespace Strict.Optimizers.Tests;

public sealed class UnreachableCodeEliminatorTests : TestOptimizers
{
	[Test]
	public void RemoveStatementsAfterUnconditionalReturn() =>
		Assert.That(Optimize([
			new LoadConstantStatement(Register.R0, Num(5)),
			new Return(Register.R0),
			new LoadConstantStatement(Register.R1, Num(10)),
			new Return(Register.R1)
		], 2)[1], Is.InstanceOf<Return>());

	private List<Statement> Optimize(List<Statement> statements, int expectedCount) =>
		Optimize(new UnreachableCodeEliminator(), statements, expectedCount);

	[Test]
	public void KeepAllStatementsWhenNoDeadCodeExists() =>
		Optimize([
			new LoadConstantStatement(Register.R0, Num(5)),
			new LoadConstantStatement(Register.R1, Num(3)),
			new Binary(Instruction.Add, Register.R0, Register.R1, Register.R2),
			new Return(Register.R2)
		], 4);

	[Test]
	public void DoNotRemoveCodeAfterConditionalJump() =>
		Optimize([
			new LoadConstantStatement(Register.R0, Num(1)),
			new LoadConstantStatement(Register.R1, Num(1)),
			new Binary(Instruction.Equal, Register.R0, Register.R1),
			new JumpToId(Instruction.JumpToIdIfFalse, 0),
			new LoadConstantStatement(Register.R2, Num(5)),
			new Return(Register.R2),
			new JumpToId(Instruction.JumpEnd, 0),
			new LoadConstantStatement(Register.R3, Num(10)),
			new Return(Register.R3)
		], 9);

	[Test]
	public void RemoveStatementsAfterReturnInsideConditionalBlock() =>
		Optimize([
			new LoadConstantStatement(Register.R0, Num(5)),
			new Return(Register.R0),
			new JumpToId(Instruction.JumpEnd, 0),
			new LoadConstantStatement(Register.R1, Num(10)),
			new Return(Register.R1)
		], 2);

	[Test]
	public void HandleEmptyStatementList() => Optimize([], 0);

	[Test]
	public void PreserveCodeInsideLoop() =>
		Optimize([
			new StoreVariableStatement(Num(0), "sum"),
			new LoadVariableToRegister(Register.R0, "numbers"),
			new LoopBeginStatement(Register.R0),
			new LoadVariableToRegister(Register.R1, "sum"),
			new LoadConstantStatement(Register.R2, Num(1)),
			new Binary(Instruction.Add, Register.R1, Register.R2, Register.R3),
			new StoreFromRegisterStatement(Register.R3, "sum"),
			new LoopEndStatement(6),
			new LoadVariableToRegister(Register.R4, "sum"),
			new Return(Register.R4)
		], 10);
}