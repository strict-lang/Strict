using Strict.Runtime.Statements;
using Binary = Strict.Runtime.Statements.Binary;
using Return = Strict.Runtime.Statements.Return;

namespace Strict.Optimizers.Tests;

public sealed class AllInstructionOptimizersTests : TestOptimizers
{
	private List<Statement> Optimize(List<Statement> statements, int expectedCount) =>
		Optimize(new AllInstructionOptimizers(), statements, expectedCount);

	[Test]
	public void ChainsMultipleOptimizers()
	{
		var optimized = Optimize([
			new StoreVariableStatement(Num(99), "unused"),
			new LoadConstantStatement(Register.R0, Num(2)),
			new LoadConstantStatement(Register.R1, Num(3)),
			new Binary(Instruction.Add, Register.R0, Register.R1, Register.R2),
			new Return(Register.R2)
		], 2);
		Assert.That(((LoadConstantStatement)optimized[0]).ValueInstance.Number, Is.EqualTo(5));
		Assert.That(optimized[1], Is.InstanceOf<Return>());
	}

	[Test]
	public void OptimizeConstantAssignmentExpression() =>
		Optimize([
			new StoreVariableStatement(Num(5), "number"),
			new StoreVariableStatement(Num(5), "five"),
			new LoadVariableToRegister(Register.R0, "five"),
			new LoadConstantStatement(Register.R1, Num(5)),
			new Binary(Instruction.Add, Register.R0, Register.R1, Register.R2),
			new StoreFromRegisterStatement(Register.R2, "something"),
			new LoadVariableToRegister(Register.R3, "something"),
			new LoadConstantStatement(Register.R4, Num(10)),
			new Binary(Instruction.Add, Register.R3, Register.R4, Register.R5),
			new Return(Register.R5)
		], 9);

	[Test]
	public void PreserveLoopStatements()
	{
		var optimized = Optimize([
			new StoreVariableStatement(Num(10), "number"),
			new StoreVariableStatement(Num(1), "result"),
			new StoreVariableStatement(Num(2), "multiplier"),
			new LoadVariableToRegister(Register.R0, "number"),
			new LoopBeginStatement(Register.R0),
			new LoadVariableToRegister(Register.R1, "result"),
			new LoadVariableToRegister(Register.R2, "multiplier"),
			new Binary(Instruction.Multiply, Register.R1, Register.R2, Register.R3),
			new StoreFromRegisterStatement(Register.R3, "result"),
			new LoopEndStatement(7),
			new LoadVariableToRegister(Register.R4, "result"),
			new Return(Register.R4)
		], 12);
		Assert.That(optimized.Any(s => s is LoopBeginStatement));
		Assert.That(optimized.Any(s => s is LoopEndStatement));
	}

	[Test]
	public void OptimizeWithRedundantLoads() =>
		Assert.That(((Binary)Optimize([
			new StoreVariableStatement(Num(5), "x"),
			new LoadVariableToRegister(Register.R0, "x"),
			new LoadVariableToRegister(Register.R1, "x"),
			new Binary(Instruction.Add, Register.R0, Register.R1, Register.R2),
			new Return(Register.R2)
		], 4)[2]).Registers, Is.EqualTo(new[] { Register.R0, Register.R0, Register.R2 }));

	[Test]
	public void OptimizedStatementsExecuteCorrectly() =>
		Assert.That(new BytecodeInterpreter(TestPackage.Instance).Execute(Optimize([
			new LoadConstantStatement(Register.R0, Num(10)),
			new LoadConstantStatement(Register.R1, Num(5)),
			new Binary(Instruction.Add, Register.R0, Register.R1, Register.R2),
			new Return(Register.R2)
		], 2)).Returns!.Value.Number, Is.EqualTo(15));

	[Test]
	public void OptimizedMultiplicationExecutesCorrectly() =>
		Assert.That(new BytecodeInterpreter(TestPackage.Instance).Execute(Optimize([
			new LoadConstantStatement(Register.R0, Num(4)),
			new LoadConstantStatement(Register.R1, Num(3)),
			new Binary(Instruction.Multiply, Register.R0, Register.R1, Register.R2),
			new LoadConstantStatement(Register.R3, Num(2)),
			new Binary(Instruction.Add, Register.R2, Register.R3, Register.R4),
			new Return(Register.R4)
		], 2)).Returns!.Value.Number, Is.EqualTo(14));

	[Test]
	public void EmptyListRemainsEmpty() => Optimize([], 0);

	[Test]
	public void PipelineRemovesPassedTestsThenFoldsConstants() =>
		Assert.That(((LoadConstantStatement)Optimize([
			new LoadConstantStatement(Register.R0, Num(5)),
			new LoadConstantStatement(Register.R1, Num(5)),
			new Binary(Instruction.Equal, Register.R0, Register.R1),
			new JumpToId(Instruction.JumpToIdIfFalse, 0),
			new JumpToId(Instruction.JumpEnd, 0),
			new LoadConstantStatement(Register.R2, Num(2)),
			new LoadConstantStatement(Register.R3, Num(3)),
			new Binary(Instruction.Add, Register.R2, Register.R3, Register.R4),
			new Return(Register.R4)
		], 2)[0]).ValueInstance.Number, Is.EqualTo(5));

	[Test]
	public void PipelineReducesStrengthAndEliminatesDeadStores()
	{
		var optimized = Optimize([
			new StoreVariableStatement(Num(42), "unused"),
			new LoadVariableToRegister(Register.R0, "x"),
			new LoadConstantStatement(Register.R1, Num(1)),
			new Binary(Instruction.Multiply, Register.R0, Register.R1, Register.R2),
			new Return(Register.R2)
		], 2);
		Assert.That(optimized.Any(s => s is LoadVariableToRegister));
		Assert.That(optimized[^1], Is.InstanceOf<Return>());
	}

	[Test]
	public void PipelineRemovesUnreachableCodeAfterFolding() =>
		Assert.That(((LoadConstantStatement)Optimize([
			new LoadConstantStatement(Register.R0, Num(5)),
			new LoadConstantStatement(Register.R1, Num(3)),
			new Binary(Instruction.Add, Register.R0, Register.R1, Register.R2),
			new Return(Register.R2),
			new LoadConstantStatement(Register.R3, Num(999)),
			new Return(Register.R3)
		], 2)[0]).ValueInstance.Number, Is.EqualTo(8));

	[Test]
	public void PipelineHandlesComplexMethodWithTestsAndIdentity() =>
		Assert.That(Optimize([
			new LoadConstantStatement(Register.R0, Num(10)),
			new LoadConstantStatement(Register.R1, Num(10)),
			new Binary(Instruction.Equal, Register.R0, Register.R1),
			new JumpToId(Instruction.JumpToIdIfFalse, 0),
			new JumpToId(Instruction.JumpEnd, 0),
			new StoreVariableStatement(Num(5), "x"),
			new LoadVariableToRegister(Register.R2, "x"),
			new LoadConstantStatement(Register.R3, Num(0)),
			new Binary(Instruction.Add, Register.R2, Register.R3, Register.R4),
			new Return(Register.R4)
		], 3)[^1], Is.InstanceOf<Return>());
}